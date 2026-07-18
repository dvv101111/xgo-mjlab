"""Benchmark an MJLab task's MuJoCo model with MJX on a local AMD GPU.

The existing MJLab simulator is MuJoCo Warp, whose GPU implementation is
CUDA-only.  This script builds the exact MJLab model on CPU, moves the physics
to MJX/JAX, and requires a ROCm device before doing any timed work.

V21B builds a 10x10 terrain atlas.  Compiling every mutually-exclusive tile
into one MJX collision graph is extremely slow, so the benchmark keeps the
selected tile's collision geoms and disables the other tiles.  This is also
the layout an MJX training environment should use: one active tile per
rollout, batched over environments.

Example:
  .venv/bin/python scripts/bench_mjx_rocm.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

import mjlab.tasks  # noqa: F401  (registry)
import src.tasks  # noqa: F401  (local task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="XGOLite-V21B")
  parser.add_argument("--row", type=int, default=9, help="terrain difficulty row")
  parser.add_argument("--col", type=int, default=2, help="terrain type column")
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--solver-iterations", type=int, default=4)
  parser.add_argument(
    "--cache-dir",
    default=os.path.expanduser("~/.cache/jax/luwu_mjlab"),
    help="persistent JAX compilation cache",
  )
  return parser.parse_args()


def _require_rocm() -> tuple[jax.Device, str]:
  devices = jax.devices()
  if len(devices) != 1:
    raise RuntimeError(f"Expected one local accelerator, found: {devices}")
  device = devices[0]
  platform_version = str(device.client.platform_version)
  if device.platform != "gpu" or "rocm" not in platform_version.lower():
    raise RuntimeError(
      "This benchmark requires the local ROCm JAX backend; "
      f"found device={device}, platform_version={platform_version!r}"
    )
  return device, platform_version


def _neutralize_manager_contact_sensors(model: mujoco.MjModel) -> int:
  """Replace unsupported manager contact sensors, not contact physics.

  MJX/JAX does not implement MuJoCo's body-matching contact-sensor semantics.
  An MJX environment reads ``data.contact`` directly instead.  CLOCK and
  SUBTREECOM preserve the original scalar/vector sensor storage layout while
  removing only the redundant sensor computations.
  """
  contact = model.sensor_type == mujoco.mjtSensor.mjSENS_CONTACT
  found = contact & (model.sensor_dim == 1)
  force = contact & (model.sensor_dim == 3)
  if int(found.sum() + force.sum()) != int(contact.sum()):
    raise RuntimeError("Unexpected V21B contact sensor dimensions")

  model.sensor_type[found] = mujoco.mjtSensor.mjSENS_CLOCK
  model.sensor_needstage[found] = mujoco.mjtStage.mjSTAGE_POS
  model.sensor_objtype[found] = mujoco.mjtObj.mjOBJ_UNKNOWN
  model.sensor_objid[found] = -1
  model.sensor_reftype[found] = mujoco.mjtObj.mjOBJ_UNKNOWN
  model.sensor_refid[found] = -1

  model.sensor_type[force] = mujoco.mjtSensor.mjSENS_SUBTREECOM
  model.sensor_needstage[force] = mujoco.mjtStage.mjSTAGE_POS
  model.sensor_objtype[force] = mujoco.mjtObj.mjOBJ_BODY
  model.sensor_objid[force] = 0
  model.sensor_reftype[force] = mujoco.mjtObj.mjOBJ_UNKNOWN
  model.sensor_refid[force] = -1
  return int(contact.sum())


def _select_terrain_tile(
  model: mujoco.MjModel,
  terrain_origins: np.ndarray,
  row: int,
  col: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Leave collision enabled only for geoms nearest the requested tile."""
  if not (0 <= row < terrain_origins.shape[0]):
    raise ValueError(f"row {row} outside [0, {terrain_origins.shape[0]})")
  if not (0 <= col < terrain_origins.shape[1]):
    raise ValueError(f"col {col} outside [0, {terrain_origins.shape[1]})")

  terrain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
  if terrain_id < 0:
    raise RuntimeError("Built task has no body named 'terrain'")
  geom_adr = int(model.body_geomadr[terrain_id])
  geom_num = int(model.body_geomnum[terrain_id])
  terrain_geom_ids = np.arange(geom_adr, geom_adr + geom_num, dtype=np.int32)

  flat_origins = terrain_origins[:, :, :2].reshape(-1, 2)
  geom_xy = model.geom_pos[terrain_geom_ids, :2]
  distances = np.square(geom_xy[:, None, :] - flat_origins[None, :, :]).sum(axis=2)
  target_index = np.ravel_multi_index((row, col), terrain_origins.shape[:2])
  selected = terrain_geom_ids[np.argmin(distances, axis=1) == target_index]
  if selected.size == 0:
    raise RuntimeError(f"No terrain geoms assigned to row={row}, col={col}")

  contype = model.geom_contype[selected].copy()
  conaffinity = model.geom_conaffinity[selected].copy()
  model.geom_contype[terrain_geom_ids] = 0
  model.geom_conaffinity[terrain_geom_ids] = 0
  model.geom_contype[selected] = contype
  model.geom_conaffinity[selected] = conaffinity
  return selected, terrain_origins[row, col].copy()


def _build_task_model(
  task: str,
  row: int,
  col: int,
  seed: int,
) -> tuple[mujoco.MjModel, np.ndarray, np.ndarray, int]:
  """Build one MJLab world on CPU and prepare its selected tile for MJX."""
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 1
  cfg.seed = seed
  if "actor" in cfg.observations:
    cfg.observations["actor"].enable_corruption = False

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
  try:
    model = env.sim.mj_model
    terrain_origins = env.scene.terrain.terrain_origins.detach().cpu().numpy().copy()
    robot = env.scene["robot"]
    root_state = robot.data.default_root_state[0].detach().cpu().numpy().copy()
    joint_pos = robot.data.default_joint_pos[0].detach().cpu().numpy().copy()
    selected, origin = _select_terrain_tile(model, terrain_origins, row, col)

    # The MJLab default state is the deployed stand pose.  Move it from its
    # local coordinates onto the selected terrain tile.
    model.qpos0[:7] = root_state[:7]
    model.qpos0[:3] += origin
    model.qpos0[7:] = joint_pos
    neutralized = _neutralize_manager_contact_sensors(model)
    return model, selected, joint_pos, neutralized
  finally:
    env.close()


def _geom_type_names(model: mujoco.MjModel, geom_ids: Sequence[int]) -> list[str]:
  return sorted({mujoco.mjtGeom(int(model.geom_type[i])).name for i in geom_ids})


def main() -> None:
  args = _parse_args()
  if args.batch_size <= 0 or args.steps <= 0:
    raise ValueError("--batch-size and --steps must be positive")

  os.makedirs(args.cache_dir, exist_ok=True)
  jax.config.update("jax_compilation_cache_dir", args.cache_dir)
  device, platform_version = _require_rocm()
  print(
    f"[ROCm] device={device.device_kind!r}, platform={platform_version!r}, "
    f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}",
    flush=True,
  )

  build_start = time.perf_counter()
  model, selected_geoms, stand_joint_pos, neutralized = _build_task_model(
    args.task, args.row, args.col, args.seed
  )
  build_s = time.perf_counter() - build_start
  model.opt.solver = mujoco.mjtSolver.mjSOL_CG
  model.opt.iterations = args.solver_iterations
  model.opt.ls_iterations = 4
  geom_types = _geom_type_names(model, selected_geoms)
  print(
    f"[MODEL] task={args.task}, tile=({args.row}, {args.col}), "
    f"active_terrain_geoms={len(selected_geoms)} {geom_types}, "
    f"ngeom={model.ngeom}, nhfield={model.nhfield}, "
    f"neutralized_contact_sensors={neutralized}, build={build_s:.2f}s",
    flush=True,
  )

  mjx_model = mjx.put_model(model)
  data0 = mjx.make_data(mjx_model).replace(ctrl=jp.asarray(stand_joint_pos))
  keys = jax.random.split(jax.random.key(args.seed), args.batch_size)
  data = jax.vmap(
    lambda key: data0.replace(
      qvel=0.01 * jax.random.normal(key, shape=(mjx_model.nv,))
    )
  )(keys)

  @jax.jit
  def rollout(batch_data):
    def step(batch_data, _):
      next_data = jax.vmap(mjx.step, in_axes=(None, 0))(mjx_model, batch_data)
      return next_data, None

    return jax.lax.scan(step, batch_data, None, length=args.steps)[0]

  compile_start = time.perf_counter()
  compiled_rollout = rollout.lower(data).compile()
  compile_s = time.perf_counter() - compile_start

  run_start = time.perf_counter()
  result = compiled_rollout(data)
  jax.block_until_ready(result.qpos)
  run_s = time.perf_counter() - run_start

  finite = bool(jp.isfinite(result.qpos).all() & jp.isfinite(result.qvel).all())
  if not finite:
    raise RuntimeError("MJX rollout produced a non-finite state")
  total_steps = args.batch_size * args.steps
  summary = {
    "task": args.task,
    "device": device.device_kind,
    "backend": "ROCm",
    "terrain_row": args.row,
    "terrain_col": args.col,
    "active_terrain_geoms": int(len(selected_geoms)),
    "terrain_geom_types": geom_types,
    "batch_size": args.batch_size,
    "steps_per_rollout": args.steps,
    "total_physics_steps": total_steps,
    "build_seconds": build_s,
    "compile_seconds": compile_s,
    "simulation_seconds": run_s,
    "physics_steps_per_second": total_steps / run_s,
    "realtime_factor": total_steps * model.opt.timestep / run_s,
    "finite": finite,
  }
  print("[RESULT] " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
