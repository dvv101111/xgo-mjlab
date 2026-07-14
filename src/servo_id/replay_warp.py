"""GPU-batched (mujoco_warp) replay evaluator for servo system ID.

One warp "world" per work item (candidate parameter set x capture): a
whole CMA-ES generation — popsize x captures items — is one batched call,
chunked to ``max_envs`` worlds when larger. Semantics mirror
:mod:`src.servo_id.replay` exactly:

* same MJCF/MjModel (free base + ground plane, position actuators
  deleted), same XML sim options (Euler, Newton, pyramidal cone);
* torque via ``qfrc_applied``: the ToddlerBot PD + asymmetric
  torque-speed clamp is a warp kernel (:func:`_servo_pd_kernel`) fused
  into the step graph — no MuJoCo actuators, so no gainprm/biasprm
  plumbing and the torque law stays bit-identical to
  ``actuator_model.pd_clamped_torque``;
* per-world MJCF dof fields: ``dof_damping`` / ``dof_armature`` /
  ``dof_frictionloss`` are expanded to a leading world dim (mujoco_warp
  kernels index every model field ``[worldid % shape[0]]``; expansion via
  ``mjlab.sim.randomization.expand_model_fields``). Like the CPU path,
  derived ``*_invweight0`` constants are NOT recomputed after the
  armature write (both backends keep the compile-time values);
* per-candidate delay handled host-side: the ZOH command index per step
  (``replay.zoh_cmd_indices``, shared code) is precomputed per world with
  that candidate's delay and uploaded — no in-kernel time shifting;
* 1 s settle at the base pose, then the replay window from
  ``replay.capture_window`` (shared code); the loss is computed host-side
  by the SAME ``replay.capture_loss`` on the downloaded step-grid
  trajectories (SimTraj), so MSE + FFT terms are definitionally
  identical. Residual CPU-vs-warp differences come only from the physics
  engines themselves (fp32 vs fp64, batched Newton solver) — measured
  well under 1e-3 relative on the 2026-07-13 stand session (tolerance
  gate in tests/test_replay_warp.py: 2%).

Throughput (RTX 4090 Laptop, fl_hip steps+sine_grid = ~31 sim-seconds
per candidate): batched stepping is strongly sublinear in worlds — one
26.3k-step chunk costs 8.4 s at 8 worlds, 11.2 s at 128, 15.1 s at 512,
i.e. 0.175 s/candidate at 128 items and 0.059 s/candidate at 512. The
crossover vs the 24-worker fork pool (measured 7.3 s per 64-candidate
generation on the i9-13980HX) is ~256 items per call: prefer the CPU
backend for small popsize x captures products, warp for large ones
(shared-mode fits over many captures, and the future PPO-over-replay
training env).

Known semantic deltas vs the CPU path (relevant to a future PPO-over-
replay training env):
* mujoco_warp is float32 (CPU MuJoCo is float64);
* contact allocation is capped by ``nconmax``/``njmax`` heuristics from a
  representative standing pose;
* worlds whose capture is shorter than the longest in a chunk keep
  simulating (holding the last command) until the chunk finishes; the
  extra steps are never scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

from src.servo_id.actuator_model import ServoParams
from src.servo_id.fitting import Evaluator as _Evaluator
from src.servo_id.loader import TestCapture
from src.servo_id.replay import (
  LOSS_SKIP_MS,
  SETTLE_S,
  SIM_DT,
  ReplayModel,
  SimTraj,
  build_replay_model,
  capture_loss,
  capture_window,
  zoh_cmd_indices,
)

DEFAULT_MAX_ENVS = 128  # chunk cap; ~160 MB traj buffer on 50 s captures

_DOF_FIELDS = ("dof_damping", "dof_armature", "dof_frictionloss")


@dataclass
class WorkItem:
  """One rollout request: a capture under one candidate parameter set."""

  capture: TestCapture
  params: list[ServoParams]  # 12, canonical leg order
  delay_ms: float


@wp.kernel(enable_backward=False)
def _servo_pd_kernel(
  step_ctr: wp.array(dtype=wp.int32),
  n_settle: int,
  qadr: wp.array(dtype=wp.int32),
  dadr: wp.array(dtype=wp.int32),
  cap_of_env: wp.array(dtype=wp.int32),
  cmd_tables: wp.array3d(dtype=wp.float32),
  cmd_idx: wp.array2d(dtype=wp.int32),
  base: wp.array2d(dtype=wp.float32),
  kp: wp.array2d(dtype=wp.float32),
  kd: wp.array2d(dtype=wp.float32),
  deadband: wp.array2d(dtype=wp.float32),
  tau_max: wp.array2d(dtype=wp.float32),
  qd_knee: wp.array2d(dtype=wp.float32),
  qd_max: wp.array2d(dtype=wp.float32),
  qpos: wp.array2d(dtype=wp.float32),
  qvel: wp.array2d(dtype=wp.float32),
  qfrc_applied: wp.array2d(dtype=wp.float32),
  traj: wp.array3d(dtype=wp.float32),
):
  """Record q, then write the PD + torque-speed-clamp torque.

  Launched immediately BEFORE each physics step (same ordering as the
  CPU loop in ``replay.replay_capture``). ``step_ctr`` counts physics
  steps from the start of the settle phase; k = step - n_settle indexes
  the replay window. Mirrors ``actuator_model.pd_clamped_torque``.
  """
  w, j = wp.tid()
  i = step_ctr[0]
  k = i - n_settle
  q = qpos[w, qadr[j]]
  qd = qvel[w, dadr[j]]
  q_des = base[w, j]
  if k >= 0:
    traj[w, k, j] = q
    ci = cmd_idx[w, k]
    if ci >= 0:
      q_des = cmd_tables[cap_of_env[w], ci, j]
  tmax = tau_max[w, j]
  err = q_des - q
  db = deadband[w, j]
  # Lost-motion deadband: sign(err) * max(|err| - db, 0), clamp form
  # mirroring actuator_model.pd_clamped_torque expression-for-expression.
  err_eff = err - wp.clamp(err, -db, db)
  tau = kp[w, j] * err_eff - kd[w, j] * qd
  qd_abs = wp.abs(qd)
  knee = qd_knee[w, j]
  if qd_abs <= knee:
    line = tmax
  else:
    taper = tmax * (qd_max[w, j] - qd_abs) / wp.max(
      qd_max[w, j] - knee, 1e-9
    )
    line = wp.clamp(taper, 0.0, tmax)
  hi = tmax
  lo = -tmax
  if qd > 0.0:
    hi = line
  if qd < 0.0:
    lo = -line
  qfrc_applied[w, dadr[j]] = wp.clamp(tau, lo, hi)


@wp.kernel(enable_backward=False)
def _advance_kernel(step_ctr: wp.array(dtype=wp.int32)):
  step_ctr[0] = step_ctr[0] + 1


@dataclass
class _CaptureRec:
  """Host-side per-capture constants shared by all its work items."""

  index: int  # row in the cmd_tables buffer
  t_lo: float
  n_replay: int
  cmd_t: np.ndarray  # (n_cmd,) float64, firmware clock
  base12: np.ndarray  # (12,) float32


class WarpReplayBatch:
  """Persistent batched replay engine over a fixed set of captures.

  Allocates one mujoco_warp model/data pair with ``max_envs`` worlds and
  buffers sized for the LONGEST capture, captures a one-physics-step CUDA
  graph (controller kernel + mjwarp.step + counter advance), and replays
  it; per-chunk state (dof fields, gains, ZOH indices, initial pose) is
  written outside the graph, so the graph is captured exactly once.
  """

  def __init__(
    self,
    captures: list[TestCapture],
    rm: ReplayModel | None = None,
    max_envs: int = DEFAULT_MAX_ENVS,
    sim_dt: float = SIM_DT,
    settle_s: float = SETTLE_S,
    device: str | None = None,
    nconmax: int | None = None,
    njmax: int | None = None,
  ):
    import mujoco_warp as mjw
    from mjlab.sim.randomization import expand_model_fields

    self._mjw = mjw
    self.rm = rm or build_replay_model(fixed_base=False)
    if self.rm.fixed_base:
      raise NotImplementedError("warp backend supports the free-base "
                                "(loaded) regime only")
    self.max_envs = int(max_envs)
    self.sim_dt = float(sim_dt)
    self.dt_ms = self.sim_dt * 1000.0
    self.n_settle = int(round(settle_s / sim_dt))
    self.wp_device = wp.get_device(device) if device else wp.get_device()

    # Snapshot dof-field defaults NOW (a shared rm may be mutated later by
    # CPU-path set_joint_dynamics calls).
    model = self.rm.model
    self._dof_defaults = {
      f: np.asarray(getattr(model, f), dtype=np.float32).copy()
      for f in _DOF_FIELDS
    }

    # Per-capture constants (keyed by object identity; captures are the
    # loader's long-lived TestCapture instances).
    self._caps: dict[int, _CaptureRec] = {}
    n_cmd_max, self.n_replay_max = 1, 1
    for i, cap in enumerate(captures):
      t_lo, t_hi = capture_window(cap)
      n_replay = int(np.ceil((t_hi - t_lo) / self.dt_ms)) + 1
      self._caps[id(cap)] = _CaptureRec(
        index=i,
        t_lo=t_lo,
        n_replay=n_replay,
        cmd_t=np.asarray(cap.cmd_time_ms, dtype=float),
        base12=np.asarray(cap.base_pose[:12], dtype=np.float32),
      )
      n_cmd_max = max(n_cmd_max, cap.cmd_pose.shape[0])
      self.n_replay_max = max(self.n_replay_max, n_replay)

    # Representative standing state so put_data's nconmax/njmax heuristics
    # see realistic contact/constraint counts.
    mjd = mujoco.MjData(model)
    mjd.qpos[2] = 0.1159
    mjd.qpos[3] = 1.0
    if self._caps:
      first = captures[0]
      mjd.qpos[self.rm.qadr] = first.base_pose[:12]
    mujoco.mj_forward(model, mjd)

    with wp.ScopedDevice(self.wp_device):
      self._wm = mjw.put_model(model)
      self._wd = mjw.put_data(
        model, mjd, nworld=self.max_envs, nconmax=nconmax, njmax=njmax
      )
      expand_model_fields(self._wm, self.max_envs, list(_DOF_FIELDS))

      ne, nc = self.max_envs, max(len(captures), 1)
      f32, i32 = wp.float32, wp.int32
      self._step_ctr = wp.zeros(1, dtype=i32)
      self._qadr = wp.array(self.rm.qadr.astype(np.int32), dtype=i32)
      self._dadr = wp.array(self.rm.dadr.astype(np.int32), dtype=i32)
      self._cap_of_env = wp.zeros(ne, dtype=i32)
      self._cmd_tables = wp.zeros((nc, n_cmd_max, 12), dtype=f32)
      self._cmd_idx = wp.zeros((ne, self.n_replay_max), dtype=i32)
      self._base = wp.zeros((ne, 12), dtype=f32)
      self._gains = {
        n: wp.zeros((ne, 12), dtype=f32)
        for n in ("kp", "kd", "deadband", "tau_max", "qd_knee", "qd_max")
      }
      self._traj = wp.zeros((ne, self.n_replay_max, 12), dtype=f32)

      tables = np.zeros((nc, n_cmd_max, 12), dtype=np.float32)
      for cap in captures:
        rec = self._caps[id(cap)]
        tables[rec.index, : cap.cmd_pose.shape[0]] = cap.cmd_pose[:, :12]
      self._cmd_tables.assign(tables)

      # Warm up kernel compilation, then capture the one-step CUDA graph.
      self._graph = None
      self._launch_step()
      if self.wp_device.is_cuda:
        with wp.ScopedCapture() as capture:
          self._launch_step()
        self._graph = capture.graph

  def _launch_step(self) -> None:
    wp.launch(
      _servo_pd_kernel,
      dim=(self.max_envs, 12),
      inputs=[
        self._step_ctr, self.n_settle, self._qadr, self._dadr,
        self._cap_of_env, self._cmd_tables, self._cmd_idx, self._base,
        self._gains["kp"], self._gains["kd"], self._gains["deadband"],
        self._gains["tau_max"], self._gains["qd_knee"],
        self._gains["qd_max"],
        self._wd.qpos, self._wd.qvel, self._wd.qfrc_applied,
      ],
      outputs=[self._traj],
    )
    self._mjw.step(self._wm, self._wd)
    wp.launch(_advance_kernel, dim=1, inputs=[self._step_ctr])

  def rollout(self, items: list[WorkItem]) -> list[SimTraj]:
    """Roll out all work items; returns SimTrajs in input order.

    Items are sorted by replay length so each chunk of ``max_envs``
    worlds is near-homogeneous and short captures don't pay for long
    ones' step counts.
    """
    for it in items:
      if id(it.capture) not in self._caps:
        raise KeyError(
          f"capture {it.capture.path.name} was not registered with this "
          "WarpReplayBatch"
        )
    order = sorted(
      range(len(items)),
      key=lambda i: self._caps[id(items[i].capture)].n_replay,
      reverse=True,
    )
    results: list[SimTraj | None] = [None] * len(items)
    for lo in range(0, len(order), self.max_envs):
      self._run_chunk(items, order[lo : lo + self.max_envs], results)
    return results  # type: ignore[return-value]

  def _run_chunk(
    self,
    items: list[WorkItem],
    chunk: list[int],
    results: list[SimTraj | None],
  ) -> None:
    ne = self.max_envs
    model = self.rm.model
    nq, nv = model.nq, model.nv
    dadr = self.rm.dadr

    cap_of_env = np.zeros(ne, dtype=np.int32)
    base = np.zeros((ne, 12), dtype=np.float32)
    gains = {n: np.zeros((ne, 12), dtype=np.float32) for n in self._gains}
    # Idle worlds (chunk smaller than max_envs) replay row 0's setup —
    # cheap, always-valid physics whose output is never read.
    cmd_idx = np.full((ne, self.n_replay_max), -1, dtype=np.int32)
    dof = {f: np.tile(self._dof_defaults[f], (ne, 1)) for f in _DOF_FIELDS}
    qpos0 = np.zeros((ne, nq), dtype=np.float32)
    qpos0[:, 2] = 0.1159
    qpos0[:, 3] = 1.0

    n_replay_chunk = 1
    for e, item_i in enumerate(chunk):
      item = items[item_i]
      rec = self._caps[id(item.capture)]
      n_replay_chunk = max(n_replay_chunk, rec.n_replay)
      cap_of_env[e] = rec.index
      base[e] = rec.base12
      qpos0[e, self.rm.qadr] = rec.base12
      idx = zoh_cmd_indices(
        rec.cmd_t, rec.t_lo, rec.n_replay, self.dt_ms, item.delay_ms
      )
      cmd_idx[e, : rec.n_replay] = idx
      cmd_idx[e, rec.n_replay :] = idx[-1]  # short worlds hold last cmd
      p = item.params
      assert len(p) == 12
      for n in gains:
        gains[n][e] = [getattr(pj, n) for pj in p]
      dof["dof_damping"][e, dadr] = [pj.damping for pj in p]
      dof["dof_armature"][e, dadr] = [pj.armature for pj in p]
      dof["dof_frictionloss"][e, dadr] = [pj.frictionloss for pj in p]
    if len(chunk) < ne:  # idle worlds copy row 0 (valid params)
      for arr in (cap_of_env, base, qpos0, cmd_idx, *gains.values(),
                  *dof.values()):
        arr[len(chunk) :] = arr[0]

    with wp.ScopedDevice(self.wp_device):
      self._cap_of_env.assign(cap_of_env)
      self._base.assign(base)
      self._cmd_idx.assign(cmd_idx)
      for n, arr in gains.items():
        self._gains[n].assign(arr)
      for f in _DOF_FIELDS:
        getattr(self._wm, f).assign(dof[f].astype(np.float32))

      # Fresh start, like mj_resetData + explicit qpos on the CPU path.
      self._mjw.reset_data(self._wm, self._wd)
      self._wd.qpos.assign(qpos0)
      self._step_ctr.zero_()

      n_total = self.n_settle + n_replay_chunk
      if self._graph is not None:
        for _ in range(n_total):
          wp.capture_launch(self._graph)
      else:
        for _ in range(n_total):
          self._launch_step()
      traj = self._traj.numpy()  # implicit synchronize

    for e, item_i in enumerate(chunk):
      rec = self._caps[id(items[item_i].capture)]
      results[item_i] = SimTraj(
        t0_ms=rec.t_lo,
        dt_ms=self.dt_ms,
        q=traj[e, : rec.n_replay].copy(),
      )


def evaluate_items(
  batch: WarpReplayBatch,
  items: list[WorkItem],
  joints_per_item: list[tuple[str, ...]],
  skip_ms: float = LOSS_SKIP_MS,
) -> np.ndarray:
  """Batched rollout + the CPU path's capture_loss per item."""
  trajs = batch.rollout(items)
  return np.array(
    [
      capture_loss(traj, item.capture, joints, skip_ms=skip_ms)
      for traj, item, joints in zip(trajs, items, joints_per_item)
    ],
    dtype=float,
  )


class WarpEvaluator(_Evaluator):
  """``fitting.Evaluator`` on the warp backend.

  Same construction signature plus ``max_envs``/``device``; inherits the
  parameter plumbing (``params_from_values`` / ``values_from_vector``)
  and overrides evaluation to route through :class:`WarpReplayBatch`.
  The extra ``eval_population`` hook is what ``fitting.run_cma_fit``
  prefers over the fork pool: one CMA generation = one batched rollout
  of popsize x captures work items.
  """

  def __init__(
    self,
    captures: list[tuple[TestCapture, tuple[str, ...]]],
    cfg,
    target_joint: str | None,
    held_values: dict[str, dict[str, float]] | None = None,
    max_envs: int = DEFAULT_MAX_ENVS,
    device: str | None = None,
  ):
    if cfg.fixed_base:
      raise NotImplementedError("warp backend supports the free-base "
                                "(loaded) regime only")
    super().__init__(captures, cfg, target_joint, held_values=held_values)
    # self.rm is freshly compiled by the base class; WarpReplayBatch
    # snapshots its dof defaults before anything can mutate them.
    self.batch = WarpReplayBatch(
      [cap for cap, _ in captures],
      rm=self.rm,
      max_envs=max_envs,
      sim_dt=cfg.sim_dt,
      settle_s=cfg.settle_s,
      device=device,
    )

  def _eval_value_dicts(
    self, value_dicts: list[dict[str, float]]
  ) -> np.ndarray:
    items: list[WorkItem] = []
    joints_per_item: list[tuple[str, ...]] = []
    for values in value_dicts:
      params, delay = self.params_from_values(values)
      for capture, joints in self.captures:
        items.append(WorkItem(capture=capture, params=params,
                              delay_ms=delay))
        joints_per_item.append(joints)
    losses = evaluate_items(
      self.batch, items, joints_per_item, skip_ms=self.cfg.skip_ms
    ).reshape(len(value_dicts), len(self.captures))
    out = np.full(len(value_dicts), np.inf)
    for i, row in enumerate(losses):
      finite = row[np.isfinite(row)]
      if finite.size:
        out[i] = float(np.mean(finite))
    return out

  def eval_population(self, zs: list[np.ndarray]) -> np.ndarray:
    """One batched call for a whole CMA population."""
    return self._eval_value_dicts(
      [self.values_from_vector(z) for z in zs]
    )

  def eval_values(self, values: dict[str, float]) -> float:
    # Overrides the CPU replay loop; eval_vector inherits and lands here.
    return float(self._eval_value_dicts([values])[0])


def warp_available(device: str | None = None) -> bool:
  """True when a CUDA device is usable for the warp backend."""
  try:
    dev = wp.get_device(device) if device else wp.get_device("cuda:0")
  except Exception:
    return False
  return bool(dev.is_cuda)
