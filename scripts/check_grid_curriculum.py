"""Validation for the grid-adaptive command curriculum (2026-07-12).

Builds a small XGOLite-V18Range env (grid mode on) and checks:

1. GRID GEOMETRY: 21 x 8 cells over vx (-0.45, 0.60) x wz (-1.0, 1.0),
   seed region = the 8 x 4 cell block whose centers lie in
   vx [-0.15, 0.25] x wz [-0.5, 0.5], weight 1.0 inside / 0.0 outside.
2. SEED SAMPLING: repeated resamples only ever produce (vx, wz) inside the
   seed box (post-mutation commands included: pose_hold zeroing and
   posed_walk halving stay inside a convex seed box containing 0), and the
   recorded cell attribution matches the cell containing the final command
   (-1 for zeroed-twist and standing envs).
3. WEIGHT BUMP: zeroing all weights except one far out-of-seed cell makes
   every subsequent draw land in that cell.
4. UNLOCK: synthetic "good episode" reward sums (0.9 of max attainable)
   pushed through ``command_grid_adaptive`` unlock the 4-connected
   neighbors of the attributed cell (+0.2, clipped at 1.0); sums below the
   thresholds, or cell index -1, change nothing. Logging dict sanity.
5. GRID OFF: XGOLite-Flat keeps grid_curriculum=None / grid_enabled=False,
   and the sampler is bit-identical to the pre-change code: the snapshot of
   velocity_command.py taken before this change is loaded side by side and
   both samplers are driven from identical RNG seeds on a stub env — every
   drawn command must match bitwise.

Usage:
  cd luwu_mjlab && PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python \
      scripts/check_grid_curriculum.py [path/to/velocity_command_before.py]
"""

import dataclasses
import sys
import types

import torch

import mjlab.tasks  # noqa: F401  (registry)
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.mdp.curriculums import command_grid_adaptive
from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

NUM_ENVS = 32
N_RESAMPLES = 40
EPS = 1e-5

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  status = "PASS" if ok else "FAIL"
  print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
  if not ok:
    _failures.append(name)


# ---------------------------------------------------------------- stubs ----


class _StubScene:
  def __getitem__(self, key):
    return None


class _StubEnv:
  """Just enough env surface for UniformVelocityCommand construction and
  _resample_command (robot is only touched when init_velocity_prob draws
  fire, which v17 keeps at 0)."""

  def __init__(self, num_envs: int, device: str):
    self.scene = _StubScene()
    self.num_envs = num_envs
    self.device = device


def load_before_module(path: str) -> types.ModuleType:
  """Load the pre-change velocity_command.py snapshot standalone (its
  relative import is rewritten to the absolute package path)."""
  with open(path) as f:
    source = f.read()
  source = source.replace(
    "from .rewards import body_pitch_from_gravity",
    "from src.tasks.velocity.mdp.rewards import body_pitch_from_gravity",
  )
  module = types.ModuleType("velocity_command_before")
  # dataclasses resolves annotations via sys.modules[cls.__module__].
  sys.modules[module.__name__] = module
  exec(compile(source, path, "exec"), module.__dict__)
  return module


def main() -> None:
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  torch.manual_seed(0)

  cfg = load_env_cfg("XGOLite-V18Range")
  cfg.scene.num_envs = NUM_ENVS
  cfg.events.pop("push_robot", None)
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  env.reset()

  term = env.command_manager.get_term("twist")
  assert isinstance(term, UniformVelocityCommand)
  gc = term.cfg.grid_curriculum
  assert gc is not None
  all_ids = torch.arange(env.num_envs, device=device)

  # -------------------------------------------------- check 1: geometry ----
  check("grid mode enabled on V18Range", term.grid_enabled)
  check(
    "grid is 21 x 8 cells (0.05 x 0.25 over the envelope)",
    (term._grid_n_vx, term._grid_n_wz) == (21, 8),
    f"got {term._grid_n_vx} x {term._grid_n_wz}",
  )
  n_seed = int(term.grid_seed_mask.sum().item())
  check("seed region is the 8 x 4 center-in-box block", n_seed == 32, f"{n_seed} cells")
  w = term.grid_weights
  check(
    "weights init: 1.0 in seed, 0.0 outside",
    bool(torch.equal(w, term.grid_seed_mask.float())),
    f"sum={w.sum().item():.1f}",
  )

  # ---------------------------------------------- check 2: seed sampling ----
  vx_lo_s, vx_hi_s = gc.seed_lin_vel_x
  wz_lo_s, wz_hi_s = gc.seed_ang_vel_z
  out_of_seed = 0
  bad_attrib = 0
  seen_cells: set[int] = set()
  for _ in range(N_RESAMPLES):
    term._resample(all_ids)
    vx = term.vel_command_b[:, 0]
    wz = term.vel_command_b[:, 2]
    out_of_seed += int(
      ((vx < vx_lo_s - EPS) | (vx > vx_hi_s + EPS) | (wz < wz_lo_s - EPS) | (wz > wz_hi_s + EPS))
      .sum()
      .item()
    )
    # Attribution: -1 iff twist zeroed or standing env, else containing cell.
    zeroed = torch.norm(term.vel_command_b[:, :3], dim=1) == 0.0
    invalid = zeroed | term.is_standing_env
    ix = torch.clamp(
      ((vx - term._grid_vx_lo) / term._grid_vx_size).floor().long(), 0, term._grid_n_vx - 1
    )
    iz = torch.clamp(
      ((wz - term._grid_wz_lo) / term._grid_wz_size).floor().long(), 0, term._grid_n_wz - 1
    )
    expected = torch.where(
      invalid, torch.full_like(ix, -1), ix * term._grid_n_wz + iz
    )
    bad_attrib += int((term.grid_cell_index != expected).sum().item())
    seen_cells.update(term.grid_cell_index[term.grid_cell_index >= 0].tolist())
  check(
    "all initial draws inside the seed region",
    out_of_seed == 0,
    f"{out_of_seed} of {N_RESAMPLES * NUM_ENVS} outside",
  )
  check("cell attribution matches the containing cell", bad_attrib == 0)
  check(
    "multiple distinct seed cells get sampled",
    len(seen_cells) >= 8,
    f"{len(seen_cells)} distinct cells",
  )

  # ------------------------------------------------ check 3: weight bump ----
  far_ix, far_iz = 20, 7  # vx [0.55, 0.60], wz [0.75, 1.00] — far out of seed
  far_cell = far_ix * term._grid_n_wz + far_iz
  saved_pose_probs = term.cfg.pose_mode_probs
  saved_standing = term.cfg.rel_standing_envs
  term.cfg.pose_mode_probs = (1.0, 0.0, 0.0)  # nominal mode: twist unmutated
  term.cfg.rel_standing_envs = 0.0
  term.grid_weights.zero_()
  term.grid_weights[far_ix, far_iz] = 1.0
  hits = 0
  total = 0
  for _ in range(5):
    term._resample(all_ids)
    vx = term.vel_command_b[:, 0]
    wz = term.vel_command_b[:, 2]
    hits += int(
      ((vx >= 0.55 - EPS) & (vx <= 0.60 + EPS) & (wz >= 0.75 - EPS) & (wz <= 1.00 + EPS))
      .sum()
      .item()
    )
    total += env.num_envs
  check(
    "bumped out-of-seed cell captures all draws",
    hits == total,
    f"{hits}/{total} draws in cell ({far_ix},{far_iz})",
  )
  check(
    "attribution follows the bumped cell",
    bool((term.grid_cell_index == far_cell).all()),
  )
  term.cfg.pose_mode_probs = saved_pose_probs
  term.cfg.rel_standing_envs = saved_standing
  term.grid_weights.copy_(term.grid_seed_mask.float())

  # ---------------------------------------------------- check 4: unlock ----
  rm = env.reward_manager
  dt = env.step_dt
  w_lin = rm.get_term_cfg("track_linear_velocity").weight
  w_ang = rm.get_term_cfg("track_angular_velocity").weight
  ids = torch.arange(6, device=device)
  steps = 200

  def set_episode(frac_lin: float, frac_ang: float, cell: int) -> None:
    env.episode_length_buf[ids] = steps
    rm._episode_sums["track_linear_velocity"][ids] = frac_lin * steps * dt * w_lin
    rm._episode_sums["track_angular_velocity"][ids] = frac_ang * steps * dt * w_ang
    term.grid_cell_index[ids] = cell

  # Seed-edge cell: vx cell 13 = [0.20, 0.25] (top of the vx seed band),
  # wz cell 4 = [0.00, 0.25]. Its +vx neighbor (14, 4) starts locked at 0.
  edge_cell = 13 * term._grid_n_wz + 4
  set_episode(0.9, 0.9, edge_cell)
  before = term.grid_weights.clone()
  state = command_grid_adaptive(env, ids, command_name="twist")
  after = term.grid_weights
  expected = before.clone()
  expected[14, 4] = 0.2  # unlocked neighbor (was 0.0)
  # (13,4), (12,4), (13,3), (13,5) are seed cells already at 1.0 -> clipped.
  check(
    "good episode unlocks the out-of-seed 4-neighbor (+0.2)",
    bool(torch.allclose(after, expected)),
    f"(14,4): {before[14, 4].item():.2f} -> {after[14, 4].item():.2f}",
  )
  check(
    "passed cell and in-seed neighbors stay clipped at 1.0",
    bool(
      after[13, 4].item() == 1.0
      and after[12, 4].item() == 1.0
      and after[13, 3].item() == 1.0
      and after[13, 5].item() == 1.0
    ),
  )
  keys_ok = set(state.keys()) == {
    "mean_cell_weight",
    "frac_cells_unlocked",
    "seed_tracking",
  }
  check(
    "curriculum term returns the logging dict",
    keys_ok
    and abs(state["seed_tracking"].item() - 0.9) < 1e-4
    and abs(state["frac_cells_unlocked"].item() - 32.0 / 168.0) < 1e-6,
    f"state={{{', '.join(f'{k}: {v.item():.4f}' for k, v in state.items())}}}",
  )

  # Below-threshold episode: no change.
  term.grid_weights.copy_(term.grid_seed_mask.float())
  set_episode(0.9, 0.5, edge_cell)  # angular below gamma_ang=0.7
  command_grid_adaptive(env, ids, command_name="twist")
  no_change_low = torch.equal(term.grid_weights, term.grid_seed_mask.float())
  # Invalid attribution (-1): no change even with perfect sums.
  set_episode(1.0, 1.0, -1)
  command_grid_adaptive(env, ids, command_name="twist")
  no_change_invalid = torch.equal(term.grid_weights, term.grid_seed_mask.float())
  check("below-threshold episode unlocks nothing", no_change_low)
  check("zeroed/standing episode (cell -1) unlocks nothing", no_change_invalid)
  rm._episode_sums["track_linear_velocity"][ids] = 0.0
  rm._episode_sums["track_angular_velocity"][ids] = 0.0
  env.episode_length_buf[ids] = 0

  env.close()
  del env

  # -------------------------------------------------- check 5: grid off ----
  flat_cfg = load_env_cfg("XGOLite-Flat")
  flat_twist = flat_cfg.commands["twist"]
  check(
    "XGOLite-Flat keeps grid_curriculum=None",
    getattr(flat_twist, "grid_curriculum", None) is None,
  )

  before_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/tmp/claude-1000/-home-dvv-dev-bots/42ed23ba-76f6-4267-92a5-1351254f6eba"
    "/scratchpad/velocity_command_before.py"
  )
  try:
    before_mod = load_before_module(before_path)
  except FileNotFoundError:
    print(
      f"[SKIP] pre-change snapshot not found at {before_path}; bit-identity "
      "established by code-path inspection only (all grid code is behind "
      "`if self.grid_enabled` and consumes no RNG when off)."
    )
    before_mod = None

  if before_mod is not None:
    # Drive the pre-change and current samplers from identical RNG streams
    # on the SAME flat (grid-off) cfg; every field they write must match
    # bitwise. The old class only reads cfg attributes, so passing the new
    # cfg instance (a strict superset) is valid.
    stub_ids = torch.arange(64)
    term_old = before_mod.UniformVelocityCommand(flat_twist, _StubEnv(64, "cpu"))
    term_new = UniformVelocityCommand(flat_twist, _StubEnv(64, "cpu"))
    check("flat sampler has grid disabled", not term_new.grid_enabled)
    mismatches = 0
    for i in range(20):
      torch.manual_seed(10_000 + i)
      term_old._resample_command(stub_ids)
      torch.manual_seed(10_000 + i)
      term_new._resample_command(stub_ids)
      if not (
        torch.equal(term_old.vel_command_b, term_new.vel_command_b)
        and torch.equal(term_old.is_standing_env, term_new.is_standing_env)
        and torch.equal(term_old.heading_target, term_new.heading_target)
      ):
        mismatches += 1
    check(
      "grid-off sampler bit-identical to pre-change snapshot (20 seeds)",
      mismatches == 0,
      f"{mismatches} mismatching resamples",
    )

  print()
  if _failures:
    print(f"OVERALL: FAIL ({len(_failures)} failed: {', '.join(_failures)})")
    raise SystemExit(1)
  print("OVERALL: PASS")


if __name__ == "__main__":
  main()
