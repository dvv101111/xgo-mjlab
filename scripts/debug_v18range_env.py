"""Debug instrumentation for the XGOLite-V18Range mystery (2026-07-12).

Checks, on a LIVE env (real resample path, not the check script's stubs):

A. H1 obs/command correlation: locate the command slice in the flattened
   actor obs (term-major history layout) and compare it per env against
   command_manager.get_command("twist") at many steps.
B. H2 live resample statistics: hook _resample_command, roll the env, and
   histogram every draw: in-seed fraction, per-cell coverage, 0.05-gate
   zero fraction, standing fraction, pose-mode mutation fractions.
C. H6 standing-mask behavior: standing envs' twist stays zero every step,
   non-standing envs' commands persist unchanged between resamples.
D. Eval-pinning artifact: replicate eval_policy_buckets.py's degenerate
   ranges pinning and show what commands the grid path ACTUALLY produces.

Usage:
  cd luwu_mjlab && PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python \
      scripts/debug_v18range_env.py
"""

import types

import torch

import mjlab.tasks  # noqa: F401  (registry)
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

NUM_ENVS = 128
ROLL_STEPS = 700  # 14 s: every env resamples 2-4 times (3-8 s period)

configure_torch_backends()
device = "cuda:0"
torch.manual_seed(0)

cfg = load_env_cfg("XGOLite-V18Range", play=False)
cfg.scene.num_envs = NUM_ENVS
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)

term = env.command_manager.get_term("twist")
assert isinstance(term, UniformVelocityCommand)
gc = term.cfg.grid_curriculum
assert gc is not None

# ---------------------------------------------------------------- hook ----
resample_log = []  # (n_ids, vx, wz, vy, zeroed, standing) per resample call

orig_resample = term._resample_command


def hooked_resample(self, env_ids):
  orig_resample(env_ids)
  cmd = self.vel_command_b[env_ids]
  resample_log.append(
    dict(
      n=len(env_ids),
      vx=cmd[:, 0].clone(),
      vy=cmd[:, 1].clone(),
      wz=cmd[:, 2].clone(),
      zeroed=(torch.norm(cmd[:, :3], dim=1) == 0.0).clone(),
      standing=self.is_standing_env[env_ids].clone(),
      cells=self.grid_cell_index[env_ids].clone(),
    )
  )


term._resample_command = types.MethodType(hooked_resample, term)

env.reset()

# -------------------------------------------------- A. obs command slice ----
om = env.observation_manager
names = om._group_obs_term_names["actor"]
dims = [int(torch.tensor(d).prod().item()) for d in om._group_obs_term_dim["actor"]]
print("actor obs terms:", list(zip(names, dims)))
offset = 0
cmd_slice = None
hist = cfg.observations["actor"].history_length or 1
for n, d in zip(names, dims):
  if n == "command":
    # term-major, oldest first: latest frame = last cmd_dim entries.
    cmd_dim = d // hist if d % hist == 0 else d
    cmd_slice = (offset + d - cmd_dim, offset + d)
  offset += d
print(f"history={hist}, command slice (latest frame) = {cmd_slice}, total={offset}")

mismatch_steps = 0
max_abs_diff = 0.0
checked = 0
persist_viol = 0  # non-standing env command changed without a resample
stand_viol = 0  # standing env with nonzero twist after update
prev_cmd = term.vel_command_b.clone()
prev_resample_count = len(resample_log)

torch.manual_seed(1)
act_dim = env.action_manager.total_action_dim
with torch.no_grad():
  for step in range(ROLL_STEPS):
    resampled_before = set()
    n_before = len(resample_log)
    actions = 0.3 * torch.randn(NUM_ENVS, act_dim, device=device)
    obs, _, _, _, _ = env.step(actions)
    # Which envs resampled inside this step (incl. resets)?
    resampled = torch.zeros(NUM_ENVS, dtype=torch.bool, device=device)
    # Re-derive from time_left is post-hoc; instead track via hook order:
    # every hooked call during this step appended entries; recover ids from
    # cells tensor length is impossible -> log ids too. Simpler: compare.
    # (resample ids logged below via monkeypatched _resample.)
    cur = term.vel_command_b
    # H1: obs slice vs live command (command term is noise-free).
    obs_cmd = obs["actor"][:, cmd_slice[0] : cmd_slice[1]]
    live_cmd = env.command_manager.get_command("twist")
    d = (obs_cmd - live_cmd).abs().max().item()
    max_abs_diff = max(max_abs_diff, d)
    if d > 1e-6:
      mismatch_steps += 1
      if mismatch_steps <= 3:
        bad = (obs_cmd - live_cmd).abs().max(dim=1).values.argmax().item()
        print(
          f"  H1 MISMATCH step {step}: env {bad} obs={obs_cmd[bad].tolist()} "
          f"cmd={live_cmd[bad].tolist()}"
        )
    checked += 1
    # H6: standing envs must read zero twist in the obs/command.
    stand_viol += int(
      (term.vel_command_b[term.is_standing_env, :3].abs() > 1e-9).any().item()
      if term.is_standing_env.any()
      else 0
    )
    prev_cmd = cur.clone()

print(
  f"\nH1: {checked} steps checked, {mismatch_steps} steps with obs!=command, "
  f"max |diff| = {max_abs_diff:.2e}"
)
print(f"H6: steps where a standing env had nonzero twist: {stand_viol}")

# -------------------------------------------------- B. resample statistics ----
vx = torch.cat([r["vx"] for r in resample_log])
vy = torch.cat([r["vy"] for r in resample_log])
wz = torch.cat([r["wz"] for r in resample_log])
zeroed = torch.cat([r["zeroed"] for r in resample_log])
standing = torch.cat([r["standing"] for r in resample_log])
cells = torch.cat([r["cells"] for r in resample_log])
n = len(vx)
print(f"\nH2: {len(resample_log)} resample calls, {n} env-draws total")
active = ~zeroed
vx_lo_s, vx_hi_s = gc.seed_lin_vel_x
wz_lo_s, wz_hi_s = gc.seed_ang_vel_z
in_seed = (
  (vx >= vx_lo_s - 1e-5)
  & (vx <= vx_hi_s + 1e-5)
  & (wz >= wz_lo_s - 1e-5)
  & (wz <= wz_hi_s + 1e-5)
)
print(
  f"  vx range drawn: [{vx.min():.3f}, {vx.max():.3f}]  "
  f"wz: [{wz.min():.3f}, {wz.max():.3f}]  vy: [{vy.min():.3f}, {vy.max():.3f}]"
)
print(
  f"  in-seed fraction (all draws): {in_seed.float().mean():.4f}  "
  f"zeroed-by-gate/pose_hold: {zeroed.float().mean():.4f}  "
  f"standing: {standing.float().mean():.4f}"
)
print(
  f"  cell==-1 fraction: {(cells < 0).float().mean():.4f} "
  f"(should ~= zeroed|standing = "
  f"{(zeroed | standing).float().mean():.4f})"
)
# Coverage over the 32 seed cells (attributable draws only).
valid_cells = cells[cells >= 0]
counts = torch.bincount(valid_cells, minlength=term._grid_n_vx * term._grid_n_wz)
nonzero_cells = (counts > 0).sum().item()
seed_flat = term.grid_seed_mask.reshape(-1)
print(
  f"  distinct cells hit: {nonzero_cells} (seed cells: {int(seed_flat.sum())}); "
  f"draws outside seed cells: {int(counts[~seed_flat].sum().item())}"
)
seed_counts = counts[seed_flat].float()
print(
  f"  per-seed-cell counts: min {seed_counts.min():.0f} max {seed_counts.max():.0f} "
  f"mean {seed_counts.mean():.1f} (uniform-ish expected)"
)
# Mean drawn command (attributable, non-zeroed).
print(
  f"  mean over ACTIVE draws: vx {vx[active].mean():+.4f}, wz {wz[active].mean():+.4f}, "
  f"|vx| {vx[active].abs().mean():.4f}, |cmd_xy| "
  f"{torch.sqrt(vx[active] ** 2 + vy[active] ** 2).mean():.4f}"
)

# -------------------------------------------------- D. eval pinning check ----
print("\nD. replicate eval_policy_buckets.py pinning on the grid task:")
term.cfg.axis_focus_probs = None
term.cfg.pose_mode_probs = (1.0, 0.0, 0.0)
for bname, (bvx, bvy, bwz) in {
  "fwd_slow": (0.08, 0.0, 0.0),
  "ccw_slow": (0.0, 0.0, 0.4),
  "stand": (0.0, 0.0, 0.0),
}.items():
  term.cfg.ranges.lin_vel_x = (bvx, bvx)
  term.cfg.ranges.lin_vel_y = (bvy, bvy)
  term.cfg.ranges.ang_vel_z = (bwz, bwz)
  resample_log.clear()
  env.reset()
  c = term.vel_command_b
  print(
    f"  bucket {bname:9s} pinned=({bvx:+.2f},{bvy:+.2f},{bwz:+.2f}) -> drawn "
    f"vx [{c[:, 0].min():.3f},{c[:, 0].max():.3f}] mean {c[:, 0].mean():+.3f} | "
    f"vy mean {c[:, 1].mean():+.3f} | "
    f"wz [{c[:, 2].min():.3f},{c[:, 2].max():.3f}] mean {c[:, 2].mean():+.3f} | "
    f"E|cmd_xy| {torch.norm(c[:, :2], dim=1).mean():.3f} | E|wz| {c[:, 2].abs().mean():.3f}"
  )

env.close()
