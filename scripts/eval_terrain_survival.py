"""Terrain survival evaluation for XGOLite-V21B checkpoints.

Pins the velocity command AND the terrain difficulty row, then measures per
(row, command) combo over a fixed horizon:

- survival: fraction of envs that never hit a fall termination (fell_over /
  illegal_contact) during the horizon (each env is counted dead at its
  FIRST termination; auto-resets respawn it on the same pinned row so the
  rest of the batch keeps running);
- frac_lin: mean linear-velocity tracking fraction, the same
  exp(-(err_xy^2 + 2 vz^2) / 0.10^2) kernel the training reward and the
  curriculum gates use (settle steps excluded);
- vx_ach / wz_ach: mean achieved body-frame velocities;
- a per-terrain-family survival breakdown (envs are evenly distributed
  across the 10 type columns, ~26 envs per family batch).

The TRAIN env cfg is used (full DR: friction low tail, compliance, slip
events, pushes, obs noise) with both curriculum terms removed so nothing
moves the pinned rows or command cells.

Usage:
  python scripts/eval_terrain_survival.py <checkpoint.pt> [task_id]

task_id defaults to XGOLite-V21B. Env overrides: EVAL_ROWS="0,3,6,9",
EVAL_ENVS=256, EVAL_STEPS=600 (12 s), EVAL_SETTLE=100.
"""

import os
import sys
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (registry)
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

CKPT = sys.argv[1]
TASK = sys.argv[2] if len(sys.argv) > 2 else "XGOLite-V21B"
NUM_ENVS = int(os.environ.get("EVAL_ENVS", "256"))
STEPS = int(os.environ.get("EVAL_STEPS", "600"))  # 12 s at 50 Hz
SETTLE_STEPS = int(os.environ.get("EVAL_SETTLE", "100"))
ROWS = [int(r) for r in os.environ.get("EVAL_ROWS", "0,3,6,9").split(",")]
TRACK_STD = 0.10  # the training tracking sigma (env_cfgs.py v16 note)

# v14 nominal body pose (pitch [rad], base height [m]).
NOMINAL_POSE = (0.0, 0.116)

# Pinned commands: mid-envelope forward (the terrain curriculum's bread and
# butter), envelope-edge forward, and a moderate turn.
COMMANDS = {
  "fwd_02": (0.2, 0.0, 0.0),
  "fwd_035": (0.35, 0.0, 0.0),
  "turn_06": (0.0, 0.0, 0.6),
}

# Column layout of xgolite_v21b_terrain_gen_cfg (cumulative proportions).
COL_FAMILY = {
  0: "flat", 1: "flat", 2: "rough", 3: "stairs", 4: "stairs",
  5: "stairs_inv", 6: "stairs_inv", 7: "slope", 8: "slope_inv", 9: "wave",
}

configure_torch_backends()
device = "cuda:0"

env_cfg = load_env_cfg(TASK, play=False)
env_cfg.scene.num_envs = NUM_ENVS
# No curriculum: command_grid asserts grid mode (disabled below for command
# pinning) and terrain_levels would move the pinned rows.
env_cfg.curriculum.pop("command_grid", None)
env_cfg.curriculum.pop("terrain_levels", None)
agent_cfg = load_rl_cfg(TASK)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

uenv = env.unwrapped
terrain = uenv.scene.terrain
assert terrain is not None and terrain.terrain_origins is not None, (
  f"Task '{TASK}' has no generator terrain; this eval needs one."
)
num_rows = terrain.terrain_origins.shape[0]

# Command pinning (eval_policy_buckets pattern): degenerate uniform ranges
# through the normal resample path; every sampler lottery off.
term = uenv.command_manager.get_term("twist")
term.cfg.axis_focus_probs = None
term.cfg.lateral_focus_prob = 0.0
term.cfg.pose_mode_probs = (1.0, 0.0, 0.0)
term.grid_enabled = False
term.cfg.rel_standing_envs = 0.0
term.is_standing_env[:] = False
term.cfg.ranges.body_pitch = NOMINAL_POSE[:1] * 2
term.cfg.ranges.base_height = (NOMINAL_POSE[1], NOMINAL_POSE[1])
term.cfg.nominal_pose = NOMINAL_POSE

families = sorted(set(COL_FAMILY.values()))
types = terrain.terrain_types  # [B], fixed for the whole eval

print(
  f"task {TASK}, ckpt {CKPT}\n"
  f"{NUM_ENVS} envs x {STEPS} steps per (row, command); full DR (friction "
  f"tail, compliance, slip, pushes, obs noise); settle {SETTLE_STEPS} steps "
  f"excluded from tracking; survival counts FIRST falls over the whole "
  f"horizon.\n"
)
header = (
  f"{'row':>3s} {'command':>8s} {'survival':>9s} {'frac_lin':>9s} "
  f"{'vx_ach':>7s} {'wz_ach':>7s}  "
  + " ".join(f"{f:>10s}" for f in families)
)
print(header)

for row in ROWS:
  assert 0 <= row < num_rows, f"row {row} outside 0..{num_rows - 1}"
  for cmd_name, (vx, vy, wz) in COMMANDS.items():
    term.cfg.ranges.lin_vel_x = (vx, vx)
    term.cfg.ranges.lin_vel_y = (vy, vy)
    term.cfg.ranges.ang_vel_z = (wz, wz)

    terrain.terrain_levels[:] = row
    terrain.env_origins[:] = terrain.terrain_origins[
      terrain.terrain_levels, terrain.terrain_types
    ]
    obs, _ = env.reset()
    # Reset re-samples per-episode DR only; re-pin nothing else moves rows.

    died = torch.zeros(NUM_ENVS, dtype=torch.bool, device=device)
    frac_sum = 0.0
    vx_sum = 0.0
    wz_sum = 0.0
    n_meas = 0
    cmd_xy = torch.tensor([vx, vy], device=device)
    with torch.no_grad():
      for step in range(STEPS):
        actions = policy(obs)
        obs, _, dones, extras = env.step(actions)
        tm = uenv.termination_manager
        died |= tm.get_term("fell_over") | tm.get_term("illegal_contact")
        if step < SETTLE_STEPS:
          continue
        robot = uenv.scene["robot"]
        lin_b = robot.data.root_link_lin_vel_b
        err = torch.sum(
          torch.square(cmd_xy - lin_b[:, :2]), dim=1
        ) + 2.0 * torch.square(lin_b[:, 2])
        frac_sum += float(torch.exp(-err / TRACK_STD**2).mean())
        vx_sum += float(lin_b[:, 0].mean())
        wz_sum += float(robot.data.root_link_ang_vel_b[:, 2].mean())
        n_meas += 1
    survival = 1.0 - float(died.float().mean())
    fam_surv = {}
    for col, fam in COL_FAMILY.items():
      mask = types == col
      if fam not in fam_surv:
        fam_surv[fam] = [0, 0]
      fam_surv[fam][0] += int((~died[mask]).sum())
      fam_surv[fam][1] += int(mask.sum())
    fam_str = " ".join(
      f"{fam_surv[f][0] / max(fam_surv[f][1], 1):10.2f}" for f in families
    )
    print(
      f"{row:3d} {cmd_name:>8s} {survival:9.3f} {frac_sum / n_meas:9.3f} "
      f"{vx_sum / n_meas:+7.3f} {wz_sum / n_meas:+7.3f}  {fam_str}"
    )

print(f"\nper-family columns show survival within that family's envs")
