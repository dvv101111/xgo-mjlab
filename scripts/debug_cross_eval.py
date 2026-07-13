"""2x2 cross-eval for the V18Range mystery: (policy x env) with PROPER pinning.

Unlike eval_policy_buckets.py (whose degenerate-ranges pinning is silently
ignored by the grid sampler), commands here are pinned by monkeypatching
_resample_command, so they hold in ANY task. Additionally:

- natural-distribution tracking (H3): roll each policy in its own env with
  the real sampler and regress achieved vs commanded (vx, wz) per step;
- curriculum-gate reachability: run ONE full 1000-step episode and compute
  the per-env episodic tracking fractions exactly like command_grid_adaptive
  does, then report the pass fraction at (gamma_lin 0.8, gamma_ang 0.7).

Usage:
  cd luwu_mjlab && PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python \
      scripts/debug_cross_eval.py
"""

import types
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (registry)
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

NUM_ENVS = 256
STEPS = 400
SETTLE = 100
NOMINAL_POSE = (0.0, 0.116)

CKPTS = {
  "v18range": "logs/rsl_rl/xgolite_v18range/2026-07-12_01-48-38_v18range_v1/model_2499.pt",
  "v18base": "logs/rsl_rl/xgolite_v18draft/2026-07-12_01-21-55_v18base_v1/model_1499.pt",
}
TASKS = {"V18Range-env": "XGOLite-V18Range", "V18Draft-env": "XGOLite-V18Draft"}

BUCKETS = {
  "fwd_slow": (0.08, 0.0, 0.0),
  "fwd": (0.20, 0.0, 0.0),
  "back": (-0.10, 0.0, 0.0),
  "ccw_slow": (0.0, 0.0, 0.4),
  "stand": (0.0, 0.0, 0.0),
}

configure_torch_backends()
device = "cuda:0"


def pin_commands(term: UniformVelocityCommand, pinned) -> None:
  vx, vy, wz = pinned

  def _pin(self, env_ids):
    self.vel_command_b[env_ids, 0] = vx
    self.vel_command_b[env_ids, 1] = vy
    self.vel_command_b[env_ids, 2] = wz
    if self.pose_enabled:
      self.vel_command_b[env_ids, 3] = NOMINAL_POSE[0]
      self.vel_command_b[env_ids, 4] = NOMINAL_POSE[1]
    self.is_standing_env[env_ids] = False
    if self.grid_enabled:
      self.grid_cell_index[env_ids] = -1

  term._resample_command = types.MethodType(_pin, term)


def frac_lin_step(cmd, robot):
  err = torch.sum(
    torch.square(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2]), dim=1
  ) + 2 * torch.square(robot.data.root_link_lin_vel_b[:, 2])
  return torch.exp(-err / 0.10**2)


def frac_ang_step(cmd, robot):
  err = torch.square(cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2]) + 0.05 * torch.sum(
    torch.square(robot.data.root_link_ang_vel_b[:, :2]), dim=1
  )
  std_eff = 0.2 + 0.4 * cmd[:, 2].abs()
  return torch.exp(-err / std_eff**2)


results = {}
natural = {}

for env_label, task in TASKS.items():
  env_cfg = load_env_cfg(task, play=False)
  env_cfg.scene.num_envs = NUM_ENVS
  agent_cfg = load_rl_cfg(task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  uenv = wenv.unwrapped
  term = uenv.command_manager.get_term("twist")
  assert isinstance(term, UniformVelocityCommand)
  orig_resample = term._resample_command
  robot = uenv.scene["robot"]

  for pol_label, ckpt in CKPTS.items():
    runner = runner_cls(wenv, asdict(agent_cfg), device=device)
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    # ---------------- pinned buckets ----------------
    for bname, pinned in BUCKETS.items():
      pin_commands(term, pinned)
      obs, _ = wenv.reset()
      sums = dict(flin=0.0, fang=0.0, err_xy=0.0, vx=0.0, vy=0.0, wz=0.0)
      falls = 0
      n = 0
      with torch.no_grad():
        for step in range(STEPS):
          obs, _, _, _ = wenv.step(policy(obs))
          tm = uenv.termination_manager
          falls += int(tm.get_term("fell_over").sum().item())
          falls += int(tm.get_term("illegal_contact").sum().item())
          if step < SETTLE:
            continue
          cmd = uenv.command_manager.get_command("twist")
          assert (cmd[:, 0] == pinned[0]).all() and (cmd[:, 2] == pinned[2]).all()
          sums["flin"] += frac_lin_step(cmd, robot).mean().item()
          sums["fang"] += frac_ang_step(cmd, robot).mean().item()
          lin_b = robot.data.root_link_lin_vel_b
          sums["err_xy"] += torch.norm(cmd[:, :2] - lin_b[:, :2], dim=1).mean().item()
          sums["vx"] += lin_b[:, 0].mean().item()
          sums["vy"] += lin_b[:, 1].mean().item()
          sums["wz"] += robot.data.root_link_ang_vel_b[:, 2].mean().item()
          n += 1
      row = {k: v / n for k, v in sums.items()}
      row["falls"] = falls
      results[(pol_label, env_label, bname)] = row
      print(
        f"[{pol_label:8s} x {env_label:12s}] {bname:9s} "
        f"cmd=({pinned[0]:+.2f},{pinned[1]:+.2f},{pinned[2]:+.2f}) "
        f"track_lin {row['flin']:.3f} track_ang {row['fang']:.3f} "
        f"err_xy {row['err_xy']:.3f} ach=({row['vx']:+.3f},{row['vy']:+.3f},"
        f"{row['wz']:+.3f}) falls {row['falls']}"
      )

    # ---------------- natural distribution + gate reachability ----------------
    term._resample_command = orig_resample  # restore real sampler
    obs, _ = wenv.reset()
    cmd_vx, ach_vx, cmd_wz, ach_wz = [], [], [], []
    with torch.no_grad():
      for step in range(999):
        obs, _, _, _ = wenv.step(policy(obs))
        if step >= 50:
          cmd = uenv.command_manager.get_command("twist")
          keep = torch.norm(cmd[:, :3], dim=1) > 0.0  # skip zeroed/standing
          cmd_vx.append(cmd[keep, 0].clone())
          ach_vx.append(robot.data.root_link_lin_vel_b[keep, 0].clone())
          cmd_wz.append(cmd[keep, 2].clone())
          ach_wz.append(robot.data.root_link_ang_vel_b[keep, 2].clone())
    cx, ax = torch.cat(cmd_vx), torch.cat(ach_vx)
    cz, az = torch.cat(cmd_wz), torch.cat(ach_wz)

    def slope(c, a):
      c0, a0 = c - c.mean(), a - a.mean()
      return (c0 * a0).sum().item() / (c0 * c0).sum().item()

    # Episodic fractions exactly like command_grid_adaptive (episode ended at
    # 999 steps of a 1000-step episode; sums/lengths not yet reset).
    rm = uenv.reward_manager
    dt = uenv.step_dt
    steps_buf = uenv.episode_length_buf.clamp(min=1).float()
    w_lin = rm.get_term_cfg("track_linear_velocity").weight
    w_ang = rm.get_term_cfg("track_angular_velocity").weight
    ep_lin = rm._episode_sums["track_linear_velocity"] / (steps_buf * w_lin * dt)
    ep_ang = rm._episode_sums["track_angular_velocity"] / (steps_buf * w_ang * dt)
    full = uenv.episode_length_buf >= 940  # exclude envs that fell mid-episode
    if term.grid_enabled:
      attributable = full & (term.grid_cell_index >= 0)
    else:
      attributable = full & (torch.norm(term.vel_command_b[:, :3], dim=1) > 0.0)
    el, ea = ep_lin[attributable], ep_ang[attributable]
    pass_frac = ((el >= 0.8) & (ea >= 0.7)).float().mean().item()
    seed_tracking = torch.minimum(el, ea).mean().item()
    natural[(pol_label, env_label)] = dict(
      slope_vx=slope(cx, ax),
      slope_wz=slope(cz, az),
      ep_lin=el.mean().item(),
      ep_ang=ea.mean().item(),
      ep_lin_p90=el.quantile(0.9).item(),
      ep_ang_p90=ea.quantile(0.9).item(),
      seed_tracking=seed_tracking,
      pass_frac=pass_frac,
      n_attr=int(attributable.sum().item()),
    )
    d = natural[(pol_label, env_label)]
    print(
      f"[{pol_label:8s} x {env_label:12s}] NATURAL: slope vx {d['slope_vx']:.3f} "
      f"wz {d['slope_wz']:.3f} | episodic frac lin {d['ep_lin']:.3f} "
      f"(p90 {d['ep_lin_p90']:.3f}) ang {d['ep_ang']:.3f} (p90 {d['ep_ang_p90']:.3f}) "
      f"| min-mean {d['seed_tracking']:.3f} | pass(0.8,0.7) {d['pass_frac']:.4f} "
      f"({d['n_attr']} envs)"
    )
    del runner, policy

  env.close()
  del env, wenv

# ---------------- summary matrix ----------------
print("\n==== 2x2 matrix: mean over ACTIVE buckets of min(track_lin, track_ang) ====")
active_buckets = [b for b in BUCKETS if b != "stand"]
for pol_label in CKPTS:
  for env_label in TASKS:
    vals = [
      min(results[(pol_label, env_label, b)]["flin"], results[(pol_label, env_label, b)]["fang"])
      for b in active_buckets
    ]
    print(f"  {pol_label:8s} x {env_label:12s}: {sum(vals) / len(vals):.3f}")
