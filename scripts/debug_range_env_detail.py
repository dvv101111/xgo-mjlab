"""V18Range-env detail: pinned buckets + natural-roll gate distribution,
with BOTH deterministic inference actions and stochastic (training-style)
sampled actions. Explains why zero episodes passed the (0.8, 0.7) unlock
gate during training while the deterministic policy passes ~50%.

Usage:
  cd luwu_mjlab && PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u \
      scripts/debug_range_env_detail.py
"""

import types
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
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
TASK = "XGOLite-V18Range"

CKPTS = {
  "v18range": "logs/rsl_rl/xgolite_v18range/2026-07-12_01-48-38_v18range_v1/model_2499.pt",
  "v18base": "logs/rsl_rl/xgolite_v18draft/2026-07-12_01-21-55_v18base_v1/model_1499.pt",
}
BUCKETS = {
  "fwd_slow": (0.08, 0.0, 0.0),
  "fwd": (0.20, 0.0, 0.0),
  "back": (-0.10, 0.0, 0.0),
  "ccw_slow": (0.0, 0.0, 0.4),
  "stand": (0.0, 0.0, 0.0),
  # out-of-seed probes (never trained under the frozen grid):
  "ccw_fast": (0.0, 0.0, 0.8),
  "fwd_med": (0.30, 0.0, 0.0),
}

configure_torch_backends()
device = "cuda:0"

env_cfg = load_env_cfg(TASK, play=False)
env_cfg.scene.num_envs = NUM_ENVS
agent_cfg = load_rl_cfg(TASK)
env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
uenv = wenv.unwrapped
term = uenv.command_manager.get_term("twist")
assert isinstance(term, UniformVelocityCommand)
orig_resample = term._resample_command
robot = uenv.scene["robot"]


def pin(pinned):
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


for pol_label, ckpt in CKPTS.items():
  runner = runner_cls(wenv, asdict(agent_cfg), device=device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  actor = runner.alg.actor
  try:
    std = actor.output_distribution_params  # populated after a call
  except Exception:
    std = None

  for bname, pinned in BUCKETS.items():
    pin(pinned)
    obs, _ = wenv.reset()
    s = dict(flin=0.0, fang=0.0, err_xy=0.0, vx=0.0, wz=0.0)
    falls, n = 0, 0
    with torch.no_grad():
      for step in range(STEPS):
        obs, _, _, _ = wenv.step(policy(obs))
        tm = uenv.termination_manager
        falls += int(tm.get_term("fell_over").sum().item())
        falls += int(tm.get_term("illegal_contact").sum().item())
        if step < SETTLE:
          continue
        cmd = uenv.command_manager.get_command("twist")
        lin = robot.data.root_link_lin_vel_b
        ang = robot.data.root_link_ang_vel_b
        err = torch.sum(torch.square(cmd[:, :2] - lin[:, :2]), dim=1) + 2 * torch.square(
          lin[:, 2]
        )
        s["flin"] += torch.exp(-err / 0.01).mean().item()
        aerr = torch.square(cmd[:, 2] - ang[:, 2]) + 0.05 * torch.sum(
          torch.square(ang[:, :2]), dim=1
        )
        s["fang"] += (
          torch.exp(-aerr / (0.2 + 0.4 * cmd[:, 2].abs()) ** 2).mean().item()
        )
        s["err_xy"] += torch.norm(cmd[:, :2] - lin[:, :2], dim=1).mean().item()
        s["vx"] += lin[:, 0].mean().item()
        s["wz"] += ang[:, 2].mean().item()
        n += 1
    r = {k: v / n for k, v in s.items()}
    print(
      f"[{pol_label:8s} x V18Range-env] {bname:9s} "
      f"cmd=({pinned[0]:+.2f},{pinned[1]:+.2f},{pinned[2]:+.2f}) "
      f"track_lin {r['flin']:.3f} track_ang {r['fang']:.3f} err_xy {r['err_xy']:.3f} "
      f"ach_vx {r['vx']:+.3f} ach_wz {r['wz']:+.3f} falls {falls}"
    )

  # -------- natural rolls: deterministic vs stochastic --------
  term._resample_command = orig_resample
  for mode in ("determ", "stoch"):
    obs, _ = wenv.reset()
    with torch.no_grad():
      for step in range(999):
        if mode == "determ":
          a = policy(obs)
        else:
          a = actor(obs, stochastic_output=True)
        obs, _, _, _ = wenv.step(a)
    rm = uenv.reward_manager
    dt = uenv.step_dt
    steps_buf = uenv.episode_length_buf.clamp(min=1).float()
    w_lin = rm.get_term_cfg("track_linear_velocity").weight
    w_ang = rm.get_term_cfg("track_angular_velocity").weight
    ep_lin = rm._episode_sums["track_linear_velocity"] / (steps_buf * w_lin * dt)
    ep_ang = rm._episode_sums["track_angular_velocity"] / (steps_buf * w_ang * dt)
    attributable = (uenv.episode_length_buf >= 940) & (term.grid_cell_index >= 0)
    el, ea = ep_lin[attributable], ep_ang[attributable]
    mn = torch.minimum(el, ea)
    passed = (el >= 0.8) & (ea >= 0.7)
    print(
      f"[{pol_label:8s} x V18Range-env] NATURAL-{mode:6s}: "
      f"lin mean {el.mean():.3f} p90 {el.quantile(0.9):.3f} max {el.max():.3f} | "
      f"ang mean {ea.mean():.3f} p90 {ea.quantile(0.9):.3f} max {ea.max():.3f} | "
      f"min-mean {mn.mean():.3f} (~seed_tracking) | "
      f"pass(0.8,0.7) {passed.float().mean():.4f} ({int(attributable.sum())} envs)"
    )
    if mode == "stoch":
      # action noise std actually used
      try:
        params = actor.output_distribution_params
        print(
          f"    action noise std (mean over dims): "
          f"{params[1].mean().item() if params is not None else float('nan'):.3f}"
        )
      except Exception as e:
        print("    (noise std unavailable:", e, ")")
  del runner, policy

env.close()
