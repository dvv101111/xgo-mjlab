"""Decompose the curriculum gate metrics: why can no episode pass (0.8, 0.7)?

Rolls a policy in an env with pinned in-seed commands and splits the two
tracking exponents into their components, per step:

  lin: exp(-(err_xy^2 + 2 vz^2)/0.10^2)  -> parts err_xy^2, 2 vz^2
  ang: exp(-(err_wz^2 + 0.05 ||w_xy||^2)/std_eff^2) -> parts err_wz^2, wobble

Reports the mean exponent budget and what the per-step exp would be with
each component alone, plus the same for a push-free env — isolating
wobble vs push transients vs genuine tracking error.

Usage:
  cd luwu_mjlab && PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python \
      scripts/debug_gate_decompose.py
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
STEPS = 500
SETTLE = 100
NOMINAL_POSE = (0.0, 0.116)
TASK = "XGOLite-V18Draft"
CKPT = "logs/rsl_rl/xgolite_v18draft/2026-07-12_01-21-55_v18base_v1/model_1499.pt"
PIN = (0.08, 0.0, 0.0)  # in-seed fwd_slow

configure_torch_backends()
device = "cuda:0"

for pushes in (True, False):
  env_cfg = load_env_cfg(TASK, play=False)
  env_cfg.scene.num_envs = NUM_ENVS
  if not pushes:
    env_cfg.events.pop("push_robot", None)
  agent_cfg = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(wenv, asdict(agent_cfg), device=device)
  runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  uenv = wenv.unwrapped
  term = uenv.command_manager.get_term("twist")
  assert isinstance(term, UniformVelocityCommand)

  def _pin(self, env_ids):
    self.vel_command_b[env_ids, 0] = PIN[0]
    self.vel_command_b[env_ids, 1] = PIN[1]
    self.vel_command_b[env_ids, 2] = PIN[2]
    if self.pose_enabled:
      self.vel_command_b[env_ids, 3] = NOMINAL_POSE[0]
      self.vel_command_b[env_ids, 4] = NOMINAL_POSE[1]
    self.is_standing_env[env_ids] = False

  term._resample_command = types.MethodType(_pin, term)
  robot = uenv.scene["robot"]

  acc = dict(
    err_xy2=0.0, vz2=0.0, err_wz2=0.0, wobble=0.0, flin=0.0, fang=0.0,
    fang_yaw_only=0.0, fang_wobble_only=0.0, flin_xy_only=0.0, w_xy_rms=0.0,
  )
  n = 0
  obs, _ = wenv.reset()
  with torch.no_grad():
    for step in range(STEPS):
      obs, _, _, _ = wenv.step(policy(obs))
      if step < SETTLE:
        continue
      cmd = uenv.command_manager.get_command("twist")
      lin = robot.data.root_link_lin_vel_b
      ang = robot.data.root_link_ang_vel_b
      err_xy2 = torch.sum(torch.square(cmd[:, :2] - lin[:, :2]), dim=1)
      vz2 = 2 * torch.square(lin[:, 2])
      err_wz2 = torch.square(cmd[:, 2] - ang[:, 2])
      wob = 0.05 * torch.sum(torch.square(ang[:, :2]), dim=1)
      std_eff2 = (0.2 + 0.4 * cmd[:, 2].abs()) ** 2
      acc["err_xy2"] += err_xy2.mean().item()
      acc["vz2"] += vz2.mean().item()
      acc["err_wz2"] += err_wz2.mean().item()
      acc["wobble"] += wob.mean().item()
      acc["flin"] += torch.exp(-(err_xy2 + vz2) / 0.01).mean().item()
      acc["flin_xy_only"] += torch.exp(-err_xy2 / 0.01).mean().item()
      acc["fang"] += torch.exp(-(err_wz2 + wob) / std_eff2).mean().item()
      acc["fang_yaw_only"] += torch.exp(-err_wz2 / std_eff2).mean().item()
      acc["fang_wobble_only"] += torch.exp(-wob / std_eff2).mean().item()
      acc["w_xy_rms"] += torch.norm(ang[:, :2], dim=1).square().mean().item()
      n += 1
  r = {k: v / n for k, v in acc.items()}
  print(
    f"pushes={pushes}: v18base policy, {TASK}, pinned {PIN}\n"
    f"  mean exponent parts: err_xy^2 {r['err_xy2']:.4f}  2vz^2 {r['vz2']:.4f} "
    f"(sigma_lin^2=0.01) | err_wz^2 {r['err_wz2']:.4f}  0.05||w_xy||^2 "
    f"{r['wobble']:.4f} (sigma_ang^2=0.04)\n"
    f"  ||w_xy|| RMS: {r['w_xy_rms'] ** 0.5:.3f} rad/s\n"
    f"  per-step means: frac_lin {r['flin']:.3f} (xy-only {r['flin_xy_only']:.3f}) | "
    f"frac_ang {r['fang']:.3f} (yaw-only {r['fang_yaw_only']:.3f}, "
    f"wobble-only {r['fang_wobble_only']:.3f})"
  )
  env.close()
  del env, wenv, runner, policy
