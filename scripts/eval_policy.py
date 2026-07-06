"""Headless evaluation of a trained XGOLite-Flat checkpoint.

Runs the trained policy in the full training environment (noise, pushes,
domain randomization all active) and reports fall rate and velocity
tracking error over complete episodes.
"""
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401  (registry)
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK = "XGOLite-Flat"
CKPT = sys.argv[1]
NUM_ENVS = 256
STEPS = 1500  # 30 s at 50 Hz -> at least one full episode + resets

configure_torch_backends()
device = "cuda:0"

env_cfg = load_env_cfg(TASK, play=False)
env_cfg.scene.num_envs = NUM_ENVS
agent_cfg = load_rl_cfg(TASK)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

falls = 0
timeouts = 0
illegal = 0
err_xy_sum = 0.0
err_yaw_sum = 0.0
n_err = 0

obs = env.get_observations()[0] if isinstance(env.get_observations(), tuple) else env.get_observations()
uenv = env.unwrapped

with torch.no_grad():
    for step in range(STEPS):
        actions = policy(obs)
        obs, _, dones, extras = env.step(actions)

        tm = uenv.termination_manager
        falls += int(tm.get_term("fell_over").sum().item())
        illegal += int(tm.get_term("illegal_contact").sum().item())
        timeouts += int(tm.get_term("time_out").sum().item())

        cmd = uenv.command_manager.get_command("twist")  # (N,5) vx vy wz pitch h
        robot = uenv.scene["robot"]
        lin_b = robot.data.root_link_lin_vel_b[:, :2]
        ang_z = robot.data.root_link_ang_vel_b[:, 2]
        err_xy_sum += torch.norm(cmd[:, :2] - lin_b, dim=1).mean().item()
        err_yaw_sum += (cmd[:, 2] - ang_z).abs().mean().item()
        n_err += 1

episodes = falls + timeouts + illegal
print(f"steps: {STEPS} x {NUM_ENVS} envs = {STEPS*NUM_ENVS} env-steps (30 s each)")
print(f"episodes ended: {episodes}  (time_out {timeouts}, fell_over {falls}, illegal_contact {illegal})")
print(f"fall rate: {100.0*falls/max(episodes,1):.2f}% of episodes")
print(f"mean lin vel tracking error: {err_xy_sum/n_err:.3f} m/s")
print(f"mean yaw rate tracking error: {err_yaw_sum/n_err:.3f} rad/s")
