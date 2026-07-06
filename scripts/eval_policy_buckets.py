"""Per-direction evaluation of a trained XGOLite-Flat checkpoint.

Same environment as eval_policy.py (noise, pushes, DR all active), but the
velocity command is forced to one fixed value per bucket so directional
asymmetry can't hide in the average. Run on a baseline checkpoint before
retraining, then compare.

Usage: python scripts/eval_policy_buckets.py <checkpoint.pt>
"""
import sys
from dataclasses import asdict

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
STEPS = 600  # 12 s at 50 Hz per bucket
SETTLE_STEPS = 100  # skip transient after reset before measuring

# (vx, vy, wz) forced per bucket. cw/ccw separated on purpose: the br hip's
# tighter joint limit (-0.65 rad) predicts a possible turn asymmetry.
BUCKETS = {
    "fwd":       (0.4, 0.0, 0.0),
    "fwd_fast":  (0.9, 0.0, 0.0),
    "back":      (-0.3, 0.0, 0.0),
    "back_slow": (-0.15, 0.0, 0.0),
    "left":      (0.0, 0.10, 0.0),
    "right":     (0.0, -0.10, 0.0),
    "ccw":       (0.0, 0.0, 0.8),
    "cw":        (0.0, 0.0, -0.8),
    "ccw_slow":  (0.0, 0.0, 0.4),
    "fwd_turn":  (0.3, 0.0, 0.6),
    "stand":     (0.0, 0.0, 0.0),
}

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

uenv = env.unwrapped
term = uenv.command_manager.get_term("twist")
# Axis-focus resampling would zero components of the forced commands.
term.cfg.axis_focus_probs = None

print(f"{'bucket':9s} {'cmd (vx,vy,wz)':>20s} {'err_xy':>8s} {'err_yaw':>8s} {'falls':>6s}")
results = {}
for name, (vx, vy, wz) in BUCKETS.items():
    # Degenerate uniform ranges force the command through the normal
    # resample path (the >0.1-norm zeroing handles the "stand" bucket).
    term.cfg.ranges.lin_vel_x = (vx, vx)
    term.cfg.ranges.lin_vel_y = (vy, vy)
    term.cfg.ranges.ang_vel_z = (wz, wz)

    obs, _ = env.reset()
    falls = 0
    err_xy_sum = 0.0
    err_yaw_sum = 0.0
    n_err = 0
    with torch.no_grad():
        for step in range(STEPS):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            tm = uenv.termination_manager
            falls += int(tm.get_term("fell_over").sum().item())
            falls += int(tm.get_term("illegal_contact").sum().item())
            if step < SETTLE_STEPS:
                continue
            cmd = torch.tensor([vx, vy, wz], device=device)
            robot = uenv.scene["robot"]
            lin_b = robot.data.root_link_lin_vel_b[:, :2]
            ang_z = robot.data.root_link_ang_vel_b[:, 2]
            err_xy_sum += torch.norm(cmd[:2] - lin_b, dim=1).mean().item()
            err_yaw_sum += (cmd[2] - ang_z).abs().mean().item()
            n_err += 1
    err_xy = err_xy_sum / n_err
    err_yaw = err_yaw_sum / n_err
    results[name] = (err_xy, err_yaw, falls)
    print(f"{name:9s} ({vx:+.2f},{vy:+.2f},{wz:+.2f})      "
          f"{err_xy:8.3f} {err_yaw:8.3f} {falls:6d}")

print(f"\n{STEPS} steps x {NUM_ENVS} envs per bucket "
      f"(first {SETTLE_STEPS} steps after reset excluded from errors)")
