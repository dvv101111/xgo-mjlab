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
from src.tasks.velocity.mdp.rewards import body_pitch_from_gravity

TASK = "XGOLite-Flat"
CKPT = sys.argv[1]
NUM_ENVS = 256
STEPS = 600  # 12 s at 50 Hz per bucket
SETTLE_STEPS = 100  # skip transient after reset before measuring

# v14 nominal body pose (pitch [rad], base height [m]); the 11 twist buckets
# all run at this pose.
NOMINAL_POSE = (0.0, 0.116)

# (vx, vy, wz, body_pitch, base_height) forced per bucket. cw/ccw separated
# on purpose: the br hip's tighter joint limit (-0.65 rad) predicts a
# possible turn asymmetry.
BUCKETS = {
    "fwd":       (0.4, 0.0, 0.0) + NOMINAL_POSE,
    "fwd_fast":  (0.9, 0.0, 0.0) + NOMINAL_POSE,
    "back":      (-0.3, 0.0, 0.0) + NOMINAL_POSE,
    "back_slow": (-0.15, 0.0, 0.0) + NOMINAL_POSE,
    "left":      (0.0, 0.08, 0.0) + NOMINAL_POSE,
    "right":     (0.0, -0.08, 0.0) + NOMINAL_POSE,
    "ccw":       (0.0, 0.0, 0.8) + NOMINAL_POSE,
    "cw":        (0.0, 0.0, -0.8) + NOMINAL_POSE,
    "ccw_slow":  (0.0, 0.0, 0.4) + NOMINAL_POSE,
    "fwd_turn":  (0.3, 0.0, 0.6) + NOMINAL_POSE,
    "stand":     (0.0, 0.0, 0.0) + NOMINAL_POSE,
    # v14 pose buckets.
    "pose_down": (0.0, 0.0, 0.0, -0.35, 0.105),
    "pose_low":  (0.0, 0.0, 0.0, 0.0, 0.098),
    "pose_walk": (0.3, 0.0, 0.0, -0.2, 0.11),
    # v15 pose buckets: nose-up hold and tall stand (both inside the
    # pitch-conditioned height band).
    "pose_up":   (0.0, 0.0, 0.0, 0.30, 0.116),
    "pose_high": (0.0, 0.0, 0.0, 0.0, 0.138),
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
# Pose-mode override: force every resample into nominal mode, which uses
# cfg.nominal_pose verbatim and leaves the sampled twist untouched (pose_hold
# would zero the twist, posed_walk would scale it by 0.5). The pose is then
# forced per bucket via nominal_pose (+ degenerate pose ranges for good
# measure), so the commands still flow through the normal resample path.
term.cfg.pose_mode_probs = (1.0, 0.0, 0.0)

print(f"{'bucket':9s} {'cmd (vx,vy,wz,pit,h)':>26s} {'err_xy':>8s} {'err_yaw':>8s} "
      f"{'err_pitch':>9s} {'err_h':>7s} {'falls':>6s}")
results = {}
for name, (vx, vy, wz, pitch, h) in BUCKETS.items():
    # Degenerate uniform ranges force the command through the normal
    # resample path (the >0.05-norm zeroing handles the "stand" bucket).
    term.cfg.ranges.lin_vel_x = (vx, vx)
    term.cfg.ranges.lin_vel_y = (vy, vy)
    term.cfg.ranges.ang_vel_z = (wz, wz)
    term.cfg.ranges.body_pitch = (pitch, pitch)
    term.cfg.ranges.base_height = (h, h)
    term.cfg.nominal_pose = (pitch, h)

    obs, _ = env.reset()
    falls = 0
    err_xy_sum = 0.0
    err_yaw_sum = 0.0
    err_pitch_sum = 0.0
    err_h_sum = 0.0
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
            pitch_meas = body_pitch_from_gravity(robot.data.projected_gravity_b)
            err_pitch_sum += (pitch - pitch_meas).abs().mean().item()
            err_h_sum += (h - robot.data.root_link_pos_w[:, 2]).abs().mean().item()
            n_err += 1
    err_xy = err_xy_sum / n_err
    err_yaw = err_yaw_sum / n_err
    err_pitch = err_pitch_sum / n_err
    err_h = err_h_sum / n_err
    results[name] = (err_xy, err_yaw, err_pitch, err_h, falls)
    print(f"{name:9s} ({vx:+.2f},{vy:+.2f},{wz:+.2f},{pitch:+.2f},{h:.3f}) "
          f"{err_xy:8.3f} {err_yaw:8.3f} {err_pitch:9.3f} {err_h:7.3f} {falls:6d}")

print(f"\n{STEPS} steps x {NUM_ENVS} envs per bucket "
      f"(first {SETTLE_STEPS} steps after reset excluded from errors)")
