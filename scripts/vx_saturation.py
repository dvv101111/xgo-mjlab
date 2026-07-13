"""Fine-grained achieved-vs-commanded speed curve for a trained checkpoint.

Purpose (aggressive-preset campaign, 2026-07-11): find where the policy's
achieved speed saturates when the command is extrapolated past the trained
range — this sets the command ranges for the Sprint/FastClock/FreeGait/Agile
presets. Same full-DR environment as eval_policy_buckets.py, vx-focused
buckets plus a fast-yaw pair for the Agile range.

Usage: PYTHONPATH=. .venv/bin/python -u scripts/vx_saturation.py <ckpt.pt>
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
NUM_ENVS = 128
STEPS = 400  # 8 s at 50 Hz per bucket
SETTLE_STEPS = 100

NOMINAL_POSE = (0.0, 0.116)

# (vx, vy, wz): forward sweep past the trained 0.45 edge, backward sweep past
# the trained -0.40 edge, then yaw/turn probes for the Agile ranges.
BUCKETS = {
    "fwd_020":  (0.20, 0.0, 0.0),
    "fwd_030":  (0.30, 0.0, 0.0),
    "fwd_040":  (0.40, 0.0, 0.0),
    "fwd_045":  (0.45, 0.0, 0.0),
    "fwd_055":  (0.55, 0.0, 0.0),
    "fwd_065":  (0.65, 0.0, 0.0),
    "fwd_080":  (0.80, 0.0, 0.0),
    "fwd_100":  (1.00, 0.0, 0.0),
    "back_015": (-0.15, 0.0, 0.0),
    "back_025": (-0.25, 0.0, 0.0),
    "back_035": (-0.35, 0.0, 0.0),
    "back_045": (-0.45, 0.0, 0.0),
    "back_060": (-0.60, 0.0, 0.0),
    "yaw_100":  (0.0, 0.0, 1.00),
    "yaw_140":  (0.0, 0.0, 1.40),
    "turn_fast": (0.40, 0.0, 1.00),
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
term.cfg.axis_focus_probs = None
term.cfg.slow_vx_prob = 0.0
term.cfg.pose_mode_probs = (1.0, 0.0, 0.0)
term.cfg.nominal_pose = NOMINAL_POSE
term.cfg.ranges.body_pitch = NOMINAL_POSE[:1] * 2
term.cfg.ranges.base_height = (NOMINAL_POSE[1], NOMINAL_POSE[1])

print(f"{'bucket':10s} {'cmd':>20s} {'vx_ach':>7s} {'vy_ach':>7s} {'wz_ach':>7s} "
      f"{'ratio':>6s} {'falls':>6s}")
for name, (vx, vy, wz) in BUCKETS.items():
    term.cfg.ranges.lin_vel_x = (vx, vx)
    term.cfg.ranges.lin_vel_y = (vy, vy)
    term.cfg.ranges.ang_vel_z = (wz, wz)

    obs, _ = env.reset()
    falls = 0
    vx_sum = vy_sum = wz_sum = 0.0
    n = 0
    with torch.no_grad():
        for step in range(STEPS):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            tm = uenv.termination_manager
            falls += int(tm.get_term("fell_over").sum().item())
            falls += int(tm.get_term("illegal_contact").sum().item())
            if step < SETTLE_STEPS:
                continue
            robot = uenv.scene["robot"]
            lin_b = robot.data.root_link_lin_vel_b[:, :2]
            vx_sum += lin_b[:, 0].mean().item()
            vy_sum += lin_b[:, 1].mean().item()
            wz_sum += robot.data.root_link_ang_vel_b[:, 2].mean().item()
            n += 1
    vx_a, vy_a, wz_a = vx_sum / n, vy_sum / n, wz_sum / n
    main_cmd = vx if abs(vx) > 1e-6 else wz
    main_ach = vx_a if abs(vx) > 1e-6 else wz_a
    ratio = main_ach / main_cmd if abs(main_cmd) > 1e-6 else float("nan")
    print(f"{name:10s} ({vx:+.2f},{vy:+.2f},{wz:+.2f}) "
          f"{vx_a:+7.3f} {vy_a:+7.3f} {wz_a:+7.3f} {ratio:6.2f} {falls:6d}",
          flush=True)

print(f"\n{STEPS} steps x {NUM_ENVS} envs per bucket "
      f"(first {SETTLE_STEPS} steps excluded)")
