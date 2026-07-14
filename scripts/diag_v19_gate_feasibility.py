"""v19 gate feasibility + wobble-floor decomposition (2026-07-14).

Runs the three cheap diagnostics from the v19 curriculum-stall
investigation against the v3_ext checkpoint, WITHOUT retraining:

1. GATE FEASIBILITY (candidate #1): natural-roll gate distribution on the
   v19 env — per-episode frac_lin / frac_ang exactly as
   ``command_grid_adaptive`` computes them, deterministic AND stochastic
   actions. If the stochastic max sits below the (0.70, 0.55) gates, the
   gates are proven unreachable on the measured plant regardless of
   policy competence (they were calibrated on the SOFT plant's
   distributions, range_curriculum.py lines 87-97).
2. REWARD-SIDE DITHER (candidate #2): while standing, compare the
   instantaneous physics-rate base velocity (what the tracking rewards
   and the curriculum metric read) against the 20 ms finite difference
   of base pose (what a control-rate-aligned metric would read). A large
   inst/fd RMS ratio = the reward channel carries relay dither the
   policy cannot remove.
3. WOBBLE DECOMPOSITION (candidates #3/#4): the same pinned buckets on
   env variants with deadband OFF, deadband at the fitted bracket
   (0, 0.008), and friction pinned to 1.0 — attributing the 4x stand
   wobble floor (err_yaw 0.135 vs v18's 0.030) between deadband slop,
   slippery-floor draws, and the bare plant.

Obs contract must match the checkpoint's training world: v3/v3_ext
trained on the instantaneous joint_vel obs (``--obs inst`` reverts the
actor term, default), v4+ on the control-rate term (``--obs control``
keeps the cfg as registered).

Usage:
  cd luwu_mjlab && PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u \
      scripts/diag_v19_gate_feasibility.py [--ckpt PATH] [--obs inst|control] \
      [--variants asis,deadband_off,deadband_fit,friction_1]
"""

import argparse

import dataclasses
import types
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as up_mdp
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity import mdp as local_mdp
from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

NUM_ENVS = 256
STEPS = 400
SETTLE = 100
NOMINAL_POSE = (0.0, 0.116)
TASK = "XGOLite-V19"
DEFAULT_CKPT = "logs/rsl_rl/xgolite_v19/2026-07-14_19-43-17_v19_v3_ext/model_4998.pt"
GATES = (0.70, 0.55)  # gamma_lin, gamma_ang shipped in range_curriculum.py

_args = argparse.Namespace(ckpt=DEFAULT_CKPT, obs="inst", zero_actions=False)

# Seed-region buckets (the only commands v3_ext ever trained on) + stand.
BUCKETS = {
  "stand": (0.0, 0.0, 0.0),
  "fwd_slow": (0.08, 0.0, 0.0),
  "fwd": (0.20, 0.0, 0.0),
  "back": (-0.10, 0.0, 0.0),
  "ccw_slow": (0.0, 0.0, 0.4),
}

configure_torch_backends()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def make_cfg(variant: str):
  cfg = load_env_cfg(TASK, play=False)
  cfg.scene.num_envs = NUM_ENVS
  if _args.obs == "inst":
    # Reproduce the v3/v3_ext training world: instantaneous joint_vel obs.
    terms = cfg.observations["actor"].terms
    terms["joint_vel"] = dataclasses.replace(
      terms["joint_vel"], func=up_mdp.joint_vel_rel
    )
  if variant == "deadband_off":
    _set_deadband(cfg, (0.0, 0.0))
  elif variant == "deadband_fit":
    _set_deadband(cfg, (0.0, 0.008))
  elif variant == "friction_1":
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
  else:
    assert variant == "asis"
  return cfg


def _set_deadband(cfg, rng):
  robot = cfg.scene.entities["robot"]
  new_actuators = []
  for act_cfg in robot.articulation.actuators:
    if isinstance(act_cfg, local_mdp.PerServoDelayedActuatorCfg):
      act_cfg = dataclasses.replace(act_cfg, deadband_range=rng)
    new_actuators.append(act_cfg)
  articulation = dataclasses.replace(
    robot.articulation, actuators=tuple(new_actuators)
  )
  cfg.scene.entities["robot"] = dataclasses.replace(
    robot, articulation=articulation
  )


def pin(term, pinned):
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


def yaw_of(quat_wxyz: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat_wxyz.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run_variant(variant: str, natural_rolls: bool) -> None:
  cfg = make_cfg(variant)
  agent_cfg = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE, render_mode=None)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  uenv = wenv.unwrapped
  term = uenv.command_manager.get_term("twist")
  assert isinstance(term, UniformVelocityCommand)
  orig_resample = term._resample_command
  robot = uenv.scene["robot"]

  runner = runner_cls(wenv, asdict(agent_cfg), device=DEVICE)
  runner.load(
    _args.ckpt, load_cfg={"actor": True}, strict=True, map_location=DEVICE
  )
  if _args.zero_actions:
    # Non-tracking reference: hold the default pose (zero actions). Gives
    # the do-nothing side of the gate-separation calibration.
    zeros = torch.zeros(
      NUM_ENVS, uenv.action_manager.total_action_dim, device=DEVICE
    )
    policy = lambda obs: zeros  # noqa: E731
    actor = None
  else:
    policy = runner.get_inference_policy(device=DEVICE)
    actor = runner.alg.actor

  for bname, pinned in BUCKETS.items():
    pin(term, pinned)
    obs, _ = wenv.reset()
    s = dict(flin=0.0, fang=0.0, err_xy=0.0, err_yaw=0.0, vx=0.0, wz=0.0)
    inst_wz_sq = fd_wz_sq = inst_vxy_sq = fd_vxy_sq = 0.0
    prev_yaw = prev_pos = None
    falls, n = 0, 0
    with torch.no_grad():
      for step in range(STEPS):
        obs, _, _, _ = wenv.step(policy(obs))
        tm = uenv.termination_manager
        falls += int(tm.get_term("fell_over").sum().item())
        falls += int(tm.get_term("illegal_contact").sum().item())
        yaw = yaw_of(robot.data.root_link_quat_w)
        pos = robot.data.root_link_pos_w[:, :2]
        if step < SETTLE:
          prev_yaw, prev_pos = yaw.clone(), pos.clone()
          continue
        cmd = uenv.command_manager.get_command("twist")
        lin = robot.data.root_link_lin_vel_b
        ang = robot.data.root_link_ang_vel_b
        err = torch.sum(
          torch.square(cmd[:, :2] - lin[:, :2]), dim=1
        ) + 2 * torch.square(lin[:, 2])
        s["flin"] += torch.exp(-err / 0.01).mean().item()
        aerr = torch.square(cmd[:, 2] - ang[:, 2]) + 0.05 * torch.sum(
          torch.square(ang[:, :2]), dim=1
        )
        s["fang"] += (
          torch.exp(-aerr / (0.2 + 0.4 * cmd[:, 2].abs()) ** 2).mean().item()
        )
        s["err_xy"] += torch.norm(cmd[:, :2] - lin[:, :2], dim=1).mean().item()
        s["err_yaw"] += (cmd[:, 2] - ang[:, 2]).abs().mean().item()
        s["vx"] += lin[:, 0].mean().item()
        s["wz"] += ang[:, 2].mean().item()
        # Reward-side dither probe: instantaneous vs control-step FD.
        dyaw = torch.atan2(
          torch.sin(yaw - prev_yaw), torch.cos(yaw - prev_yaw)
        )
        fd_wz = dyaw / uenv.step_dt
        fd_v = torch.norm(pos - prev_pos, dim=1) / uenv.step_dt
        inst_wz_sq += torch.square(ang[:, 2]).mean().item()
        fd_wz_sq += torch.square(fd_wz).mean().item()
        inst_vxy_sq += torch.sum(torch.square(lin[:, :2]), dim=1).mean().item()
        fd_vxy_sq += torch.square(fd_v).mean().item()
        prev_yaw, prev_pos = yaw.clone(), pos.clone()
        n += 1
    r = {k: v / n for k, v in s.items()}
    line = (
      f"[{variant:12s}] {bname:9s} "
      f"cmd=({pinned[0]:+.2f},{pinned[1]:+.2f},{pinned[2]:+.2f}) "
      f"track_lin {r['flin']:.3f} track_ang {r['fang']:.3f} "
      f"err_xy {r['err_xy']:.3f} err_yaw {r['err_yaw']:.3f} "
      f"ach_vx {r['vx']:+.3f} ach_wz {r['wz']:+.3f} falls {falls}"
    )
    if bname == "stand":
      line += (
        f" | dither probe: wz inst-RMS {(inst_wz_sq / n) ** 0.5:.3f}"
        f" vs fd-RMS {(fd_wz_sq / n) ** 0.5:.3f};"
        f" vxy inst-RMS {(inst_vxy_sq / n) ** 0.5:.3f}"
        f" vs fd-RMS {(fd_vxy_sq / n) ** 0.5:.3f}"
      )
    print(line)

  if natural_rolls:
    term._resample_command = orig_resample
    modes = ("determ",) if _args.zero_actions else ("determ", "stoch")
    for mode in modes:
      obs, _ = wenv.reset()
      with torch.no_grad():
        for _ in range(999):
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
      ep_lin = rm._episode_sums["track_linear_velocity"] / (
        steps_buf * w_lin * dt
      )
      ep_ang = rm._episode_sums["track_angular_velocity"] / (
        steps_buf * w_ang * dt
      )
      attributable = (uenv.episode_length_buf >= 940) & (
        term.grid_cell_index >= 0
      )
      el, ea = ep_lin[attributable], ep_ang[attributable]
      mn = torch.minimum(el, ea)
      passed = (el >= GATES[0]) & (ea >= GATES[1])
      print(
        f"[{variant:12s}] NATURAL-{mode:6s}: "
        f"lin mean {el.mean():.3f} p90 {el.quantile(0.9):.3f} "
        f"max {el.max():.3f} | "
        f"ang mean {ea.mean():.3f} p90 {ea.quantile(0.9):.3f} "
        f"max {ea.max():.3f} | "
        f"min-mean {mn.mean():.3f} (~seed_tracking) | "
        f"pass{GATES} {passed.float().mean():.4f} "
        f"({int(attributable.sum())} envs)"
      )
      if mode == "stoch":
        try:
          params = actor.output_distribution_params
          print(
            f"    action noise std (mean over dims): "
            f"{params[1].mean().item() if params is not None else float('nan'):.3f}"
          )
        except Exception as e:
          print("    (noise std unavailable:", e, ")")
  del runner, policy, actor
  env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt", default=DEFAULT_CKPT)
  parser.add_argument("--obs", choices=("inst", "control"), default="inst")
  parser.add_argument(
    "--variants", default="asis,deadband_off,deadband_fit,friction_1"
  )
  parser.add_argument("--zero-actions", action="store_true")
  ns = parser.parse_args()
  _args.ckpt, _args.obs = ns.ckpt, ns.obs
  _args.zero_actions = ns.zero_actions
  print(f"# ckpt={ns.ckpt} obs={ns.obs} zero_actions={ns.zero_actions}")
  # asis first (gate feasibility is the headline), then the ablations.
  for i, variant in enumerate(ns.variants.split(",")):
    run_variant(variant.strip(), natural_rolls=(i == 0))
