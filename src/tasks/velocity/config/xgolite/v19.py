"""XGOLite-V19: the v18range recipe on the MEASURED servo plant (2026-07-14).

Fidelity swap, not a recipe change: rewards, grid curriculum (gates
0.70/0.55), stand injection, command envelope, observations and PPO are
bit-identical to XGOLite-V18Range, so any result delta is attributable to
the plant. The plant changes come from the completed Stage-1 system-ID fit
(servo_fit_v3.json -> measured_actuators.py):

1. MJCF joint dynamics + PD gains: damping 0.0111 / armature 0.0 /
   frictionloss 0.0005, kp 39.71 / kd 0.0068 (the real servo is a stiff
   torque-clamped relay; the previous kp 5.0 / kd 0.12 were hand-set).
   Applied via a spec_fn wrapper — the XML stays untouched.
2. Torque-speed clamp: measured piecewise knee curve — flat 0.22 N*m up to
   qd_knee 3.65 rad/s, linear taper to zero at qd_max 12.09 (v18 used a
   single line drooping from qd 0 to 4.5).
3. Per-servo delay DR: measured 42.9 ms global delay +/- 22 ms onset
   spread -> 10-32 physics steps = 21-65 ms (v18 used 60-100 ms).
4. Deadband ON, randomized WIDE: (0.0, 0.025) rad per (env, servo) at
   reset. The fitted nominals are small (0.0005-0.008 rad) but measured
   small-amplitude delivery is 0.79-0.93, implying up to ~0.02 rad
   effective lost motion (dynamic hysteresis the Stage-1 model cannot
   express) — so the range brackets 0..~0.025 instead of pinning nominals.
5. Strength DR shrunk: per-env 0.85-1.15 x per-servo 0.95-1.05 (v18:
   0.75-1.25 x 0.90-1.10). The gains are now measured; what remains is
   battery sag and servo wear.
6. Surface friction DR widened: sliding friction 0.25-2.0 (v18: 0.3-1.6),
   per Margolis "Walk These Ways" (0.05-4.0 on Mini Cheetah) scaled to the
   indoor deploy surfaces (hard tile to rubber mat). The foot pads get
   collision priority 1: MuJoCo combines equal-priority contact friction
   as the element-wise MAX of the two geoms, so with the default plane
   friction 1.0 any draw below 1.0 would otherwise be masked. Terrain-type
   (per-tile material) machinery does not exist in this stack; the flat
   task randomizes the single per-env coefficient at startup.
7. PPO init_std 1.0 -> 0.4 (v19 v1 postmortem, 2026-07-14): on the
   measured plant the v18 exploration noise is violently self-destructive
   — kp 39.7 with the speed envelope open to 12.1 rad/s turns std-1.0
   action noise (x0.25 rad scale) into ~14 rad/s flailing and immediate
   illegal-contact terminations (the soft kp-5 / qd_max-4.5 v18 plant
   capped the same noise at ~1.7 rad/s). Probe data (zero/0.3/1.0-
   amplitude rollouts): quiet standing is stable and +/-0.3-amplitude
   actions are survivable. The std remains learned (scalar) + entropy
   bonus, so it can re-grow once the policy earns tracking reward.
8. joint_acc_l2 measured at CONTROL rate (v19 v2 postmortem, the actual
   suicide-attractor root cause): the fitted stiff relay dithers at the
   500 Hz physics rate (sub-milliradian, invisible to the 100 Hz fit
   telemetry — damped variants replay 3x worse, so the fit stands), and
   mjlab's qacc-based joint_acc_l2 turned that uncontrollable dither
   into ~-0.8/step, i.e. -660 of the -704 mean episode reward at v2
   iter 50 — living cost more than the -200 termination, and both v1
   (std 1.0) and v2 (std 0.4) collapsed to 5-step episodes. Same weight
   (-2.5e-7), same semantics at gait frequencies; only the acceleration
   source changes to the 50 Hz finite difference of joint_vel (see
   ``mdp.rewards.joint_acc_control_rate_l2``).
9. ACTOR joint_vel obs measured at CONTROL rate (v19 v3/v3_ext
   postmortem, 2026-07-14): with points 7+8 in place both v3 runs
   trained clean (reward ~64, eplen 1000) but seed_tracking plateaued
   at 0.13-0.16 vs the 0.70/0.55 unlock gates for 5000 iterations — the
   grid never expanded and bucket tracking stalled at ~40-55 % of
   command. The actor's joint_vel obs read the instantaneous
   physics-rate velocity, which on the relay plant aliases ~0.5-1 rad/s
   of structured dither noise into all 12 joint channels of every obs
   frame; the deployed loop feeds finite-differenced telemetry
   positions instead (locomotion.py), so the sim actor trains against
   proprioception noise that hardware does not even have. Fix: the
   actor term becomes the 50 Hz position finite difference
   (``mdp.observations.joint_vel_control_rate_rel``) — the obs-side
   analog of point 8 and the exact deploy signal. The CRITIC keeps the
   instantaneous privileged joint_vel (never deployed).
10. Grid unlock gates recalibrated to the measured plant (v19 v4
   postmortem, 2026-07-14): the v18-calibrated (0.70, 0.55) gates are
   PROVEN unreachable here — natural-roll distributions of the v3_ext
   and v4 checkpoints (diag_v19_gate_feasibility.py) max out at
   0.28-0.32 stochastic / 0.64-0.72 deterministic, 0 of ~180 stochastic
   episodes pass, and the stochastic min-mean (0.13-0.14) reproduces
   the training seed_tracking plateau exactly. The whole distribution
   shifted, not its tail: the measured plant's trot carries a REAL
   ~0.13 rad/s standing / 0.24-0.30 rad/s walking yaw wobble (4x the
   soft plant; confirmed real motion, not physics-rate sampling — inst
   vs control-rate fd RMS agree within 7 %) that consumes the angular
   sigma budget. Recalibration follows the same recipe that set the
   v18 gates (gates ~= the competent policy's stochastic episode
   means, range_curriculum.py comment): v4 stoch lin mean 0.149 ->
   gamma_lin 0.15, ang mean 0.172 -> gamma_ang 0.17. CAVEAT, measured:
   on this plant the metric no longer separates competence from
   idleness — a zero-action stand scores lin 0.535 / ang 0.640 on seed
   cells (wobble-free), ABOVE the walker. The gates therefore only
   restore expansion ordering (harder cells still score lower);
   restoring separation requires shrinking the real wobble (deadband
   retrain ablation / Stage-2), not gate tuning.
"""

import dataclasses

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.assets.robots.xgolite.xgolite_constants import get_spec
from src.tasks.velocity import mdp as local_mdp

from .env_cfgs import FOOT_PAD_GEOMS
from .measured_actuators import (
  DELAY_MAX_LAG_STEPS,
  DELAY_MIN_LAG_STEPS,
  apply_measured_joint_dynamics,
  enable_measured_torque_speed_clamp,
)
from .range_curriculum import xgolite_v18range_env_cfg
from .sim_fidelity import enable_servo_deadband

# Lost-motion draw range [rad]; see module docstring point 4.
V19_DEADBAND_RANGE = (0.0, 0.025)
V19_STRENGTH_ENV_RANGE = (0.85, 1.15)
V19_STRENGTH_SERVO_RANGE = (0.95, 1.05)
V19_FRICTION_RANGE = (0.25, 2.0)
# Exploration noise sized to the stiff measured plant; docstring point 7.
V19_INIT_NOISE_STD = 0.4
# Grid unlock gates recalibrated to the measured plant's stochastic
# episode-mean distributions (v4 checkpoint); docstring point 10.
V19_GAMMA_LIN = 0.15
V19_GAMMA_ANG = 0.17


def xgolite_v19_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """v18range PPO cfg with plant-appropriate initial exploration noise."""
  from .aggressive import xgolite_aggressive_ppo_runner_cfg

  cfg = xgolite_aggressive_ppo_runner_cfg("xgolite_v19")
  cfg.actor.distribution_cfg["init_std"] = V19_INIT_NOISE_STD
  return cfg


def v19_spec() -> mujoco.MjSpec:
  """XGO-Lite2 spec with the measured servo plant written in.

  cfg-level override (EntityCfg.spec_fn); the checked-in XML keeps the
  hand-set values so every pre-v19 task compiles the same model as before.
  """
  spec = get_spec()
  apply_measured_joint_dynamics(spec)
  # Foot friction DR authority: see module docstring point 6.
  for name in FOOT_PAD_GEOMS:
    spec.geom(name).priority = 1
  return spec


def _set_per_servo_delay_range(
  cfg: ManagerBasedRlEnvCfg, min_lag: int, max_lag: int
) -> None:
  """Replace the delay DR range on the robot's per-servo delayed actuators."""
  robot = cfg.scene.entities["robot"]
  assert robot.articulation is not None
  new_actuators = []
  found = False
  for act_cfg in robot.articulation.actuators:
    if isinstance(act_cfg, local_mdp.PerServoDelayedActuatorCfg):
      act_cfg = dataclasses.replace(
        act_cfg, delay_min_lag=min_lag, delay_max_lag=max_lag
      )
      found = True
    new_actuators.append(act_cfg)
  if not found:
    raise TypeError(
      "_set_per_servo_delay_range found no PerServoDelayedActuatorCfg."
    )
  articulation = dataclasses.replace(
    robot.articulation, actuators=tuple(new_actuators)
  )
  cfg.scene.entities["robot"] = dataclasses.replace(
    robot, articulation=articulation
  )


def xgolite_v19_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = xgolite_v18range_env_cfg(play=play)

  # 1. Measured MJCF dynamics + PD gains + foot-pad contact priority.
  robot = cfg.scene.entities["robot"]
  cfg.scene.entities["robot"] = dataclasses.replace(robot, spec_fn=v19_spec)

  # 2. Measured piecewise clamp (overwrites the v18 single-line event).
  enable_measured_torque_speed_clamp(cfg)

  # 3. Measured delay DR (21-65 ms), still per (env, servo).
  _set_per_servo_delay_range(cfg, DELAY_MIN_LAG_STEPS, DELAY_MAX_LAG_STEPS)

  # 4. Wide lost-motion deadband, drawn per (env, servo) at episode reset.
  enable_servo_deadband(cfg, V19_DEADBAND_RANGE)

  # 5. Shrunk strength DR (kp/kv scale ranges stay v17: worn-servo spread).
  strength = cfg.events["actuator_strength"].params
  strength["strength_scale_range"] = V19_STRENGTH_ENV_RANGE
  strength["strength_servo_range"] = V19_STRENGTH_SERVO_RANGE

  # 6. Wide surface friction (sliding coefficient, per env at startup).
  cfg.events["foot_friction"].params["ranges"] = V19_FRICTION_RANGE

  # 8. Control-rate joint-acc penalty (same weight; docstring point 8).
  cfg.rewards["joint_acc_l2"] = dataclasses.replace(
    cfg.rewards["joint_acc_l2"], func=local_mdp.joint_acc_control_rate_l2
  )

  # 9. Actor joint_vel at control rate (deploy-aligned; docstring point 9).
  #    Noise/scale/history settings carry over unchanged; critic untouched.
  actor_terms = cfg.observations["actor"].terms
  actor_terms["joint_vel"] = dataclasses.replace(
    actor_terms["joint_vel"], func=local_mdp.joint_vel_control_rate_rel
  )

  # 10. Unlock gates recalibrated to this plant (docstring point 10).
  if not play and "command_grid" in cfg.curriculum:
    grid_params = cfg.curriculum["command_grid"].params
    grid_params["gamma_lin"] = V19_GAMMA_LIN
    grid_params["gamma_ang"] = V19_GAMMA_ANG

  return cfg
