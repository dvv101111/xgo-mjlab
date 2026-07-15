"""XGOLite-V20: the v18range recipe on the CORRECTED measured plant (fit v4).

v19 trained against servo_fit_v3 — fitted on v2.3-firmware-poisoned
captures (~37-52 Hz real per-servo sampling, ~21-25 % silently dropped
commands) that masqueraded as a stiff torque-clamped relay (kp 39.7,
kd 0.007, armature 0, qd_max 12.1). Fit v4 on clean v2.4 captures with
the CAD-derived mass model overturned that: the real servo is a soft
damped spring (kp 5.59, kd 0.074, damping 0.004, armature 5.6e-4,
frictionloss 0.017, qd_max 10.46) — close in kind to the hand-set plant
that trained v15-v18. measured_actuators.py is regenerated from
servo_fit_v4_perjoint.json, so every measured-plant hook v19 installed
(spec_fn dynamics, piecewise clamp, delay DR 9-31 steps) now writes the
v4 values, and build_model.py bakes the CAD inertials into the XML
(total 577 g measured, leg 49.8 g, swing inertia -45 %).

v20 therefore KEEPS from v19 (plant-independent or still-measured):
- the measured plant wiring itself (points 1-3 of v19.py, now v4 values);
- wide lost-motion deadband DR (0, 0.025): v4 fitted nominals are
  0.009-0.016 rad and the hysteresis captures show 0.010-0.025 rad/side
  — the bracket is now centered on measurement, not on a guess;
- shrunk strength DR + wide friction DR (points 5-6);
- control-rate joint_acc penalty (point 8) and control-rate ACTOR
  joint_vel obs (point 9): deploy-aligned regardless of plant — the
  hardware loop feeds finite-differenced telemetry positions, never the
  instantaneous physics-rate velocity.

v20 REVERTS the two knobs that were sized to the artifact plant:
- init_std back to the v18 default 1.0 (v19 point 7 cut it to 0.4
  because kp 39.7 turned std-1.0 noise into ~14 rad/s flailing; on a
  kp-5.6 spring the same noise caps near the v18 plant's ~1.7 rad/s);
- grid unlock gates back to the v18-calibrated (0.70, 0.55) (v19
  point 10 dropped them to 0.15/0.17 because the relay plant's real
  0.13 rad/s standing yaw wobble consumed the angular sigma budget —
  but those gates sit BELOW the zero-action idle score, so v19 v5
  degenerated to free full-grid expansion. On the soft plant the v18
  gates were validated end-to-end by v18range v2: 76.2 % unlocked with
  expansion ordering intact).

The run doubles as the plant-correction experiment: watch the standing
yaw wobble floor (v3 plant 0.13 rad/s vs old empirical ~0.03 — measured
post-run by scripts/diag_v19_gate_feasibility.py --task XGOLite-V20
--obs control) and whether seed_tracking clears the restored gates.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from .v19 import xgolite_v19_env_cfg

# v18-calibrated unlock gates, restored (see module docstring).
V20_GAMMA_LIN = 0.70
V20_GAMMA_ANG = 0.55


def xgolite_v20_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """v18range PPO cfg, unmodified (init_std back to the default 1.0)."""
  from .aggressive import xgolite_aggressive_ppo_runner_cfg

  return xgolite_aggressive_ppo_runner_cfg("xgolite_v20")


def xgolite_v20_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = xgolite_v19_env_cfg(play=play)
  # Restore the v18-calibrated unlock gates over v19's relay-plant values.
  if not play and "command_grid" in cfg.curriculum:
    grid_params = cfg.curriculum["command_grid"].params
    grid_params["gamma_lin"] = V20_GAMMA_LIN
    grid_params["gamma_ang"] = V20_GAMMA_ANG
  return cfg
