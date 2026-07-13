"""XGOLite-Precision: full-range tracking-ACCURACY preset (2026-07-11).

Spec: docs/research/speed-accuracy-analysis.md section 3 (main repo). The
aggressive presets push the speed envelope; this preset fixes the measured
v17 accuracy holes WITHOUT touching the envelope:

- R1 gate sandwich: stand_still / foot_clearance / foot_slip / soft_landing
  gated at 0.1 while foot_gait / phase / deploy STAND_CMD_NORM gate at 0.05
  left commands in (0.05, 0.10] half-standing. All gates now flip at 0.05.
- R2 flat error pricing: the fixed tracking sigma 0.10 prices error in
  absolute m/s, so the measured slow-band shortfalls (vx 0.08 -> 0.059,
  ratio 0.74) forfeited ~0.065 weight-units — essentially free. Relative
  sigma (0.25 * ||cmd||, clipped to [0.02, 0.10]) prices RELATIVE error
  uniformly; angular tracking tightened the same way (std 0.2 -> 0.05,
  std_gain 0.4 -> 0.15).
- R3 unmodeled stiction: per-joint dof_frictionloss DR 0.005-0.040 N*m
  (2-18% of the 0.22 N*m forcerange) + damping DR 0.03-0.10 around the
  0.05 nominal; per-joint independent draws supply the side-asymmetric
  breakaway the hardware shows. XML nominal stays untouched.
- R4 exposure: more slow-band mass hugging the 0.05 gate, vy sampled past
  the deploy full-scale, turn-at-speed pairs, shorter dwells and mild
  velocity-jump inits for transition/decel training.
- R7 gyro noise floor: white noise +-0.2 -> +-0.1 rad/s (still ~5x the real
  sensor); the per-episode bias DR is measured-bias coverage and stays.

Explicitly UNCHANGED from v17 (see spec 3.6): actuator truth values, plant
and strength DR, track_yaw_zero and the lateral term, the 0.4 s clock +
phase contract, action_rate -0.5, symmetry augmentation, PPO config, the
(-0.40, 0.45) vx envelope. Pure delta on ``xgolite_flat_env_cfg()`` —
``diff`` against that function is the full spec.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.dr import joint as dr_joint
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.tasks.velocity import mdp as local_mdp

from .env_cfgs import xgolite_flat_env_cfg


def xgolite_precision_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = xgolite_flat_env_cfg(play=play)
  twist = cfg.commands["twist"]
  assert isinstance(twist, local_mdp.UniformVelocityCommandCfg)

  # --- 3.1 gate alignment (fixes R1) -----------------------------------
  # Every reward gate now flips at the same 0.05 twist norm as the phase
  # clock, foot_gait, pose.walking_threshold, the resample-zeroing gate and
  # deploy STAND_CMD_NORM — the (0.05, 0.10] sandwich zone (walk clock +
  # gait reward active, but slip/clearance/impact free and stand_still
  # dragging joints to the stand pose) disappears.
  cfg.rewards["stand_still"].params["command_threshold"] = 0.05     # was 0.1
  cfg.rewards["foot_clearance"].params["command_threshold"] = 0.05  # was 0.1
  cfg.rewards["foot_slip"].params["command_threshold"] = 0.05       # was 0.1
  cfg.rewards["soft_landing"].params["command_threshold"] = 0.05    # was 0.1
  # track_yaw_zero.command_threshold stays 0.1: it gates on |wz_cmd| only
  # and must keep covering slow-turn commands.

  # --- 3.2 speed-scaled tracking stds (fixes R2) ------------------------
  # sigma_eff = clip(0.25 * ||cmd_xy||, 0.02, 0.10); std_max when standing.
  # The measured shortfalls stop being free: vx 0.08 -> 0.059 forfeits
  # 1.00 wu (was 0.06), vx 0.06 -> 0.015 forfeits 1.49 (was 0.27), while
  # the well-tracked band above 0.40 is unchanged (sigma caps at 0.10).
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=local_mdp.track_linear_velocity_relative,
    weight=1.5,
    params={"command_name": "twist", "rel": 0.25, "std_min": 0.02, "std_max": 0.10},
  )
  # Angular: same medicine via the existing adaptive term — sigma_eff
  # 0.05 + 0.15|wz| (was 0.2 + 0.4|wz|): wz 0.30 -> 0.195 forfeits 1.06 wu
  # (was 0.15). Tuning guard (spec 3.2): if pure-lateral episodes
  # reward-starve (mean track_linear_velocity < 0.2 in the lat bucket),
  # raise std_min to 0.03 — track_lateral_velocity stays as the backstop.
  cfg.rewards["track_angular_velocity"].params["std"] = 0.05       # was 0.2
  cfg.rewards["track_angular_velocity"].params["std_gain"] = 0.15  # was 0.4

  # --- 3.3 stiction / friction DR (fixes R3) ----------------------------
  # Per-joint INDEPENDENT draws (dr helpers default shared_random=False):
  # direction- and side-asymmetric breakaway, the regime behind the slow-
  # dwell heading drift. 0.005-0.040 N*m = 2-18% of the 0.22 N*m
  # forcerange, bracketing plausible Feetech static friction; the XML
  # nominal (frictionloss 0.001) stays untouched — E2 actuator ID should
  # recenter this range, not narrow it to zero. joint_names ".*" matches
  # the 12 actuated joints only (Entity.joint_names excludes the free
  # joint). No backlash modeling (not natively supported; E2 owns that).
  cfg.events["joint_friction"] = EventTermCfg(
    func=dr_joint.joint_friction,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      "ranges": (0.005, 0.040),
      "operation": "abs",
    },
  )
  cfg.events["joint_damping"] = EventTermCfg(
    func=dr_joint.joint_damping,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      "ranges": (0.03, 0.10),  # nominal 0.05
      "operation": "abs",
    },
  )

  # --- 3.4 sampling curriculum (fixes R4, transition exposure) -----------
  twist.axis_focus_probs = (0.15, 0.25, 0.10)  # rot 0.20 -> 0.15; lat, back kept
  twist.slow_vx_prob = 0.15                    # was 0.10
  # Hug the 0.05 gate and cover the measured ratio dip (band edge must stay
  # above the gate or slow episodes degrade to standing).
  twist.slow_vx_band = (0.055, 0.14)           # was (0.06, 0.12)
  twist.turn_at_speed_prob = 0.10              # NEW here: vx x strong wz pairs
  twist.turn_vx_band = (0.15, 0.45)
  twist.turn_wz_band = (0.5, 1.0)              # inside the kept +-1.0 wz range
  # Deploy vy full scale (0.08) off the sample edge.
  twist.ranges.lin_vel_y = (-0.10, 0.10)       # was +-0.08
  twist.resampling_time_range = (2.0, 5.0)     # was (3.0, 8.0): more edges
  # Agile's decel/momentum-catch trick, mild dose.
  twist.init_velocity_prob = 0.10

  # --- 3.5 observation noise (fixes R7) ---------------------------------
  # +-0.10 is still ~5x the real gyro white noise; it lowers the trainable
  # yaw-nulling floor from ~0.03 to ~0.015 rad/s without pretending the
  # sensor is clean. The per-episode bias DR (+-0.03 abs) is measured-bias
  # coverage and stays untouched.
  cfg.observations["actor"].terms["base_ang_vel"].noise.noise_cfg = Unoise(
    n_min=-0.10, n_max=0.10
  )

  return cfg
