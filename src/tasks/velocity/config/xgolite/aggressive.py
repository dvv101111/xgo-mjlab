"""Aggressive XGO-Lite2 locomotion presets (2026-07-11 campaign).

Motivation (deploy-measured): the v17 gait is only stable at 1.0x of its
nominal speed profile — command scaling extrapolates badly (forward
saturates at ~0.19-0.26 m/s true, backward deadbands below 0.06, and the
2026-07-11 sim probe shows the same shape: achieved vx peaks ~0.24 near
command 0.4-0.45 and DROPS when over-driven). These presets TRAIN the
speeds we want instead of scaling at deploy, and probe what the hardware
can actually handle.

All presets are pure deltas on ``xgolite_flat_env_cfg()`` (the v17
baseline) so ``diff`` against that function is the full spec. Invariants
kept everywhere (physics + deploy contract, not aesthetics):

- actuator model untouched: kp 5.0 / kv 0.12 / 0.22 N*m forcerange /
  60-100 ms delay are hardware-truth values;
- v17 plant DR untouched (foot friction, encoder bias, base CoM,
  per-env + per-servo strength, sensor-bias DR);
- 49-dim obs frame, 5-dim command, phase term present, history 5 — the
  deploy PolicyRunner contract; only the phase PERIOD may change and is
  exported as ONNX metadata ``phase_period`` (runner.py);
- fall terminations (fell_over 70 deg, illegal_contact thigh > 10 N) and
  joint_pos_limits respect stay in every preset;
- stand gate 0.05 and stand_still stay: the robot must still stand safely
  at zero command between hardware test runs.

Preset summary:

============ ========== ============================= =====================
preset       gait clock what it frees/pushes          deploy-side change
============ ========== ============================= =====================
Sprint       0.4 s      command range + fat high-|vx| widen CMD_LIMITS only
                        sampling, relaxed smoothness/
                        impact penalties
FastClock    0.28 s     stride FREQUENCY (small       read phase_period
                        quadrupeds gain speed from    metadata + widen
                        cycle rate), v17 rewards      CMD_LIMITS
FreeGait     0.4 s obs, no gait/clearance/pose        widen CMD_LIMITS;
             no gait    shaping — RL discovers the    pose sliders inert
             reward     gait; symmetry aug OFF        (trained nominal-only)
Agile        0.4 s      fast command switching,       widen CMD_LIMITS
                        turning at speed, stronger    (notably wz)
                        pushes, velocity-jump inits
============ ========== ============================= =====================
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.velocity import mdp as local_mdp

from .env_cfgs import xgolite_flat_env_cfg
from .rl_cfg import xgolite_ppo_runner_cfg

# 2026-07-11 sim saturation probe (scripts/vx_saturation.py on the v17
# checkpoint 2026-07-11_14-16-46/model_1499, 128 envs x 6 s, full DR):
#   fwd:  0.20->0.176  0.30->0.223  0.40->0.243  0.45->0.244 (PEAK)
#         0.55->0.241  0.65->0.238  0.80->0.228  1.00->0.219
#   back: -0.15->-0.119  -0.25->-0.174 (PEAK)  -0.35->-0.169
#         -0.45->-0.152  -0.60->-0.131
#   yaw:  1.0->0.868  1.4->1.143 ; turn (0.4, wz 1.0) -> vx 0.222 @ wz 0.861
# Same shape as the hardware ladders: over-driving DEGRADES speed. The
# ceilings are reward-shaped (trot at the 0.4 s clock + v17 penalties),
# not raw physics. Ranges below give the presets headroom over the v17
# saturation while fat top-band sampling keeps the high commands trained
# rather than gradient-dead.


def _relax_pose_reward(cfg: ManagerBasedRlEnvCfg, weight: float) -> None:
  """De-weight the joint-posture prior and let 'running' stds engage.

  v17 never used std_running (running_threshold 1.5 > any command norm);
  aggressive presets sample twist norms past 0.5, where the walking stds
  (thigh 0.35) actively fight long strides.
  """
  cfg.rewards["pose"].weight = weight
  cfg.rewards["pose"].params["running_threshold"] = 0.5
  cfg.rewards["pose"].params["std_running"] = {
    r".*(fl|fr|bl|br)_hip_joint.*": 0.4,
    r".*(fl|fr|bl|br)_thigh_joint.*": 0.6,
    r".*(fl|fr|bl|br)_calf_joint.*": 0.8,
  }


##
# SPRINT — velocity-first, keeps the 0.4 s deploy clock (drop-in testable).
##


def xgolite_sprint_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = xgolite_flat_env_cfg(play=play)
  twist = cfg.commands["twist"]
  assert isinstance(twist, local_mdp.UniformVelocityCommandCfg)

  # Command envelope: ~2x the measured v17 sim saturation (fwd 0.24 / back
  # 0.17). Backward gets a real range + dedicated sampling so the deploy
  # deadband below 0.06 true is trained away, not remapped away.
  twist.ranges.lin_vel_x = (-0.55, 0.75)
  # Fat sampling at high |vx| (30%) + keep minority-direction coverage.
  twist.axis_focus_probs = (0.10, 0.10, 0.10)
  twist.slow_vx_prob = 0.05
  twist.fast_vx_prob = 0.30
  twist.fast_vx_band_fwd = (0.40, 0.75)
  twist.fast_vx_band_back = (-0.55, -0.30)
  twist.fast_vx_fwd_frac = 0.65
  # Sprint is a twist preset: pose channels stay trained (obs must not go
  # OOD when the operator moves sliders) but only lightly.
  twist.pose_mode_probs = (0.90, 0.05, 0.05)
  cfg.rewards["track_body_pitch"].weight = 0.5
  cfg.rewards["track_base_height"].weight = 0.5

  # Tracking: the v16 fixed sigma 0.10 was chosen FOR the (-0.40, 0.45)
  # envelope; at cmd 0.75 it is gradient-dead (exp(-(0.75-0.24)^2/0.01) ~ 0).
  # Adaptive sigma keeps the top band shaped (sigma_eff 0.10 + 0.30|cmd| =
  # 0.33 at 0.75) while small commands keep the tight v17 shaping.
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=local_mdp.track_linear_velocity_adaptive,
    weight=2.0,
    params={"command_name": "twist", "std": 0.10, "std_gain": 0.30},
  )

  # Relax smoothness/impact/style — exploit the real actuator envelope.
  # Falls in sim are information; falls on hardware are accepted.
  cfg.rewards["action_rate_l2"].weight = -0.15   # was -0.5
  cfg.rewards["joint_acc_l2"].weight = -1e-7     # was -2.5e-7
  cfg.rewards["body_ang_vel"].weight = -0.02     # was -0.08
  cfg.rewards["angular_momentum"].weight = -0.01  # was -0.03
  cfg.rewards["foot_slip"].weight = -0.05        # was -0.15
  cfg.rewards["foot_clearance"].weight = -1.0    # was -3
  cfg.rewards["foot_clearance"].params["target_height"] = 0.03  # taller swing
  cfg.rewards["soft_landing"].weight = 0.0       # impact penalty OFF
  cfg.rewards["nonfoot_contact"].weight = -1.0   # was -3 (termination stays)
  # Gait clock kept at 0.4 s (deploy drop-in) but the trot prior is
  # weakened so stride length/timing can stretch at speed.
  cfg.rewards["foot_gait"].weight = 0.4          # was 0.7
  _relax_pose_reward(cfg, weight=0.5)

  return cfg


##
# FAST-CLOCK — v17 stability rewards, 0.28 s gait clock.
##

FASTCLOCK_PERIOD = 0.28  # s; 3.6 Hz stride vs the v17 2.5 Hz


def xgolite_fastclock_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = xgolite_flat_env_cfg(play=play)
  twist = cfg.commands["twist"]
  assert isinstance(twist, local_mdp.UniformVelocityCommandCfg)

  # The experiment: small quadrupeds gain speed from stride FREQUENCY.
  # v17 saturates at ~0.24 m/s = stride 0.096 m x 2.5 Hz; at 3.6 Hz the
  # same stride predicts ~0.34 m/s IF the servos can track the cycle
  # (swing time 0.12 s vs the 60-100 ms actuation delay — that is exactly
  # the "how much can hardware handle" question).
  cfg.observations["actor"].terms["phase"].params["period"] = FASTCLOCK_PERIOD
  cfg.rewards["foot_gait"].params["period"] = FASTCLOCK_PERIOD

  # Command range up, sized to the frequency-scaling prediction.
  twist.ranges.lin_vel_x = (-0.45, 0.60)
  twist.axis_focus_probs = (0.15, 0.15, 0.10)
  twist.slow_vx_prob = 0.10
  twist.fast_vx_prob = 0.20
  twist.fast_vx_band_fwd = (0.30, 0.60)
  twist.fast_vx_band_back = (-0.45, -0.25)
  twist.fast_vx_fwd_frac = 0.70

  # Mild adaptive sigma: the fixed 0.10 was tuned for the 0.45 envelope;
  # at 0.60 the top band needs a nonzero gradient (sigma_eff 0.22 at 0.60).
  # All other rewards stay at v17 values — this preset isolates the clock.
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=local_mdp.track_linear_velocity_adaptive,
    weight=1.5,
    params={"command_name": "twist", "std": 0.10, "std_gain": 0.20},
  )

  return cfg


##
# FREE-GAIT — drop the gait-shaping entirely; RL discovers the gait.
##


def xgolite_freegait_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = xgolite_flat_env_cfg(play=play)
  twist = cfg.commands["twist"]
  assert isinstance(twist, local_mdp.UniformVelocityCommandCfg)

  # Widest envelope of the family: with no gait prior the policy may find
  # bounding/pronking regimes past the trot's stride ceiling.
  twist.ranges.lin_vel_x = (-0.55, 0.85)
  twist.axis_focus_probs = (0.10, 0.10, 0.10)
  twist.slow_vx_prob = 0.05
  twist.fast_vx_prob = 0.30
  twist.fast_vx_band_fwd = (0.40, 0.85)
  twist.fast_vx_band_back = (-0.55, -0.30)
  twist.fast_vx_fwd_frac = 0.70
  # Pose channels pinned to nominal: the command stays 5-dim (deploy
  # contract) but this policy does not learn pitch/height tracking —
  # operator pose sliders will be inert on this preset.
  twist.pose_mode_probs = (1.0, 0.0, 0.0)

  # --- rewards: velocity tracking + survival + minimal smoothness only ---
  # The phase OBS keeps ticking exactly like deploy does (0.4 s clock,
  # 0.05 stand gate) so the frame stays contract-identical; with foot_gait
  # gone there is no reward tying the policy to it — a free clock signal
  # it can entrain to or ignore.
  for name in (
    "foot_gait",        # the gait prior IS the experiment control
    "soft_landing",     # impact aesthetics
    "body_ang_vel",     # pronking/bounding needs body pitch rate
    "angular_momentum",
    "pose",             # joint-posture prior
    "track_body_pitch",   # pose channels untrained (nominal-only)
    "track_base_height",
  ):
    del cfg.rewards[name]

  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=local_mdp.track_linear_velocity_adaptive,
    weight=2.0,
    params={"command_name": "twist", "std": 0.10, "std_gain": 0.25},
  )
  # Keep drift shaping: a discovered gait that yaws while "going straight"
  # is useless on hardware.
  cfg.rewards["track_yaw_zero"].weight = 0.3
  # Minimal smoothness. Slip/clearance kept as TINY exploit guards only:
  # zero slip cost rewards skating and zero clearance cost rewards foot
  # dragging — both sim exploits that cannot transfer, not gaits.
  cfg.rewards["action_rate_l2"].weight = -0.10
  cfg.rewards["joint_acc_l2"].weight = -1e-7
  cfg.rewards["foot_slip"].weight = -0.05
  cfg.rewards["foot_clearance"].weight = -0.5
  cfg.rewards["nonfoot_contact"].weight = -1.0
  # Orientation: keep a LIGHT flat-body pull (was -1.0) — the 70 deg
  # termination alone leaves a 40-deg-nose-down local optimum open.
  cfg.rewards["body_orientation_l2"].weight = -0.5

  return cfg


##
# AGILE — accel/decel and turning at speed, 0.4 s clock (drop-in).
##


def xgolite_agile_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = xgolite_flat_env_cfg(play=play)
  twist = cfg.commands["twist"]
  assert isinstance(twist, local_mdp.UniformVelocityCommandCfg)

  # Short dwells: tracking reward integrated over a 1.5-4 s dwell makes
  # convergence SPEED the dominant term (v17's 3-8 s dwells amortize a
  # slow ramp-in almost completely).
  twist.resampling_time_range = (1.5, 4.0)
  # Velocity-jump inits: 20% of resamples teleport the base to the new
  # commanded velocity — the policy must catch and stabilize an abrupt
  # momentum change (decel training is otherwise unreachable in sim
  # because commands change but momentum does not).
  twist.init_velocity_prob = 0.20

  twist.ranges.lin_vel_x = (-0.45, 0.55)
  # Yaw envelope up 1.0 -> 1.5 rad/s; the v17 hardware ladder reached
  # 0.856 rad/s true at command 1.0 without saturating cleanly.
  twist.ranges.ang_vel_z = (-1.5, 1.5)
  twist.axis_focus_probs = (0.15, 0.15, 0.10)
  twist.slow_vx_prob = 0.05
  twist.fast_vx_prob = 0.10
  twist.fast_vx_band_fwd = (0.35, 0.55)
  twist.fast_vx_band_back = (-0.45, -0.25)
  # 20% mixed vx+wz episodes: turning at speed, the regime uniform
  # sampling almost never pairs strongly.
  twist.turn_at_speed_prob = 0.20
  twist.turn_vx_band = (0.15, 0.55)
  twist.turn_wz_band = (0.5, 1.5)
  # Twist-focused pose mix (posed_walk halves the twist, diluting agility).
  twist.pose_mode_probs = (0.80, 0.10, 0.10)
  cfg.rewards["track_body_pitch"].weight = 0.75
  cfg.rewards["track_base_height"].weight = 0.75

  # Tracking weights up (transient error is the objective).
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].weight = 2.0

  # Transients need action authority; moderate (not sprint-level) relax.
  cfg.rewards["action_rate_l2"].weight = -0.25
  cfg.rewards["body_ang_vel"].weight = -0.04
  cfg.rewards["angular_momentum"].weight = -0.015
  _relax_pose_reward(cfg, weight=0.75)

  # Stronger pushes: recovery-at-speed robustness.
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.7, 0.7),
      "y": (-0.7, 0.7),
      "z": (-0.4, 0.4),
      "roll": (-0.52, 0.52),
      "pitch": (-0.52, 0.52),
      "yaw": (-1.2, 1.2),
    }

  return cfg


##
# RL configs.
##


def xgolite_aggressive_ppo_runner_cfg(
  experiment_name: str,
  max_iterations: int = 1500,
  symmetry: bool = True,
) -> RslRlOnPolicyRunnerCfg:
  """v17 PPO config with a per-preset experiment dir.

  ``symmetry=False`` (FreeGait) removes the L/R mirror augmentation +
  mirror loss so asymmetric gaits (gallop, lateral-sequence) are not
  penalized at the source.
  """
  cfg = xgolite_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  cfg.max_iterations = max_iterations
  if not symmetry:
    cfg.algorithm.symmetry_cfg = None  # type: ignore[attr-defined]
  return cfg
