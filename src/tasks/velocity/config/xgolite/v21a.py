"""XGOLite-V21A: gait discovery on the v20 plant (v21 campaign stage 1).

Everything plant/curriculum/PPO-side is v20 (fit-v4 measured soft-spring
plant, grid gates 0.70/0.55, control-rate obs/penalty); V21A changes the
GAIT STACK per docs/research/v21-terrain-gait-litreview-2026-07-15.md
(shortlist B, items B1-B4). The 49-dim obs contract is frozen — every
change below is computable from signals the deploy loop already has.

1. ORC-style phase-contact reward (B1, arXiv:2402.08662) replaces the
   binary ``feet_gait`` match: saturated per-foot GRF weighted by a smooth
   stance/swing phase weight (``mdp.rewards.gait_contact_orc``). The ORC
   ablation is the FreeGait autopsy — phase observation ON + phase reward
   OFF gives "unbalanced gaits (2-3 legs)" (our rear-shuffle); balanced
   4-leg use needs the CONTACT reward, graded by force, not a boolean.
   Weight 0.7 = the trot-reward scale it replaces (reward in [-1, 1] vs
   feet_gait's [0, 1]; the lit review gives no alternative anchor).
   ``force_scale`` 2.8 N ~ mg/2 for the 577 g robot: a foot carrying its
   trot-pair share of body weight saturates the signal.

2. Speed/direction-scheduled gait spec (B2 + B3), no new command dims:
   - Gait member (B3): lateral-dominant command (|vy| > |vx| and
     |vy| > 0.05, the stand-gate threshold) selects a lateral-sequence
     WALK — 4-beat footfall LH-LF-RH-RF, offsets (fl, fr, bl, br) =
     (0.75, 0.25, 0.0, 0.5), duty 0.75 (">= 3 feet down" static walk,
     lit review section 2.0: at our Fr ~ 0.1-0.2 quadrupeds walk, and
     walk is the lowest-peak-torque gait for 0.22 N*m servos). All other
     commands keep the diagonal trot (0, 0.5, 0.5, 0), duty 0.5. The
     fixed diagonal-only offset is the strongest available hypothesis for
     the +-0.06 m/s lateral ceiling (section 2.5).
   - Stepping frequency (B2): f(speed_equiv) piecewise-linear between
     knots (0.10 m/s, 1.6 Hz) and (0.45 m/s, 2.5 Hz), clamped outside;
     speed_equiv = ||v_xy_cmd|| + 0.10 * |wz_cmd| (0.10 m/rad ~ the foot
     turning radius, so pure rotation is not stuck at the slow anchor).
     Low anchor 1.6 Hz: the lit-review walk band ("~1.5-1.75 Hz near
     stand-adjacent speeds") — dynamic similarity at Fr ~ 0.2 says walk,
     and a longer cycle shrinks the 41-120 ms latency fraction (10-30%
     of the 0.4 s cycle; section 2.1: stay at the LOW frequency end).
     High anchor 2.5 Hz at 0.45 m/s: the validated v17-v20 trot clock at
     the envelope edge — the schedule never exceeds what deploy has
     already proven. Implemented as a per-env stateful accumulator
     (``mdp.observations.phase_scheduled``): phase += f(cmd)*dt, reset 0,
     frozen + zeroed below twist norm 0.05 (same contract as the global
     clock it replaces; obs stays 2-dim sin/cos). Exported as ONNX
     metadata ``phase_schedule`` (rl/runner.py) for the deploy-parity
     task; ../src/xgo/openfw/locomotion.py is deliberately NOT touched.

3. Symmetry (B4): the hard mirror loss + augmentation are OFF (their
   phase mirror (sin,cos) -> (-sin,-cos) hard-assumes the trot's
   half-cycle equivalence — wrong for the walk member); replaced by the
   morphological-symmetry REWARD ``gait_lr_symmetry`` (Ding et al.
   arXiv:2403.10723: dropping it raises gait-consistency error 0.2->0.4,
   "prevents limping"): mirrored joints half a gait period apart, exp
   kernel std 0.3 rad, weight 0.3 (insurance-term scale, below the 0.7
   gait reward — symmetry should shape, not dominate).

4. Anti-degeneracy guards (the FreeGait lesson, section 2.4):
   feet_clearance / feet_slip / soft_landing stay at v20 levels;
   ``feet_air_time`` (existing, unused) is enabled at weight 0.1 —
   MuJoCo Playground's Go1 Joystick ships it at exactly +0.1 (section
   2.7) as a swing-phase guarantee; threshold 0.25 s ~ mid-band swing
   time across the schedule (0.2 s at 2.5 Hz trot to 0.31 s at 1.6 Hz).
   Its single-stance gate makes it trot-only (inert for the 3-feet-down
   walk) — a guard, not a shaper. Energy/torque penalties stay at v20
   levels (energy-solo degeneracy: three independent ablations).

5. vy widened to +-0.20 (was +-0.08): the walk member needs lateral
   commands past the old ceiling to matter. Grid stays vx x wz (vy is
   not gridded). The axis_focus lottery is structurally OFF under the
   grid sampler, so pure-lateral exposure comes from the new
   grid-compatible ``lateral_focus_prob`` 0.15, |vy| ~ U(0.08, 0.20)
   (velocity_command.py) — restoring roughly the exposure share v17's
   25% lottery gave lateral, scaled down because unfocused draws over
   the widened vy range are now also informative.
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.velocity import mdp as local_mdp

from .v20 import xgolite_v20_env_cfg

# Stepping-frequency schedule f(speed_equiv) [ (m/s, Hz), ... ]; see
# module docstring point 2. Exported to deploy as ONNX phase_schedule.
V21A_FREQ_KNOTS = ((0.10, 1.6), (0.45, 2.5))
V21A_WZ_EQUIV = 0.10   # m/rad; speed_equiv = ||v_xy|| + wz_equiv * |wz|
V21A_STAND_NORM = 0.05  # twist-norm freeze gate (project-wide stand gate)

# Gait members (foot order fl, fr, bl, br; offsets added to the base
# phase, stance = leg_phase < duty). Trot = the v17-v20 diagonal pair;
# walk = lateral-sequence LH-LF-RH-RF at quarter-cycle spacing.
V21A_TROT_OFFSETS = (0.0, 0.5, 0.5, 0.0)
V21A_TROT_DUTY = 0.5
V21A_WALK_OFFSETS = (0.75, 0.25, 0.0, 0.5)
V21A_WALK_DUTY = 0.75
V21A_LAT_THRESHOLD = 0.05  # |vy| dominance gate for the walk member

V21A_FORCE_SCALE = 2.8       # N, ~ mg/2 at 577 g — GRF saturation point
V21A_LIN_VEL_Y = (-0.20, 0.20)
V21A_LATERAL_FOCUS_PROB = 0.15
V21A_LATERAL_FOCUS_BAND = (0.08, 0.20)


def xgolite_v21a_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """v20 PPO cfg with the L/R mirror loss + augmentation OFF (point 3)."""
  from .aggressive import xgolite_aggressive_ppo_runner_cfg

  return xgolite_aggressive_ppo_runner_cfg("xgolite_v21a", symmetry=False)


def xgolite_v21a_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = xgolite_v20_env_cfg(play=play)
  twist = cfg.commands["twist"]
  assert isinstance(twist, local_mdp.UniformVelocityCommandCfg)

  # 5. vy widening + grid-compatible pure-lateral focus episodes.
  twist.ranges.lin_vel_y = V21A_LIN_VEL_Y
  twist.lateral_focus_prob = V21A_LATERAL_FOCUS_PROB
  twist.lateral_focus_band = V21A_LATERAL_FOCUS_BAND

  # 2. Scheduled per-env phase clock, actor AND critic (env_cfgs.py builds
  # the critic terms from the actor dict, so both carry the phase term;
  # they must tell the same story). Separate cfg objects per group — the
  # manager instantiates one stateful accumulator per group, and both are
  # computed once per step, staying in lockstep. Obs stays 2-dim sin/cos.
  phase_params = {
    "command_name": "twist",
    "freq_knots": V21A_FREQ_KNOTS,
    "wz_equiv": V21A_WZ_EQUIV,
    "stand_norm": V21A_STAND_NORM,
  }
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms["phase"] = dataclasses.replace(
      terms["phase"],
      func=local_mdp.phase_scheduled,
      params=dict(phase_params),
    )

  # 1. ORC phase-contact reward replaces the binary feet_gait match.
  del cfg.rewards["foot_gait"]
  cfg.rewards["gait_contact"] = RewardTermCfg(
    func=local_mdp.gait_contact_orc,
    weight=0.7,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "trot_offsets": V21A_TROT_OFFSETS,
      "walk_offsets": V21A_WALK_OFFSETS,
      "trot_duty": V21A_TROT_DUTY,
      "walk_duty": V21A_WALK_DUTY,
      "lat_threshold": V21A_LAT_THRESHOLD,
      "force_scale": V21A_FORCE_SCALE,
      "command_threshold": 0.05,
    },
  )

  # 3. Morphological-symmetry reward (replaces the mirror loss/aug).
  cfg.rewards["gait_symmetry"] = RewardTermCfg(
    func=local_mdp.gait_lr_symmetry,
    weight=0.3,
    params={
      "command_name": "twist",
      "std": 0.3,
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      "buffer_capacity": 32,
    },
  )

  # 4. feet_air_time anti-degeneracy guard (trot swing-phase guarantee).
  cfg.rewards["feet_air_time"] = RewardTermCfg(
    func=local_mdp.feet_air_time,
    weight=0.1,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold": 0.25,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )

  return cfg
