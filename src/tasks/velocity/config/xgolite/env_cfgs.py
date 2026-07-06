"""XGO-Lite2 walking velocity environment configurations.

Derived from the xgomini configs; differences:
- 12-DoF action space (arm frozen in the model)
- smaller foot clearance target and velocity command ranges (shorter legs,
  weaker servos)
"""

import dataclasses

from src.assets.robots import get_xgolite_robot_cfg
from src.tasks.velocity import mdp as local_mdp
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

FOOT_NAMES = ("fl", "fr", "bl", "br")
FOOT_PAD_GEOMS = tuple(f"{name}_foot_pad" for name in FOOT_NAMES)
THIGH_GEOMS = ("fl_thigh", "fr_thigh", "bl_thigh", "br_thigh")

# v14 body-pose command nominal: (body_pitch [rad, positive = nose up],
# base_height [m, root z above the floor; "stand" keyframe root z 0.1159]).
NOMINAL_POSE = (0.0, 0.116)

_POSE_STD = {
  "standing": {
    r".*(fl|fr|bl|br)_hip_joint.*": 0.05,
    r".*(fl|fr|bl|br)_thigh_joint.*": 0.1,
    r".*(fl|fr|bl|br)_calf_joint.*": 0.15,
  },
  "walking": {
    # 0.30 (was 0.15): lateral stepping and in-place turning need
    # +/-0.2-0.35 rad hip abduction; at 0.15 the pose term paid the policy
    # ~0.6/step to keep its hips frozen (v6 could not sidestep or turn well).
    r".*(fl|fr|bl|br)_hip_joint.*": 0.30,
    r".*(fl|fr|bl|br)_thigh_joint.*": 0.35,
    r".*(fl|fr|bl|br)_calf_joint.*": 0.5,
  },
}


def xgolite_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10          # 50 Hz control, matches the real deploy loop
  cfg.sim.mujoco.ccd_iterations = 100
  cfg.sim.contact_sensor_maxmatch = 128

  cfg.scene.entities = {"robot": get_xgolite_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_PAD_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(mode="geom", entity="robot", pattern=THIGH_GEOMS),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, nonfoot_ground_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.25

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 1.0
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = FOOT_NAMES
  cfg.observations["actor"].terms["phase"].params["period"] = 0.4

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_PAD_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)

  cfg.rewards["pose"].params["std_standing"] = _POSE_STD["standing"]
  cfg.rewards["pose"].params["std_walking"] = _POSE_STD["walking"]
  cfg.rewards["pose"].params["std_running"] = _POSE_STD["walking"]
  # 0.05 stand thresholds everywhere (pose posture, gait reward, command
  # zeroing, phase obs, deploy STAND_CMD_NORM): lateral commands top out
  # at 0.08, and at the default 0.1 a sidestep command was treated as
  # "standing" by all five gates — v11b never trained or executed lateral.
  cfg.rewards["pose"].params["walking_threshold"] = 0.05
  cfg.rewards["foot_gait"].params["command_threshold"] = 0.05

  # Adaptive tracking sigma (sigma_eff = std + std_gain*|cmd|): a fixed
  # sigma can't serve both ends of the command range — v8 (std 0.5)
  # ignored small commands (lateral dead, slow turns sloppy), v9
  # (std 0.25) flatlined on large ones (0.9 m/s never tracked, err 0.88).
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=local_mdp.track_linear_velocity_adaptive,
    weight=1.5,
    params={"command_name": "twist", "std": 0.15, "std_gain": 0.5},
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=local_mdp.track_angular_velocity_adaptive,
    weight=1.5,
    params={"command_name": "twist", "std": 0.2, "std_gain": 0.4},
  )
  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03
  cfg.rewards["foot_gait"].params["period"] = 0.4
  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["foot_gait"].weight = 0.7
  cfg.rewards["foot_slip"].weight = -0.15
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = FOOT_NAMES
  cfg.rewards["foot_clearance"].params["target_height"] = 0.02
  cfg.rewards["foot_clearance"].weight = -3
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = FOOT_NAMES

  cfg.rewards["nonfoot_contact"] = RewardTermCfg(
    func=mdp.illegal_contact,
    weight=-3,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 0.5},
  )
  # -0.5: hardware walk logs showed action step p50 0.32/tick at -0.25 —
  # too twitchy through the ~70 ms real actuation lag (standing limit cycle)
  cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.5)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def xgolite_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = xgolite_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Rebuild the twist command as the FORK's class: the base cfg uses the
  # upstream mjlab term, which silently ignores axis_focus_probs (in v8/v9
  # the attribute was set but never read — lateral exposure stayed ~1%).
  old_twist = cfg.commands["twist"]
  assert isinstance(old_twist, UniformVelocityCommandCfg)
  twist_cmd = local_mdp.UniformVelocityCommandCfg(
    **{f.name: getattr(old_twist, f.name) for f in dataclasses.fields(old_twist)}
  )
  # Ranges too: the fork Ranges adds the v14 body_pitch/base_height fields;
  # setting them on the copied upstream instance would only attach dynamic
  # attributes (invisible to dataclasses.asdict / config dumps).
  twist_cmd.ranges = local_mdp.UniformVelocityCommandCfg.Ranges(
    **{
      f.name: getattr(old_twist.ranges, f.name)
      for f in dataclasses.fields(old_twist.ranges)
    }
  )
  cfg.commands["twist"] = twist_cmd
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_standing_envs = 0.0
  # Hardware is driven at up to ~1.0 fwd and fast backward; v8 trained to
  # 0.7 max and became unstable/lost traction when extrapolating past 0.8.
  twist_cmd.ranges.lin_vel_x = (-0.8, 1.0)
  # +/-0.08: just above the measured ~0.045-0.05 m/s lateral capability
  # (and above the 0.05 stand gates, or lateral episodes become standing).
  twist_cmd.ranges.lin_vel_y = (-0.08, 0.08)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  twist_cmd.ranges.heading = None
  # 20% pure-rotation / 25% pure-lateral / 10% backward-only episodes
  # (uniform sampling alone gives lateral ~1% and rotation ~6% exposure;
  # lateral raised 15->25% after v8 hardware still ignored vy).
  twist_cmd.axis_focus_probs = (0.20, 0.25, 0.10)

  # v14: two body-pose command channels -> [vx, vy, wz, body_pitch,
  # base_height]. Flat task only: the height reward uses absolute root z.
  # v15: pitch extended nose-up 0.151 -> 0.33 (high-object camera/reach) and
  # the height range widened to the full standable envelope (0.080..0.141);
  # the non-rectangular (pitch, height) workspace is enforced by
  # pose_height_band below instead of shrinking the ranges to a safe box.
  twist_cmd.ranges.body_pitch = (-0.436, 0.33)    # rad, positive = nose up
  twist_cmd.ranges.base_height = (0.080, 0.141)   # m, root z above the floor
  twist_cmd.nominal_pose = NOMINAL_POSE
  # Standable height band per pitch (floor convention), from Quadruped-robot
  # body.py leg IK with 12% width margin (2026-07-07): nose-up needs extended
  # rear legs so the deep crouch disappears; max height lives near level pitch.
  twist_cmd.pose_height_band = (
    (-0.436, 0.095, 0.122),
    (-0.300, 0.088, 0.128),
    (-0.150, 0.081, 0.134),
    (0.000, 0.080, 0.141),
    (0.150, 0.092, 0.131),
    (0.250, 0.102, 0.125),
    (0.330, 0.110, 0.121),
  )
  # 50% nominal pose + twist sampled as before (incl. axis_focus) /
  # 25% pose-hold (random pose, twist zero) /
  # 25% posed walking (random pose, twist scaled by 0.5).
  twist_cmd.pose_mode_probs = (0.50, 0.25, 0.25)

  # v14 pose-tracking rewards (fork functions; command channels 3 and 4).
  cfg.rewards["track_body_pitch"] = RewardTermCfg(
    func=local_mdp.track_body_pitch,
    weight=1.0,
    params={"command_name": "twist", "std": 0.1},
  )
  cfg.rewards["track_base_height"] = RewardTermCfg(
    func=local_mdp.track_base_height,
    weight=1.0,
    params={"command_name": "twist", "std": 0.015},
  )
  # v13b lateral fix: the adaptive sigma pays standing ~84% of the tracking
  # reward at vy=0.08, so lateral asymptoted at ~0.012 m/s of the ~0.045
  # physical ceiling. A dedicated tight-sigma vy term restores the gradient
  # (standing -> 0.08, full capability -> 0.61). Always active.
  cfg.rewards["track_lateral_velocity"] = RewardTermCfg(
    func=local_mdp.track_lateral_velocity,
    weight=0.5,
    params={"command_name": "twist", "std": 0.05},
  )

  # v14 conflict fixes -- terms that would fight the pose commands:
  # 1. body_orientation_l2 penalizes ANY non-flat body; replace with the fork
  #    variant that penalizes pitch toward the COMMANDED pitch (roll still
  #    toward 0). Same weight; magnitude matches for small angles.
  ori = cfg.rewards["body_orientation_l2"]
  ori.func = local_mdp.body_orientation_cmd_l2
  ori.params["command_name"] = "twist"
  # 2. `pose` holds joints near stand defaults; relax its stds as the pose
  #    command deviates from nominal (see variable_posture docstring: at max
  #    pitch the stds grow ~4.5x, keeping regularization without fighting
  #    the tracking terms, which dominate near their tight sigmas).
  cfg.rewards["pose"].params["pose_relax_gain"] = 8.0
  cfg.rewards["pose"].params["pose_dev_height_scale"] = 20.0
  cfg.rewards["pose"].params["nominal_pose"] = NOMINAL_POSE
  # 3. stand_still drags joints to the stand default in every low-twist
  #    episode, including pose_hold crouches; gate it to (near-)nominal pose
  #    commands only. 0.02 rad-equivalent: 50% of episodes are nominal
  #    EXACTLY (dev 0); anything past ~0.02 is a deliberate pose.
  cfg.rewards["stand_still"].params["pose_dev_threshold"] = 0.02
  cfg.rewards["stand_still"].params["pose_dev_height_scale"] = 20.0
  cfg.rewards["stand_still"].params["nominal_pose"] = NOMINAL_POSE
  # fell_over margin check: at the pitch extreme -0.436 rad, projected
  # gravity z is -cos(0.436) = -0.906 -> acos(0.906) = 25 deg, well inside
  # the 70 deg bad_orientation limit. No termination change needed.

  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum.pop("command_vel", None)

  return cfg
