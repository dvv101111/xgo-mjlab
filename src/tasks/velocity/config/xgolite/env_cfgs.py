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
  cfg.commands["twist"] = twist_cmd
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_standing_envs = 0.0
  # Hardware is driven at up to ~1.0 fwd and fast backward; v8 trained to
  # 0.7 max and became unstable/lost traction when extrapolating past 0.8.
  twist_cmd.ranges.lin_vel_x = (-0.8, 1.0)
  # +/-0.08: measured capability. A hand-tuned open-loop crawl tops out
  # at ~0.045 m/s and trained policies (three reward schemes) all
  # asymptote at ~0.04 — 0.12+ m/s sidestep exceeds what 0.22 N.m
  # servos + point feet can do. Commands must stay trackable.
  twist_cmd.ranges.lin_vel_y = (-0.08, 0.08)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  twist_cmd.ranges.heading = None
  # 20% pure-rotation / 25% pure-lateral / 10% backward-only episodes
  # (uniform sampling alone gives lateral ~1% and rotation ~6% exposure;
  # lateral raised 15->25% after v8 hardware still ignored vy).
  twist_cmd.axis_focus_probs = (0.20, 0.25, 0.10)

  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum.pop("command_vel", None)

  return cfg
