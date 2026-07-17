"""XGOMini robot constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

XGOMINI_XML: Path = (
  SRC_PATH / "assets" / "robots" / "xgomini" / "xmls" / "xgomini.xml"
)
assert XGOMINI_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  """Load mesh assets for XGOMINI from the mesh directory referenced by meshdir."""
  # mjlab.utils.os.update_assets was removed in mjlab 1.5; keys keep the
  # meshdir prefix so MjSpec resolves them like file references.
  assets: dict[str, bytes] = {}
  meshdir_clean = meshdir.rstrip("/")
  mesh_path = (XGOMINI_XML.parent / meshdir_clean).resolve()
  for f in mesh_path.glob("*"):
    if f.is_file():
      assets[f"{meshdir_clean}/{f.name}"] = f.read_bytes()
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(XGOMINI_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

XGOMINI_XML_ACTUATOR = XmlActuatorCfg(
  target_names_expr=(".*",),
  command_field="position",
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.25),
  joint_pos={
    "^(fl|fr|bl|br)_thigh_joint$": 1,
    "^(fl|fr|bl|br)_calf_joint$": -16.0/57.3,
    "^(fl|fr|bl|br)_hip_joint$": 0.0,
    "arm_yaw_joint": 0.0,
    "arm_thigh_joint": -1.57,
    "arm_calf_joint": 1.57,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

XGOMINI_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XGOMINI_XML_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_XGOMINI_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(),
    spec_fn=get_spec,
    articulation=XGOMINI_ARTICULATION,
  )


def get_xgomini_robot_cfg() -> EntityCfg:
  return get_XGOMINI_robot_cfg()


def get_xgomini_walk_robot_cfg() -> EntityCfg:
  return get_XGOMINI_robot_cfg()
