"""XGO-Lite2 robot constants (open-firmware stack, 12-DoF locomotion).

The arm (planar shoulder/elbow + claw) is frozen in the model; the real
robot holds it at the calibrated fold pose via the host driver.
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF.
##

XGOLITE_XML: Path = (
  SRC_PATH / "assets" / "robots" / "xgolite" / "xmls" / "xgolite.xml"
)
assert XGOLITE_XML.exists()


def get_spec() -> mujoco.MjSpec:
  # Primitive-geometry model: no mesh assets to attach.
  return mujoco.MjSpec.from_file(str(XGOLITE_XML))


##
# Actuator config.
##

XGOLITE_XML_ACTUATOR = XmlPositionActuatorCfg(
  target_names_expr=(".*",),
)

##
# Keyframes.
##

# Stand pose derived on the URDF-generated model (build_model.py): support
# polygon [-53, +81] mm brackets the CoM (+12 mm) with 69/65 mm margins,
# hip height 144 mm, calf 0.24 rad from its extension limit. Note the URDF
# zero has the calf swept ~63 deg forward — earlier hand-authored models
# assumed straight-down zero and placed feet 50-90 mm wrong.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.15),
  joint_pos={
    "^(fl|fr|bl|br)_thigh_joint$": -0.15,
    "^(fl|fr|bl|br)_calf_joint$": -0.85,
    "^(fl|fr|bl|br)_hip_joint$": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

XGOLITE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XGOLITE_XML_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_xgolite_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(),
    spec_fn=get_spec,
    articulation=XGOLITE_ARTICULATION,
  )
