"""Sim-fidelity fixes for the XGO-Lite2 servo model (2026-07-12).

Motivated by the 2026-07-11 aggressive-preset hardware failures and the sim
autopsy (parent repo docs): in sim the servos deliver the full 0.22 N*m at
ANY joint speed, so aggressive policies exploit torque the real DC motors
cannot produce (up to 84% above the torque-speed line), and the actuator
response delay is drawn once per env, so the differential lag between
loaded/unloaded legs — what actually desynchronizes gaits on hardware —
is never seen in training.

Three OPT-IN fixes, each a pure add-on (existing tasks stay bit-identical):

1. ``enable_torque_speed_clamp``: per-control-step one-sided forcerange
   update implementing the piecewise servo torque-speed curve — driving
   torque flat at tau_max for |qd| <= qd_knee, linear taper to zero at
   qd_max; braking keeps full authority. Composes with the v17 per-env x
   per-servo strength DR by reading back the DR'd forcerange at first fire
   (see ``mdp.events.TorqueSpeedClamp``). Defaults reproduce the v18
   single-line envelope exactly: tau_max = 0.22 N*m (XML forcerange),
   qd_knee = 0 (droop starts at qd 0), qd_max = 4.5 rad/s (bench line fit).

2. ``enable_per_servo_delay``: swaps the robot's ``DelayedActuatorCfg``
   for the local ``PerServoDelayedActuatorCfg`` — same 60-100 ms range,
   hold probability and staggered 0.5 s refresh, but each servo draws its
   lag independently (see ``mdp.actuators``).

3. ``enable_servo_deadband`` (2026-07-13 servo-ID finding): sets
   ``deadband_range`` on the per-servo delayed actuator so each (env,
   servo) draws a lost-motion half-width at episode reset and the
   post-delay position target is shrunk toward the measured position by
   it (see ``mdp.actuators.PerServoDelayedActuator``). Default range
   0.015-0.035 rad brackets the measured ~0.02 rad / published 0.023 rad
   backlash; the v19 preset passes (0.0, 0.025) explicitly (see v19.py).
   Requires ``enable_per_servo_delay`` first (deadband is a per-servo
   property and lives on the per-servo actuator cfg).

``xgolite_v18draft_env_cfg`` = v17 (``xgolite_flat_env_cfg``) + fixes 1+2,
registered as task id ``XGOLite-V18Draft``. To adopt the fixes in any other
preset, call the helpers on its cfg after construction, e.g.::

    cfg = xgolite_sprint_env_cfg()
    enable_torque_speed_clamp(cfg)
    enable_per_servo_delay(cfg)
    enable_servo_deadband(cfg)  # v19: randomized lost motion
"""

import dataclasses

from mjlab.actuator import DelayedActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity import mdp as local_mdp

from .env_cfgs import xgolite_flat_env_cfg
from .precision import xgolite_precision_env_cfg

# v18 clamp envelope for the XGO-Lite2 Feetech-class servos: stall torque
# from the XML forcerange (hardware-truth value used since v1), no-load
# speed from the servo spec sheet / bench line fit used in the sim autopsy.
# qd_knee 0 = the pre-v19 single line (droop starts at qd 0); the v19
# preset passes the measured piecewise curve instead (measured_actuators).
V18_TAU_MAX = 0.22  # N*m
V18_QD_KNEE = 0.0  # rad/s
V18_QD_MAX = 4.5  # rad/s

# Lost-motion draw range [rad]: hardware sine sweeps (2026-07-13) fit
# ~0.02 rad, published backlash for this servo class ~1.3 deg = 0.023 rad.
DEADBAND_RANGE = (0.015, 0.035)


def enable_torque_speed_clamp(
  cfg: ManagerBasedRlEnvCfg,
  tau_max: float = V18_TAU_MAX,
  qd_knee: float = V18_QD_KNEE,
  qd_max: float = V18_QD_MAX,
) -> None:
  """Add the per-step one-sided piecewise torque-speed clamp to ``cfg.events``."""
  cfg.events["torque_speed_clamp"] = EventTermCfg(
    func=local_mdp.TorqueSpeedClamp,
    mode="step",
    params={
      "tau_max": tau_max,
      "qd_knee": qd_knee,
      "qd_max": qd_max,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )


def enable_per_servo_delay(cfg: ManagerBasedRlEnvCfg) -> None:
  """Swap the robot's delayed actuator(s) for the per-servo variant.

  Keeps every delay parameter (range, hold prob, update period) unchanged;
  only the lag draw granularity changes from per-env to per-(env, servo).
  Uses ``dataclasses.replace`` so the shared module-level robot cfg objects
  (``XGOLITE_ARTICULATION`` et al.) are never mutated.
  """
  robot = cfg.scene.entities["robot"]
  assert robot.articulation is not None
  new_actuators = []
  for act_cfg in robot.articulation.actuators:
    if isinstance(act_cfg, DelayedActuatorCfg) and not isinstance(
      act_cfg, local_mdp.PerServoDelayedActuatorCfg
    ):
      act_cfg = local_mdp.PerServoDelayedActuatorCfg(
        base_cfg=act_cfg.base_cfg,
        delay_target=act_cfg.delay_target,
        delay_min_lag=act_cfg.delay_min_lag,
        delay_max_lag=act_cfg.delay_max_lag,
        delay_hold_prob=act_cfg.delay_hold_prob,
        delay_update_period=act_cfg.delay_update_period,
        delay_per_env_phase=act_cfg.delay_per_env_phase,
      )
    new_actuators.append(act_cfg)
  articulation = dataclasses.replace(
    robot.articulation, actuators=tuple(new_actuators)
  )
  cfg.scene.entities["robot"] = dataclasses.replace(
    robot, articulation=articulation
  )


def enable_servo_deadband(
  cfg: ManagerBasedRlEnvCfg,
  deadband_range: tuple[float, float] = DEADBAND_RANGE,
) -> None:
  """Enable per-(env, servo) lost-motion deadband on the robot's servos.

  Deadband lives on ``PerServoDelayedActuatorCfg`` as a first-class field,
  so the robot must already use the per-servo delayed actuator — call
  ``enable_per_servo_delay(cfg)`` first. Follows the same
  ``dataclasses.replace`` copy-on-write pattern so shared module-level
  robot cfg objects are never mutated.
  """
  robot = cfg.scene.entities["robot"]
  assert robot.articulation is not None
  new_actuators = []
  found = False
  for act_cfg in robot.articulation.actuators:
    if isinstance(act_cfg, local_mdp.PerServoDelayedActuatorCfg):
      act_cfg = dataclasses.replace(act_cfg, deadband_range=deadband_range)
      found = True
    elif isinstance(act_cfg, DelayedActuatorCfg):
      raise TypeError(
        "enable_servo_deadband requires the per-servo delayed actuator; "
        "call enable_per_servo_delay(cfg) first."
      )
    new_actuators.append(act_cfg)
  if not found:
    raise TypeError(
      "enable_servo_deadband found no PerServoDelayedActuatorCfg on the robot."
    )
  articulation = dataclasses.replace(
    robot.articulation, actuators=tuple(new_actuators)
  )
  cfg.scene.entities["robot"] = dataclasses.replace(
    robot, articulation=articulation
  )


def xgolite_v18draft_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v17 baseline + torque-speed clamp + per-servo delay DR."""
  cfg = xgolite_flat_env_cfg(play=play)
  enable_torque_speed_clamp(
    cfg, tau_max=V18_TAU_MAX, qd_knee=V18_QD_KNEE, qd_max=V18_QD_MAX
  )
  enable_per_servo_delay(cfg)
  return cfg


def xgolite_precision2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Precision preset + both sim-fidelity fixes; NO other changes.

  Intended to be trained by FINE-TUNING from a v17 checkpoint: from-scratch
  training of the tight-tolerance Precision reward collapses to walking in
  place (no reward gradient — see range_curriculum.py for the from-scratch
  fix, XGOLite-V18Range).
  """
  cfg = xgolite_precision_env_cfg(play=play)
  enable_torque_speed_clamp(
    cfg, tau_max=V18_TAU_MAX, qd_knee=V18_QD_KNEE, qd_max=V18_QD_MAX
  )
  enable_per_servo_delay(cfg)
  return cfg
