"""Sim-fidelity fixes for the XGO-Lite2 servo model (2026-07-12).

Motivated by the 2026-07-11 aggressive-preset hardware failures and the sim
autopsy (parent repo docs): in sim the servos deliver the full 0.22 N*m at
ANY joint speed, so aggressive policies exploit torque the real DC motors
cannot produce (up to 84% above the torque-speed line), and the actuator
response delay is drawn once per env, so the differential lag between
loaded/unloaded legs — what actually desynchronizes gaits on hardware —
is never seen in training.

Two OPT-IN fixes, each a pure add-on (existing tasks stay bit-identical):

1. ``enable_torque_speed_clamp``: per-control-step one-sided forcerange
   update implementing tau_max(qd) = tau_stall * clip(1 - |qd|/omega_nl,
   0, 1) on driving torque only (braking keeps full authority). Composes
   with the v17 per-env x per-servo strength DR by reading back the DR'd
   forcerange at first fire (see ``mdp.events.TorqueSpeedClamp``).
   Defaults tau_stall = 0.22 N*m (XML forcerange), omega_nl = 4.5 rad/s.

2. ``enable_per_servo_delay``: swaps the robot's ``DelayedActuatorCfg``
   for the local ``PerServoDelayedActuatorCfg`` — same 60-100 ms range,
   hold probability and staggered 0.5 s refresh, but each servo draws its
   lag independently (see ``mdp.actuators``).

``xgolite_v18draft_env_cfg`` = v17 (``xgolite_flat_env_cfg``) + both fixes,
registered as task id ``XGOLite-V18Draft``. To adopt the fixes in any other
preset, call the two helpers on its cfg after construction, e.g.::

    cfg = xgolite_sprint_env_cfg()
    enable_torque_speed_clamp(cfg)
    enable_per_servo_delay(cfg)
"""

import dataclasses

from mjlab.actuator import DelayedActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity import mdp as local_mdp

from .env_cfgs import xgolite_flat_env_cfg
from .precision import xgolite_precision_env_cfg

# Defaults for the XGO-Lite2 Feetech-class servos: stall torque from the
# XML forcerange (hardware-truth value used since v1), no-load speed from
# the servo spec sheet / bench line fit used in the sim autopsy.
TAU_STALL = 0.22  # N*m
OMEGA_NL = 4.5  # rad/s


def enable_torque_speed_clamp(
  cfg: ManagerBasedRlEnvCfg,
  tau_stall: float = TAU_STALL,
  omega_nl: float = OMEGA_NL,
) -> None:
  """Add the per-step one-sided torque-speed clamp to ``cfg.events``."""
  cfg.events["torque_speed_clamp"] = EventTermCfg(
    func=local_mdp.TorqueSpeedClamp,
    mode="step",
    params={
      "tau_stall": tau_stall,
      "omega_nl": omega_nl,
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


def xgolite_v18draft_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v17 baseline + torque-speed clamp + per-servo delay DR."""
  cfg = xgolite_flat_env_cfg(play=play)
  enable_torque_speed_clamp(cfg)
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
  enable_torque_speed_clamp(cfg)
  enable_per_servo_delay(cfg)
  return cfg
