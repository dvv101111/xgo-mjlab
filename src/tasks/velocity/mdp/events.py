"""Domain-randomization events specific to the XGO velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.actuator import BuiltinPositionActuator, XmlPositionActuator
from mjlab.actuator.delayed_actuator import DelayedActuator
from mjlab.entity import Entity
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _resolve_actuators(asset: Entity, asset_cfg: SceneEntityCfg) -> list:
  """Resolve the actuator list selected by ``asset_cfg`` (shared DR pattern)."""
  if isinstance(asset_cfg.actuator_ids, list):
    return [asset.actuators[i] for i in asset_cfg.actuator_ids]
  if isinstance(asset_cfg.actuator_ids, slice):
    return asset.actuators[asset_cfg.actuator_ids]
  return [asset.actuators[asset_cfg.actuator_ids]]


@requires_model_fields(
  "actuator_gainprm", "actuator_biasprm", "actuator_forcerange"
)
def actuator_gains_and_strength(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_scale_range: tuple[float, float],
  kv_scale_range: tuple[float, float],
  strength_scale_range: tuple[float, float],
  strength_servo_range: tuple[float, float] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomize per-servo PD gains plus a correlated per-env strength factor.

  v16: the real robot has 12 individually-worn Feetech servos driven from a
  shared 6.6-8.4 V battery. kp/kv scales are sampled per servo (left-right
  strength asymmetry); the strength factor is sampled once per env and
  multiplies kp, kv and the torque ceiling together (tau' = f * tau), acting
  as a voltage proxy. Ranges follow walk-these-ways motor-strength DR.

  v17: optional ``strength_servo_range`` adds an INDEPENDENT per-servo
  strength factor on top of the correlated per-env one, so the torque
  ceiling (forcerange) — not just the PD gains — differs servo to servo.
  Motivation (2026-07-11 ladders): per-dwell parasitic yaw under straight
  commands spans -0.10..+0.09 rad/s and flips sign with operating point and
  battery; at vx 0.14 m/s a 0.09 rad/s curvature over the 0.166 m hip track
  needs only ~5% left/right thrust mismatch, and in the velocity-saturation-
  dominated regime the per-servo CEILING — which v16 kept common across the
  env — is what sets delivered thrust. The policy must null signed plant
  asymmetry via gyro feedback; mirror augmentation alone cannot expose it.

  Unlike mjlab's ``dr.pd_gains`` the two randomizations compose in one event
  (pd_gains rescales from defaults, so two calls would not stack).
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  if isinstance(asset_cfg.actuator_ids, list):
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
  elif isinstance(asset_cfg.actuator_ids, slice):
    actuators = asset.actuators[asset_cfg.actuator_ids]
  else:
    actuators = [asset.actuators[asset_cfg.actuator_ids]]
  actuators = [
    a.base_actuator if isinstance(a, DelayedActuator) else a for a in actuators
  ]

  def _uniform(lo: float, hi: float, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.empty(shape, device=env.device).uniform_(lo, hi)

  strength_env = _uniform(*strength_scale_range, (len(env_ids), 1))

  default_gainprm = env.sim.get_default_field("actuator_gainprm")
  default_biasprm = env.sim.get_default_field("actuator_biasprm")
  default_forcerange = env.sim.get_default_field("actuator_forcerange")

  for actuator in actuators:
    if not isinstance(actuator, (BuiltinPositionActuator, XmlPositionActuator)):
      raise TypeError(
        "actuator_gains_and_strength only supports position actuators "
        f"(optionally delayed), got {type(actuator).__name__}"
      )
    ctrl_ids = actuator.global_ctrl_ids
    shape = (len(env_ids), len(ctrl_ids))
    # Per-servo strength composes with the per-env (voltage-proxy) factor;
    # (1, 1) when disabled keeps the v16 behavior bit-for-bit.
    if strength_servo_range is not None:
      strength = strength_env * _uniform(*strength_servo_range, shape)
    else:
      strength = strength_env.expand(shape).clone()
    kp_scale = _uniform(*kp_scale_range, shape) * strength
    kv_scale = _uniform(*kv_scale_range, shape) * strength

    env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
      default_gainprm[ctrl_ids, 0] * kp_scale
    )
    env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = (
      default_biasprm[ctrl_ids, 1] * kp_scale
    )
    env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = (
      default_biasprm[ctrl_ids, 2] * kv_scale
    )
    env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, :] = (
      default_forcerange[ctrl_ids, :] * strength.unsqueeze(-1)
    )


class TorqueSpeedClamp(ManagerTermBase):
  """Per-step one-sided torque-speed clamp for hobby serial-bus servos.

  Sim autopsy (2026-07-11/12): the plain forcerange model delivers the full
  stall torque at ANY joint speed, so aggressive policies exploit "superhero
  torque" far above the real DC-motor line (measured up to 84% above it) and
  stall on hardware. Real cheap servos droop linearly to zero torque at the
  no-load speed:

      tau_max(qd) = tau_stall * clip(1 - |qd| / omega_nl, 0, 1)

  The clamp is ONE-SIDED: it only limits DRIVING torque (same sign as the
  joint velocity); BRAKING torque (opposite sign, i.e. the motor plugging /
  back-EMF regime) keeps the full +/-tau_stall authority. MuJoCo's per-
  actuator ``forcerange`` is an independent [min, max] pair, so asymmetric
  per-step limits implement this exactly:

      qd > 0:  max = +tau_line(|qd|),  min = -tau_stall
      qd < 0:  min = -tau_line(|qd|),  max = +tau_stall
      qd = 0:  +/-tau_stall  (tau_line(0) == tau_stall, continuous)

  Composition with strength DR: on the first fire (which happens strictly
  after all startup events) the term reads back the DR'd forcerange written
  by ``actuator_gains_and_strength`` and recovers each servo's per-env x
  per-servo strength factor s = forcerange_max / default_forcerange_max.
  Both tau_stall AND omega_nl are scaled by s — a weak (low-voltage / worn)
  servo has proportionally lower stall torque and lower no-load speed. The
  captured factors are cached: strength DR is a startup-only event, so they
  are constant for the whole run. (Any other event that rewrites forcerange
  after startup would be silently overwritten by this term — keep forcerange
  DR in startup mode when the clamp is enabled.)

  Wire with ``mode="step"`` so the bounds refresh every control step (50 Hz)
  from the current joint velocities; the bounds then hold for the next
  control step's physics substeps. Updating at control rate instead of
  physics rate is an accepted approximation (one 20 ms step of lag).

  Opt-in: existing tasks are unchanged unless the event is added to
  ``cfg.events`` (see ``enable_torque_speed_clamp`` in
  ``config/xgolite/sim_fidelity.py``).
  """

  # Read by EventManager._prepare_terms so actuator_forcerange gets expanded
  # to a per-world tensor even when no other forcerange DR term is active.
  model_fields = ("actuator_forcerange",)

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    del cfg  # Params arrive via __call__ kwargs.
    self._initialized = False
    self._joint_ids: torch.Tensor | None = None
    self._ctrl_ids: torch.Tensor | None = None
    self._all_env_ids: torch.Tensor | None = None
    self._tau_stall: torch.Tensor | None = None
    self._omega_nl: torch.Tensor | None = None

  def _lazy_init(
    self,
    env: ManagerBasedRlEnv,
    tau_stall: float,
    omega_nl: float,
    asset_cfg: SceneEntityCfg,
  ) -> None:
    asset: Entity = env.scene[asset_cfg.name]
    joint_ids: list[torch.Tensor] = []
    ctrl_ids: list[torch.Tensor] = []
    for actuator in _resolve_actuators(asset, asset_cfg):
      # DelayedActuator proxies target_ids/global_ctrl_ids of its base, and
      # both index the same joints, so no unwrapping is needed.
      joint_ids.append(actuator.target_ids)
      ctrl_ids.append(actuator.global_ctrl_ids)
    self._joint_ids = torch.cat(joint_ids)
    self._ctrl_ids = torch.cat(ctrl_ids)
    self._all_env_ids = torch.arange(
      env.num_envs, device=env.device, dtype=torch.long
    )

    default_fr = env.sim.get_default_field("actuator_forcerange")
    default_upper = default_fr[self._ctrl_ids, 1]
    default_lower = default_fr[self._ctrl_ids, 0]
    if not torch.allclose(default_lower, -default_upper):
      raise ValueError(
        "TorqueSpeedClamp expects a symmetric default forcerange "
        "(the strength factor is recovered from the upper bound only)."
      )
    if (default_upper <= 0).any():
      raise ValueError("TorqueSpeedClamp requires a positive default forcerange.")

    # Recover the per-env x per-servo strength factor written by the startup
    # strength DR (identity when no forcerange DR is configured).
    current_upper = env.sim.model.actuator_forcerange[
      self._all_env_ids[:, None], self._ctrl_ids, 1
    ]
    strength = current_upper / default_upper
    self._tau_stall = (strength * tau_stall).clone()
    self._omega_nl = (strength * omega_nl).clamp_min(1e-6).clone()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    tau_stall: float = 0.22,
    omega_nl: float = 4.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    del env_ids  # mode="step" fires unconditionally on all envs.
    if not self._initialized:
      self._lazy_init(env, tau_stall, omega_nl, asset_cfg)
      self._initialized = True
    assert self._tau_stall is not None and self._omega_nl is not None
    assert self._joint_ids is not None and self._ctrl_ids is not None
    assert self._all_env_ids is not None

    asset: Entity = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel[:, self._joint_ids]
    tau_line = self._tau_stall * (1.0 - qd.abs() / self._omega_nl).clamp(0.0, 1.0)
    driving_pos = qd >= 0
    upper = torch.where(driving_pos, tau_line, self._tau_stall)
    lower = torch.where(driving_pos, -self._tau_stall, -tau_line)

    fr = env.sim.model.actuator_forcerange
    fr[self._all_env_ids[:, None], self._ctrl_ids, 0] = lower
    fr[self._all_env_ids[:, None], self._ctrl_ids, 1] = upper
