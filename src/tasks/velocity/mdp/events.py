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


def _terrain_level_severity(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor
) -> torch.Tensor:
  """Per-env hazard-DR severity fraction from the terrain curriculum (V21B).

  s = terrain_level / top_row in [0, 1]: envs on difficulty row 0 get benign
  physics, envs on the top row the full configured DR severity, linear in
  between — curriculum-coupled DR (v21b-v4 postmortem: full-severity hazards
  on ALL rows from iteration 0 crashed the warm-started flat gait before
  terrain skill could form, teaching "walking = crashing = bad").

  The authoritative per-env level source is the terrain entity's
  ``terrain_levels`` tensor (moved by ``terrain_levels_reward_gated`` at the
  top of ``_reset_idx``, i.e. BEFORE reset-mode events fire, so redraws see
  the episode's new level). ``max_terrain_level`` is the row COUNT
  (terrain_entity.py), so the top reachable row is ``max_terrain_level - 1``.
  Requires generator terrain with curriculum env origins — fails fast
  otherwise.
  """
  terrain = env.scene.terrain
  if terrain is None or getattr(terrain, "terrain_levels", None) is None:
    raise ValueError(
      "severity_by_terrain_level requires generator terrain with curriculum "
      "env origins (no terrain_levels on this scene)."
    )
  top_row = int(terrain.max_terrain_level) - 1
  if top_row <= 0:
    raise ValueError(
      "severity_by_terrain_level needs at least 2 terrain difficulty rows."
    )
  return terrain.terrain_levels[env_ids].float() / float(top_row)


@requires_model_fields("geom_solref", "geom_solimp")
def foot_contact_compliance(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  timeconst_range: tuple[float, float],
  solimp_d0_range: tuple[float, float],
  solimp_dmax_range: tuple[float, float],
  solimp_width_range: tuple[float, float],
  severity_by_terrain_level: bool = False,
  benign_timeconst_max: float = 0.05,
  benign_solimp: tuple[float, float, float] = (0.9, 0.95, 0.001),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Per-env contact-compliance DR on the foot-pad geoms (V21B).

  Singh et al. (arXiv:2504.13619, Humanoids 2024) fake soft household
  ground (foam / mattress / grass, real transfer) purely by randomizing the
  MuJoCo contact ``solref`` time constant in (0.02, 0.4) — "completely
  stiff to spring-like". This event transplants that recipe per ENV: one
  compliance draw per environment at startup, written to the foot-pad
  geoms' ``geom_solref``/``geom_solimp`` model rows (mjwarp carries both
  per-world once expanded — collision_core.py indexes them with
  ``worldid % shape[0]``).

  Why the FOOT side carries the ground's compliance: the foot pads have
  ``priority = 1`` (v19_spec) vs the terrain's 0, and MuJoCo resolves
  differing priorities by taking the HIGHER-priority geom's solref/solimp
  outright (mix = 1.0) — so writing the pads is exactly equivalent to
  writing every terrain geom, at 4 writes/env instead of ~100, and it
  composes with the same priority rule that makes the low foot friction
  win (the max-mixing fix).

  Ranges, anchored at the ~1.4 kPa foot pressure (577 g / 4 pads):
  - ``timeconst`` U(0.02, 0.4) s — Singh's verbatim band (>= 2 * the
    0.002 s physics timestep, MuJoCo's stability floor).
  - ``solimp`` d0 U(0.6, 0.9), dmax U(0.9, 0.97), width U(0.001, 0.010) m
    — brackets the (0.9, 0.95, 0.001) default from "hard tile" toward
    "carpet pile": at 1.4 kPa real rugs/foam deflect mm-scale, so the
    mushy low-impedance zone (width) is capped at 1 cm. d0 is clamped
    below dmax (MuJoCo requires d0 <= dmax).
  Midpoint/power (solimp[3:5]) stay at model defaults.

  ``severity_by_terrain_level`` (opt-in, v21b-v5, wire with ``mode="reset"``
  so the draw tracks the env's current level): the ranges above are the
  FULL-severity (s=1, top difficulty row) endpoints; per env they are
  linearly interpolated toward benign anchors with
  s = level / top_row (``_terrain_level_severity``):
  - timeconst upper endpoint: ``benign_timeconst_max`` (0.05 s, near-rigid)
    at s=0 -> ``timeconst_range[1]`` at s=1; the lower endpoint stays.
  - each solimp endpoint: the ``benign_solimp`` anchor (the MuJoCo model
    default (d0, dmax, width) = (0.9, 0.95, 0.001)) at s=0 -> the
    configured endpoint at s=1, so row-0 draws collapse to default
    contacts and top-row draws are exactly the ranges above.
  Default False keeps the original single-severity draw bit-identical.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids].to(torch.long)

  def _u(lo: float, hi: float) -> torch.Tensor:
    # One draw per env, shared across the selected geoms (compliance is a
    # property of the ground the whole robot stands on).
    return torch.empty(len(env_ids), 1, device=env.device).uniform_(lo, hi)

  if severity_by_terrain_level:
    if not timeconst_range[0] <= benign_timeconst_max <= timeconst_range[1]:
      raise ValueError(
        f"benign_timeconst_max must lie inside timeconst_range, got "
        f"{benign_timeconst_max} vs {timeconst_range}."
      )
    s = _terrain_level_severity(env, env_ids).unsqueeze(1)  # [n, 1]

    def _rand() -> torch.Tensor:
      return torch.rand(len(env_ids), 1, device=env.device)

    def _u_sev(rng: tuple[float, float], anchor: float) -> torch.Tensor:
      # Per-env range endpoints interpolated benign-anchor -> full range.
      eff_lo = anchor + s * (rng[0] - anchor)
      eff_hi = anchor + s * (rng[1] - anchor)
      return eff_lo + _rand() * (eff_hi - eff_lo)

    eff_tc_hi = benign_timeconst_max + s * (timeconst_range[1] - benign_timeconst_max)
    timeconst = timeconst_range[0] + _rand() * (eff_tc_hi - timeconst_range[0])
    dmax = _u_sev(solimp_dmax_range, benign_solimp[1])
    d0 = torch.minimum(_u_sev(solimp_d0_range, benign_solimp[0]), dmax - 1e-3)
    width = _u_sev(solimp_width_range, benign_solimp[2])
  else:
    timeconst = _u(*timeconst_range)
    dmax = _u(*solimp_dmax_range)
    d0 = torch.minimum(_u(*solimp_d0_range), dmax - 1e-3)
    width = _u(*solimp_width_range)

  env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
  shape = env_grid.shape
  model = env.sim.model
  model.geom_solref[env_grid, geom_grid, 0] = timeconst.expand(shape)
  # solref[1] (dampratio) stays at the default 1.0: Singh randomizes the
  # time constant only, and underdamped contacts bounce.
  model.geom_solimp[env_grid, geom_grid, 0] = d0.expand(shape)
  model.geom_solimp[env_grid, geom_grid, 1] = dmax.expand(shape)
  model.geom_solimp[env_grid, geom_grid, 2] = width.expand(shape)


class foot_slip_event(ManagerTermBase):
  """Transient per-foot slip events (V21B): rug-edge / tile-strip slips.

  Miki et al. (arXiv:2201.08117, Science Robotics 2022) inject transient
  slip by "occasionally setting the feet's friction low"; the lit review
  ranks it (with the friction low tail) as the evidenced slippery-surface
  recipe. Each environment, at random intervals drawn from
  ``interval_range_s``, has ONE random foot's sliding friction dropped to
  an absolute U(``mu_range``) for U(``duration_range_s``) seconds, then
  restored to the exact pre-slip (startup-DR'd) value.

  The drop is effective against every terrain geom because the foot pads
  carry ``priority = 1``: MuJoCo's friction combination takes the
  higher-priority geom's friction outright instead of the element-wise MAX
  (the "ice tile is a silent no-op" gotcha), so a 0.02 foot beats the
  1.0-friction terrain.

  Implementation note (why ``mode="step"`` and not ``mode="interval"``):
  mjlab interval events fire a callback when a per-env timer expires but
  provide no delayed second callback, and the RESTORE must happen a
  precise 0.2-0.5 s after the drop. The term therefore runs as a step-mode
  state machine that reproduces interval semantics internally: per-env
  start timer (resampled from ``interval_range_s`` after each slip and on
  reset) plus per-env active-slip countdown. Envs that reset mid-slip get
  their saved friction restored in ``reset()`` when ``restore_on_reset``
  is True (the right semantics when the base friction DR is startup-only
  and never re-runs on episode resets).

  ``severity_by_terrain_level`` (opt-in, v21b-v5): the drawn slip friction
  is interpolated per env between the foot's CURRENT (pre-slip) value and
  the drawn ``mu_range`` low, mu_eff = saved + s * (drawn - saved) with
  s = level / top_row (``_terrain_level_severity``). Row 0 slips are exact
  no-ops (benign physics for beginners), top-row slips are the full Miki
  drop — the pre-severity behavior bit-for-bit.

  ``restore_on_reset=False`` (v21b-v5, REQUIRED when the base friction is
  re-randomized by a reset-mode event): ``_reset_idx`` applies reset-mode
  events BEFORE calling ``event_manager.reset`` (manager_based_rl_env.py),
  so restoring the stale saved value here would overwrite the fresh
  friction draw on the slipped pad. With the flag off, reset only
  deactivates the slip and resamples the timer; the reset-mode friction
  event owns the value.
  """

  # Read by EventManager._prepare_terms: geom_friction must be a per-world
  # tensor even if no other friction DR term is active.
  model_fields = ("geom_friction",)

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    # Remaining params arrive via __call__ kwargs; reset() has no kwargs,
    # so the reset-behavior switch is consumed here.
    self._restore_on_reset = bool(cfg.params.get("restore_on_reset", True))
    self._initialized = False
    self._geom_ids: torch.Tensor | None = None
    self._interval_range: tuple[float, float] = (0.0, 0.0)
    n = env.num_envs
    dev = env.device
    self._active = torch.zeros(n, dtype=torch.bool, device=dev)
    self._slip_foot = torch.zeros(n, dtype=torch.long, device=dev)
    self._slip_left = torch.zeros(n, device=dev)
    self._time_to_next = torch.zeros(n, device=dev)
    self._saved_mu = torch.zeros(n, device=dev)

  def _sample_interval(self, n: int) -> torch.Tensor:
    lo, hi = self._interval_range
    return torch.empty(n, device=self.device).uniform_(lo, hi)

  def _restore(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    gids = self._geom_ids[self._slip_foot[env_ids]]  # type: ignore[index]
    self._env.sim.model.geom_friction[env_ids, gids, 0] = self._saved_mu[env_ids]
    self._active[env_ids] = False
    self._time_to_next[env_ids] = self._sample_interval(len(env_ids))

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if not self._initialized:
      return
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    elif isinstance(env_ids, slice):
      env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
    if self._restore_on_reset:
      self._restore(env_ids[self._active[env_ids]])
    else:
      # A reset-mode friction event already redrew the base friction for
      # these envs (it fires before this reset hook); just drop the slip.
      self._active[env_ids] = False
    self._time_to_next[env_ids] = self._sample_interval(len(env_ids))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    interval_range_s: tuple[float, float],
    duration_range_s: tuple[float, float],
    mu_range: tuple[float, float],
    severity_by_terrain_level: bool = False,
    restore_on_reset: bool = True,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    del env_ids  # mode="step" fires unconditionally on all envs.
    del restore_on_reset  # Consumed in __init__ (reset() has no kwargs).
    if not self._initialized:
      asset: Entity = env.scene[asset_cfg.name]
      self._geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids].to(torch.long)
      self._interval_range = tuple(interval_range_s)  # type: ignore[assignment]
      self._time_to_next = self._sample_interval(env.num_envs)
      self._initialized = True
    assert self._geom_ids is not None
    dt = env.step_dt
    friction = env.sim.model.geom_friction

    # 1. Expire running slips and restore the saved friction.
    self._slip_left = torch.where(
      self._active, self._slip_left - dt, self._slip_left
    )
    expired = self._active & (self._slip_left <= 0.0)
    self._restore(expired.nonzero().flatten())

    # 2. Tick the start timers of inactive envs; fire due slips.
    inactive = ~self._active
    self._time_to_next = torch.where(
      inactive, self._time_to_next - dt, self._time_to_next
    )
    start_ids = (inactive & (self._time_to_next <= 0.0)).nonzero().flatten()
    if len(start_ids) > 0:
      n = len(start_ids)
      foot = torch.randint(0, len(self._geom_ids), (n,), device=self.device)
      gids = self._geom_ids[foot]
      self._saved_mu[start_ids] = friction[start_ids, gids, 0]
      slip_mu = torch.empty(n, device=self.device).uniform_(*mu_range)
      if severity_by_terrain_level:
        # s=0: mu_eff == the saved value, an exact no-op slip; s=1: the
        # full drawn drop (pre-severity behavior, bit-for-bit).
        s = _terrain_level_severity(env, start_ids)
        saved = self._saved_mu[start_ids]
        slip_mu = saved + s * (slip_mu - saved)
      friction[start_ids, gids, 0] = slip_mu
      self._slip_foot[start_ids] = foot
      self._slip_left[start_ids] = torch.empty(
        n, device=self.device
      ).uniform_(*duration_range_s)
      self._active[start_ids] = True


@requires_model_fields("geom_friction")
def foot_friction_terrain_scaled(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  friction_range: tuple[float, float],
  benign_low: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Per-env foot friction DR with a terrain-level-scaled low tail (V21B).

  Replaces the startup ``dr.geom_friction`` draw for v21b (other tasks keep
  the upstream event untouched): the (0.05, 2.0) ice tail at FULL severity
  from iteration 0 on ALL terrain rows crashed the warm-started flat gait
  before terrain skill could form (v21b-v4 postmortem). Wire with
  ``mode="reset"`` so the draw tracks the env's CURRENT terrain level (the
  curriculum moves levels at the top of ``_reset_idx``, before reset-mode
  events fire). The LOW endpoint interpolates with severity,

    mu ~ U(benign_low + s * (lo - benign_low), hi),  s = level / top_row,

  so row 0 draws mu >= ``benign_low`` (no ice under beginners) and the top
  row draws the full configured range. The high end stays fixed — high
  friction is not a hazard. One tangential draw per env, shared across the
  selected pads (the ``shared_random=True`` contract of the event this
  replaces); torsional/rolling components stay at model defaults. Requires
  generator terrain with the level curriculum (fails fast otherwise).
  Composes with ``foot_slip_event``: set its ``restore_on_reset=False``
  (see that docstring for the ``_reset_idx`` ordering).
  """
  lo, hi = friction_range
  if not lo <= benign_low <= hi:
    raise ValueError(
      f"benign_low must lie inside friction_range, got {benign_low} vs "
      f"{friction_range}."
    )
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids].to(torch.long)
  s = _terrain_level_severity(env, env_ids)
  eff_lo = benign_low + s * (lo - benign_low)
  mu = eff_lo + torch.rand(len(env_ids), device=env.device) * (hi - eff_lo)
  env.sim.model.geom_friction[env_ids[:, None], geom_ids, 0] = mu.unsqueeze(1)


def piecewise_tau_line(
  qd_abs: torch.Tensor,
  tau_max: torch.Tensor | float,
  qd_knee: torch.Tensor | float,
  qd_max: torch.Tensor | float,
) -> torch.Tensor:
  """Driving-torque ceiling of the piecewise servo torque-speed curve.

  Flat at ``tau_max`` for ``|qd| <= qd_knee``, then linear taper to zero at
  ``qd_max``. With ``qd_knee == 0`` this reduces exactly to the pre-v19
  single line ``tau_max * clip(1 - |qd|/qd_max, 0, 1)``.
  """
  return tau_max * ((qd_max - qd_abs) / (qd_max - qd_knee)).clamp(0.0, 1.0)


class TorqueSpeedClamp(ManagerTermBase):
  """Per-step one-sided torque-speed clamp for hobby serial-bus servos.

  Sim autopsy (2026-07-11/12): the plain forcerange model delivers the full
  stall torque at ANY joint speed, so aggressive policies exploit "superhero
  torque" far above the real DC-motor line (measured up to 84% above it) and
  stall on hardware. The 2026-07-14 Stage-1 servo fit measured the real
  curve as PIECEWISE: full torque up to a knee speed, then a linear taper to
  zero (see ``piecewise_tau_line``):

      tau_line(qd) = tau_max * clip((qd_max - |qd|) / (qd_max - qd_knee), 0, 1)

  The clamp is ONE-SIDED: it only limits DRIVING torque (same sign as the
  joint velocity); BRAKING torque (opposite sign, i.e. the motor plugging /
  back-EMF regime) keeps the full +/-tau_max authority. MuJoCo's per-
  actuator ``forcerange`` is an independent [min, max] pair, so asymmetric
  per-step limits implement this exactly:

      qd > 0:  max = +tau_line(|qd|),  min = -tau_max
      qd < 0:  min = -tau_line(|qd|),  max = +tau_max
      qd = 0:  +/-tau_max  (tau_line(0) == tau_max, continuous)

  Composition with strength DR: on the first fire (which happens strictly
  after all startup events) the term reads back the DR'd forcerange written
  by ``actuator_gains_and_strength`` and recovers each servo's per-env x
  per-servo strength factor s = forcerange_max / default_forcerange_max.
  Both axes scale with s — a DC motor's stall torque AND no-load speed are
  proportional to supply voltage, so the strength (voltage-proxy) factor
  multiplies tau_max, qd_knee and qd_max together. The captured factors are
  cached: strength DR is a startup-only event, so they are constant for the
  whole run. (Any other event that rewrites forcerange after startup would
  be silently overwritten by this term — keep forcerange DR in startup mode
  when the clamp is enabled.)

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
    self._tau_max: torch.Tensor | None = None
    self._qd_knee: torch.Tensor | None = None
    self._qd_max: torch.Tensor | None = None

  def _lazy_init(
    self,
    env: ManagerBasedRlEnv,
    tau_max: float,
    qd_knee: float,
    qd_max: float,
    asset_cfg: SceneEntityCfg,
  ) -> None:
    if not 0.0 <= qd_knee < qd_max:
      raise ValueError(
        f"TorqueSpeedClamp requires 0 <= qd_knee < qd_max, got "
        f"qd_knee={qd_knee}, qd_max={qd_max}"
      )
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
    self._tau_max = (strength * tau_max).clone()
    self._qd_knee = (strength * qd_knee).clone()
    self._qd_max = (strength * qd_max).clone()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    tau_max: float,
    qd_knee: float,
    qd_max: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    del env_ids  # mode="step" fires unconditionally on all envs.
    if not self._initialized:
      self._lazy_init(env, tau_max, qd_knee, qd_max, asset_cfg)
      self._initialized = True
    assert self._tau_max is not None
    assert self._qd_knee is not None and self._qd_max is not None
    assert self._joint_ids is not None and self._ctrl_ids is not None
    assert self._all_env_ids is not None

    asset: Entity = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel[:, self._joint_ids]
    tau_line = piecewise_tau_line(
      qd.abs(), self._tau_max, self._qd_knee, self._qd_max
    )
    driving_pos = qd >= 0
    upper = torch.where(driving_pos, tau_line, self._tau_max)
    lower = torch.where(driving_pos, -self._tau_max, -tau_line)

    fr = env.sim.model.actuator_forcerange
    fr[self._all_env_ids[:, None], self._ctrl_ids, 0] = lower
    fr[self._all_env_ids[:, None], self._ctrl_ids, 1] = upper
