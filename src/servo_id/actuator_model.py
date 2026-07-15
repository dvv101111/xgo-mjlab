"""ToddlerBot-style servo model + fit-parameter spec.

Torque law (per joint, per physics step):

    err     = q_des - q
    err_eff = sign(err) * max(|err| - deadband, 0)      lost motion / backlash
    tau_pd  = kp * err_eff - kd * qd
    line    = tau_max                                   for |qd| <= qd_knee
              tau_max * (qd_max - |qd|)/(qd_max - qd_knee)  taper to 0
              0                                         for |qd| >= qd_max
    driving side (sign(tau) == sign(qd)) clamped to the line;
    braking side clamped to +/- tau_max (full authority, back-EMF regime).

``deadband`` models the servo's lost motion (gear backlash + pot
deadzone, published ~1.3 deg = 0.023 rad for this class): inside the
band the position error produces no torque, outside it the response is
shifted-linear. It is what reproduces the measured small-amplitude sine
attenuation (~0.85x at 0.08 rad) that frictionloss alone cannot.

Coulomb friction, viscous damping and rotor inertia are NOT part of the
torque law: they are the native MJCF ``frictionloss`` / ``damping`` /
``armature`` joint fields, set on the model per candidate.

Identifiability (spec 2026-07-12): any common scaling of {armature,
damping, kp, tau_max} is trajectory-invariant on UNLOADED data, so
``tau_max`` is PINNED by default (0.22 N*m, the hardware-truth XML
forcerange). 2026-07-15: tau_max is now in PARAM_BOUNDS so it CAN be
freed, but only when LOADED (stand, on-floor) sessions are in the fit
input — the robot's own weight is the known external torque that breaks
the scaling degeneracy (foot payload cuffs are physically impossible on
this platform). The fit CLI enforces this (scripts/fit_servo_model.py).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

TAU_MAX_DEFAULT = 0.22  # N*m, pinned torque-scale anchor (XML forcerange)
QD_MAX_MARGIN = 0.5  # rad/s, enforced qd_max > qd_knee separation

# Fit-parameter catalog: name -> (lower, upper) physical bounds. CMA-ES
# works in [-1, 1] per coordinate; these map linearly. Priors: XML defaults
# kp 5.0 / kv 0.12 / damping 0.05 / armature 0.002 / frictionloss 0.001;
# ToddlerBot XC330 (similar class): damping 0.134, armature 0.0035,
# frictionloss 0.014, knee 3.29 rad/s. Measured on the 2026-07-13 stand
# sessions: step-response peak |qd_meas| 7.7-9.7 rad/s per joint UNDER
# STAND LOAD (the legacy omega_nl = 4.5 rad/s is ~2x too pessimistic) ->
# qd_knee in [3, 10], qd_max (taper end / no-load speed) in [8, 14].
# Delay: measured onset latency median 54 ms, range 33-77; fit v1 pinned
# the per-joint delay at the 20 ms floor -> lowered to [5, 90] ms. kp/kd
# upper bounds raised after fit v1 pinned kp at 20 on most joints.
# Deadband: published backlash ~1.3 deg = 0.023 rad -> [0, 0.06] rad.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
  "kp": (0.5, 40.0),
  "kd": (0.0, 2.0),
  "damping": (0.0, 0.4),
  # armature lower bound 0 -> 5e-4 (2026-07-15): reflected rotor inertia is
  # physically nonzero (literature scale ~1e-3 for this servo class) and the
  # 2026-07-15 decay session showed ZERO backdrive, so armature cannot fit
  # to a measured floor; the old fitted 0 (v3: 3.9e-7) produced a 500 Hz
  # dither plant in training. Vendor datasheet (2026-07-15) confirms an
  # IRON-CORE motor: several times the rotor inertia of a coreless unit, so
  # through the ~250:1 gearing the reflected armature is plausibly order
  # 1e-3 kg*m^2 -- the 5e-4 floor is conservative. If a fit pins at the
  # floor, a 1e-3 floor is the physically motivated variant to try.
  "armature": (5e-4, 0.02),
  "frictionloss": (0.0, 0.1),
  # qd_knee: bounds kept, but UNIDENTIFIABLE without load (nothing in an
  # unloaded session sits on the torque-speed taper below saturation). For
  # unloaded-only fits pin it at 3.65 (the v3 fitted value) pending
  # stance-loading data; the fit CLI enforces the pin (2026-07-15).
  "qd_knee": (3.0, 10.0),
  # qd_max (8.0, 14.0) -> (8.5, 10.5) (2026-07-15): measured unloaded ramp
  # plateaus are 8.9-10.1 rad/s per joint at full charge (v2.4 bench
  # session 2026-07-15_14-24-44_midrange).
  "qd_max": (8.5, 10.5),
  "deadband": (0.0, 0.06),
  "delay_ms": (5.0, 90.0),
  # tau_max FITTABLE bounds (2026-07-15): stall torque may be freed ONLY
  # when loaded (stand) sessions are among the fit inputs (identifiability
  # anchor otherwise — see module docstring). Default fit configs keep it
  # pinned at 0.22.
  "tau_max": (0.15, 0.35),
}
JOINT_PARAM_NAMES = ("kp", "kd", "damping", "armature", "frictionloss",
                     "qd_knee", "qd_max", "deadband")
# Full per-servo constructor key set (JOINT_PARAM_NAMES + the torque-scale
# anchor). delay_ms is handled separately (single global convention).
SERVO_PARAM_NAMES = (*JOINT_PARAM_NAMES, "tau_max")
DEFAULT_PARAMS: dict[str, float] = {
  # The CURRENT training stack = the "before fit" baseline: XML position
  # gains + MJCF joint defaults + the one-sided TorqueSpeedClamp with
  # tau_stall 0.22 / omega_nl 4.5 (a single line drooping from qd = 0,
  # i.e. qd_knee = 0, qd_max = 4.5 in the piecewise form).
  "kp": 5.0,
  "kd": 0.12,
  "damping": 0.05,
  "armature": 0.002,
  "frictionloss": 0.001,
  "qd_knee": 0.0,
  "qd_max": 4.5,
  "deadband": 0.0,  # current stack has no lost-motion term
  "delay_ms": 54.0,  # measured median onset latency
  "tau_max": TAU_MAX_DEFAULT,
}


@dataclass(frozen=True)
class ServoParams:
  """One servo's model parameters (plus the shared delay convention)."""

  kp: float
  kd: float
  damping: float
  armature: float
  frictionloss: float
  qd_knee: float
  qd_max: float
  deadband: float
  delay_ms: float
  tau_max: float = TAU_MAX_DEFAULT

  def __post_init__(self) -> None:
    # Keep the taper well-posed even when the box bounds allow
    # qd_knee > qd_max (knee up to 10, qd_max down to 8).
    if self.qd_max < self.qd_knee + QD_MAX_MARGIN:
      object.__setattr__(self, "qd_max", self.qd_knee + QD_MAX_MARGIN)

  def as_dict(self) -> dict[str, float]:
    return {
      "kp": self.kp, "kd": self.kd, "damping": self.damping,
      "armature": self.armature, "frictionloss": self.frictionloss,
      "qd_knee": self.qd_knee, "qd_max": self.qd_max,
      "deadband": self.deadband,
      "delay_ms": self.delay_ms, "tau_max": self.tau_max,
    }


def default_params(**overrides) -> ServoParams:
  return replace(ServoParams(**DEFAULT_PARAMS), **overrides)


def torque_speed_line(
  qd_abs: np.ndarray,
  tau_max: np.ndarray | float,
  qd_knee: np.ndarray | float,
  qd_max: np.ndarray | float,
) -> np.ndarray:
  """Piecewise driving-torque envelope tau_line(|qd|)."""
  with np.errstate(divide="ignore", invalid="ignore"):
    taper = tau_max * (qd_max - qd_abs) / np.maximum(qd_max - qd_knee, 1e-9)
  return np.where(qd_abs <= qd_knee, tau_max, np.clip(taper, 0.0, tau_max))


def pd_clamped_torque(
  q: np.ndarray,
  qd: np.ndarray,
  q_des: np.ndarray,
  kp: np.ndarray,
  kd: np.ndarray,
  deadband: np.ndarray,
  tau_max: np.ndarray | float,
  qd_knee: np.ndarray,
  qd_max: np.ndarray,
) -> np.ndarray:
  """PD torque with deadband and the asymmetric torque-speed clamp.

  Vectorized; every gain argument is an array like ``kp``. Matches
  ToddlerBot's MotorController.step with tau_brake_max == tau_max and
  tau at qd_max == 0 (their Dynamixel self-protection branch is dropped:
  these hobby servos have no such firmware mode), plus the lost-motion
  deadband on the position error.
  """
  err = q_des - q
  # sign(err) * max(|err| - deadband, 0), written clamp-style so the warp
  # kernel mirrors it expression-for-expression.
  err_eff = err - np.clip(err, -deadband, deadband)
  tau = kp * err_eff - kd * qd
  line = torque_speed_line(np.abs(qd), tau_max, qd_knee, qd_max)
  hi = np.where(qd > 0.0, line, tau_max)
  lo = np.where(qd < 0.0, -line, -tau_max)
  return np.clip(tau, lo, hi)


def normalize(values: np.ndarray, names: list[str]) -> np.ndarray:
  """Physical -> [-1, 1] (linear in PARAM_BOUNDS)."""
  lo = np.array([PARAM_BOUNDS[n][0] for n in names])
  hi = np.array([PARAM_BOUNDS[n][1] for n in names])
  return 2.0 * (np.asarray(values, dtype=float) - lo) / (hi - lo) - 1.0


def denormalize(z: np.ndarray, names: list[str]) -> np.ndarray:
  """[-1, 1] -> physical, clipped to bounds."""
  lo = np.array([PARAM_BOUNDS[n][0] for n in names])
  hi = np.array([PARAM_BOUNDS[n][1] for n in names])
  z = np.clip(np.asarray(z, dtype=float), -1.0, 1.0)
  return lo + (z + 1.0) / 2.0 * (hi - lo)
