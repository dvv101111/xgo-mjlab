#!/usr/bin/env python3
"""Map a servo-model fit (fit.json) into the training stack.

Prints — and with ``--write-cfg`` generates a config helper module for —
the concrete stack changes implied by the fit:

* MJCF joint fields per joint: damping / armature / frictionloss (current
  XML defaults 0.05 / 0.002 / 0.001);
* position-actuator gains per joint: kp / kv (current XML 5.0 / 0.12);
* TorqueSpeedClamp per servo: the fitted piecewise curve (tau_max flat to
  qd_knee, tapering to 0 at qd_max), consumed directly by the knee-form
  clamp (v18 hand-set envelope: tau_max 0.22 / qd_knee 0 / qd_max 4.5);
* per-servo delay DR range centered on the fitted global delay with the
  measured onset spread 33-77 ms (half-width 22 ms), in 2 ms physics
  steps (currently hand-set 30-50 steps = 60-100 ms);
* the fitted deadband (lost motion, rad) per joint — reported and stored
  for reference; the v19 preset randomizes the deadband wide (0-0.025 rad)
  instead of pinning the nominals (measured small-amplitude delivery
  0.79-0.93 implies up to ~0.02 rad effective lost motion);
* DR shrinkage recommendation per the spec: DROP strength/gain DR on
  fitted joints, KEEP link mass/CoM DR and per-servo delay DR (stochastic
  bus jitter is not a deterministic residual).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/apply_servo_fit.py fit.json
  PYTHONPATH=. .venv/bin/python scripts/apply_servo_fit.py fit.json \\
    --write-cfg src/tasks/velocity/config/xgolite/measured_actuators.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.servo_id import LEG_JOINTS

# Current hand-set stack constants (deltas are reported against these).
CURRENT = {
  "damping": 0.05, "armature": 0.002, "frictionloss": 0.001,
  "kp": 5.0, "kd": 0.12,
  "tau_max": 0.22, "qd_knee": 0.0, "qd_max": 4.5,
  "delay_min_ms": 60.0, "delay_max_ms": 100.0,
}
DELAY_SPREAD_HALF_MS = 22.0  # measured onset spread 33-77 ms
PHYSICS_DT_MS = 2.0


def joint_params(fit: dict) -> dict[str, dict]:
  """Per-joint fitted params; shared-class fills any missing joint."""
  out: dict[str, dict] = {}
  shared = (fit.get("shared_class") or {}).get("params")
  for joint in LEG_JOINTS:
    rec = (fit.get("per_joint") or {}).get(joint)
    if rec:
      out[joint] = dict(rec["params"], source="per_joint")
    elif shared:
      out[joint] = dict(shared, source="shared_class")
  return out


def delay_dr(delay_ms: float) -> tuple[int, int]:
  lo = max(delay_ms - DELAY_SPREAD_HALF_MS, 0.0)
  hi = delay_ms + DELAY_SPREAD_HALF_MS
  return (
    max(1, int(round(lo / PHYSICS_DT_MS))),
    max(2, int(round(hi / PHYSICS_DT_MS))),
  )


def print_mapping(fit: dict) -> None:
  params = joint_params(fit)
  if not params:
    raise SystemExit("fit.json has neither per_joint nor shared_class params")
  delay = float(fit["global_delay_ms"])

  print("=== MJCF joint fields + PD gains (current XML defaults in []) ===")
  print(
    f"{'joint':<10s} {'damping':>18s} {'armature':>18s} "
    f"{'frictionloss':>18s} {'kp':>14s} {'kd':>14s} {'deadband':>9s}"
  )
  for joint, p in params.items():
    print(
      f"{joint:<10s}"
      f" {p['damping']:>10.4f} [{CURRENT['damping']:.3f}]"
      f" {p['armature']:>10.5f} [{CURRENT['armature']:.3f}]"
      f" {p['frictionloss']:>10.4f} [{CURRENT['frictionloss']:.3f}]"
      f" {p['kp']:>7.3f} [{CURRENT['kp']:.1f}]"
      f" {p['kd']:>7.3f} [{CURRENT['kd']:.2f}]"
      f" {p['deadband']:>9.4f}"
    )
  print(
    "  NOTE: deadband (lost motion, rad) is reported and stored for"
    " reference; the v19 preset randomizes the deadband wide (0-0.025 rad)"
    " instead of pinning these nominals."
  )

  print()
  print(
    "=== TorqueSpeedClamp, knee form (current hand-set: tau_max "
    f"{CURRENT['tau_max']:.2f} N*m, qd_knee {CURRENT['qd_knee']:.1f}, "
    f"qd_max {CURRENT['qd_max']:.1f} rad/s) ==="
  )
  print(
    f"{'joint':<10s} {'tau_max':>8s} {'qd_knee':>8s} {'qd_max':>8s} "
    f"{'d_qd_max':>9s}"
  )
  for joint, p in params.items():
    print(
      f"{joint:<10s} {p['tau_max']:>8.3f} {p['qd_knee']:>8.2f} "
      f"{p['qd_max']:>8.2f} "
      f"{p['qd_max'] - CURRENT['qd_max']:>+9.2f}"
    )

  lo, hi = delay_dr(delay)
  print()
  print("=== Per-servo delay DR ===")
  print(
    f"fitted global delay {delay:.1f} ms; recommended DR "
    f"[{delay - DELAY_SPREAD_HALF_MS:.0f}, "
    f"{delay + DELAY_SPREAD_HALF_MS:.0f}] ms = lag steps [{lo}, {hi}] at "
    f"{PHYSICS_DT_MS:.0f} ms physics"
  )
  print(
    f"  (current hand-set: [{CURRENT['delay_min_ms']:.0f}, "
    f"{CURRENT['delay_max_ms']:.0f}] ms = steps "
    f"[{int(CURRENT['delay_min_ms'] / PHYSICS_DT_MS)}, "
    f"{int(CURRENT['delay_max_ms'] / PHYSICS_DT_MS)}])"
  )

  print()
  print("=== DR shrinkage (spec recommendation) ===")
  print(
    "  DROP actuator strength/gain DR on the fitted joints (params are"
    " now measured);\n  KEEP link mass/CoM DR and the per-servo delay DR"
    " (bus jitter stays stochastic)."
  )


CFG_TEMPLATE = '''"""Measured XGO-Lite2 actuator constants (GENERATED — do not hand-edit).

Generated by scripts/apply_servo_fit.py on {created}
from {fit_path} (fit created {fit_created}, sessions: {sessions}).

Stage-1 servo fit: per-joint MJCF dynamics, PD gains, torque-speed clamp
and the global command delay. See the fit report next to the fit.json for
sim-vs-real validation. Regenerate with:
  PYTHONPATH=. .venv/bin/python scripts/apply_servo_fit.py {fit_path} \\
    --write-cfg src/tasks/velocity/config/xgolite/measured_actuators.py
"""

# joint -> dict(damping, armature, frictionloss, kp, kd,
#               qd_knee, qd_max, deadband)
# qd_knee/qd_max parameterize the fitted piecewise torque-speed curve
# (driving torque flat at TAU_MAX up to qd_knee, linear taper to zero at
# qd_max) consumed by the knee-form TorqueSpeedClamp.
# deadband (fitted lost-motion nominal, rad) is carried for reference: the
# v19 preset randomizes the deadband WIDE instead of pinning these — the
# measured small-amplitude delivery of 0.79-0.93 implies up to ~0.02 rad
# effective lost motion (dynamic hysteresis the model cannot express).
MEASURED_JOINTS = {joints_repr}

TAU_MAX = {tau_max}  # N*m, pinned fit anchor (== XML forcerange)

# Class means (for the scalar knee-form TorqueSpeedClamp API).
MEASURED_QD_KNEE = {qd_knee_mean}
MEASURED_QD_MAX = {qd_max_mean}

# Global command delay (firmware receipt -> actuation) and the
# recommended per-servo delay DR range (measured onset spread +/-22 ms),
# in physics steps at {physics_dt} ms.
MEASURED_DELAY_MS = {delay_ms}
DELAY_MIN_LAG_STEPS = {delay_min_lag}
DELAY_MAX_LAG_STEPS = {delay_max_lag}


def apply_measured_joint_dynamics(spec) -> None:
  """Write fitted MJCF joint fields + PD gains into an MjSpec in place.

  Call on the spec returned by ``xgolite_constants.get_spec()`` before
  compilation. Position actuator convention: gainprm[0] = kp,
  biasprm = (0, -kp, -kd).
  """
  for joint, p in MEASURED_JOINTS.items():
    j = spec.joint(joint + "_joint")
    j.damping = p["damping"]
    j.armature = p["armature"]
    j.frictionloss = p["frictionloss"]
    a = spec.actuator(joint + "_joint")
    a.gainprm[0] = p["kp"]
    a.biasprm[1] = -p["kp"]
    a.biasprm[2] = -p["kd"]


def enable_measured_torque_speed_clamp(cfg) -> None:
  """Wire the knee-form clamp with measured class-mean parameters.

  The clamp API is scalar (one tau_max/qd_knee/qd_max for all joints);
  per-joint values live in MEASURED_JOINTS for a future per-joint clamp.
  """
  from src.tasks.velocity.config.xgolite.sim_fidelity import (
    enable_torque_speed_clamp,
  )

  enable_torque_speed_clamp(
    cfg,
    tau_max=TAU_MAX,
    qd_knee=MEASURED_QD_KNEE,
    qd_max=MEASURED_QD_MAX,
  )


def measured_delay_actuator_cfg():
  """DelayedActuatorCfg with the measured delay DR range."""
  import dataclasses

  from src.assets.robots.xgolite.xgolite_constants import (
    XGOLITE_XML_ACTUATOR,
  )

  return dataclasses.replace(
    XGOLITE_XML_ACTUATOR,
    delay_min_lag=DELAY_MIN_LAG_STEPS,
    delay_max_lag=DELAY_MAX_LAG_STEPS,
  )
'''


def write_cfg(fit: dict, fit_path: Path, out_path: Path) -> None:
  params = joint_params(fit)
  delay = float(fit["global_delay_ms"])
  lo, hi = delay_dr(delay)
  joints = {}
  for joint, p in params.items():
    joints[joint] = {
      "damping": round(p["damping"], 6),
      "armature": round(p["armature"], 6),
      "frictionloss": round(p["frictionloss"], 6),
      "kp": round(p["kp"], 4),
      "kd": round(p["kd"], 4),
      "qd_knee": round(p["qd_knee"], 4),
      "qd_max": round(p["qd_max"], 4),
      "deadband": round(p["deadband"], 4),
    }
  joints_repr = "{\n" + "\n".join(
    f"  {j!r}: {v!r}," for j, v in joints.items()
  ) + "\n}"
  tau_max = next(iter(params.values()))["tau_max"]
  n = max(len(joints), 1)
  content = CFG_TEMPLATE.format(
    created=datetime.now().isoformat(timespec="seconds"),
    fit_path=fit_path,
    fit_created=fit.get("created"),
    sessions=[s["path"] for s in fit.get("sessions", [])],
    joints_repr=joints_repr,
    tau_max=tau_max,
    qd_knee_mean=round(
      sum(v["qd_knee"] for v in joints.values()) / n, 4
    ),
    qd_max_mean=round(
      sum(v["qd_max"] for v in joints.values()) / n, 4
    ),
    physics_dt=PHYSICS_DT_MS,
    delay_ms=round(delay, 2),
    delay_min_lag=lo,
    delay_max_lag=hi,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(content)
  print(f"\nconfig helper written to {out_path}")


def main() -> int:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("fit_json", help="fit.json from fit_servo_model.py")
  ap.add_argument(
    "--write-cfg", nargs="?", const="src/tasks/velocity/config/xgolite/"
    "measured_actuators.py", default=None,
    help="also generate the config helper module (optional path)",
  )
  args = ap.parse_args()
  fit_path = Path(args.fit_json)
  fit = json.loads(fit_path.read_text())
  if fit.get("schema") != "xgo-servo-fit/v2":
    raise SystemExit(
      f"unsupported fit schema {fit.get('schema')!r} (need "
      "'xgo-servo-fit/v2'; re-run scripts/fit_servo_model.py — v1 fits "
      "predate the deadband term and the block-coordinate per-joint pass)"
    )
  print_mapping(fit)
  if args.write_cfg:
    write_cfg(fit, fit_path, Path(args.write_cfg))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
