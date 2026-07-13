"""Aggregate gait_autopsy.py outputs into comparison tables.

Reads <dir>/{preset}_summary.json + the per-bucket npz raws and prints:
  1. per-bucket gait table (tracking, force share, GRFx front/rear, swing,
     phase, slip, friction utilization);
  2. actuator-envelope occupancy vs the identification validity envelope
     (|qvel| percentiles vs the ~4.4 rad/s sim/hardware no-load slew,
     torque-cap saturation fraction, qvel direction-reversal rate = backlash
     exposure per second).

Usage: .venv/bin/python scripts/gait_autopsy_report.py <dir> [preset ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

FOOT_NAMES = ("fl", "fr", "bl", "br")
NOLOAD_SLEW = 4.4  # rad/s: forcerange 0.22 / joint damping 0.05; matches the
                   # 2026-07-05 hardware step fit (~0.45 rad in ~100 ms).


def reversal_rate(qvel: np.ndarray, dt: float, thresh: float = 0.5) -> float:
  """Mean direction reversals per joint per second (hysteresis at +/-thresh)."""
  # qvel: (T, C, J). State machine per (env, joint): sign of last excursion
  # beyond +/-thresh; a reversal = crossing to the opposite side.
  T, C, J = qvel.shape
  sign = np.zeros((C, J), dtype=np.int8)
  revs = np.zeros((C, J), dtype=np.int64)
  for t in range(T):
    pos = qvel[t] > thresh
    neg = qvel[t] < -thresh
    revs += ((sign == 1) & neg) | ((sign == -1) & pos)
    sign = np.where(pos, 1, np.where(neg, -1, sign))
  return float(revs.mean() / (T * dt))


def main() -> None:
  d = Path(sys.argv[1])
  presets = sys.argv[2:] or ["v17", "fastclock", "sprint", "agile"]
  dt = 0.02

  rows = []
  occ_rows = []
  for p in presets:
    summary = json.loads((d / f"{p}_summary.json").read_text())
    for bucket, r in summary.items():
      f = lambda key: [r[key][n] for n in FOOT_NAMES]
      front = lambda v: 0.5 * (v[0] + v[1])
      rear = lambda v: 0.5 * (v[2] + v[3])
      npz = np.load(d / f"{p}_{bucket}.npz")
      qvel = npz["qvel"][:, npz["clean"]]
      rev = reversal_rate(qvel, dt)
      rows.append({
        "preset": p, "bucket": bucket,
        "cmd_vx": r["cmd"][0], "cmd_wz": r["cmd"][2],
        "vx": r["vx_ach"], "wz": r["wz_ach"], "falls": r["falls"],
        "f0": r["f0_hz_mean"],
        "front_share": r["front_force_share"],
        "grfx_F": front(f("grfx_stance_mean")),
        "grfx_R": rear(f("grfx_stance_mean")),
        "zexc_F": front(f("foot_z_excursion")) * 1e3,
        "zexc_R": rear(f("foot_z_excursion")) * 1e3,
        "thigh_F": r["joint_excursion"]["front_thigh"],
        "thigh_R": r["joint_excursion"]["rear_thigh"],
        "calf_F": r["joint_excursion"]["front_calf"],
        "calf_R": r["joint_excursion"]["rear_calf"],
        "ph_flbr": r["phase"]["fl_br"]["mean_cycles"],
        "R_flbr": r["phase"]["fl_br"]["R"],
        "ph_flfr": abs(r["phase"]["fl_fr"]["mean_cycles"]),
        "R_flfr": r["phase"]["fl_fr"]["R"],
        "slip_F": front(f("slip_stance_mean")) * 1e3,
        "slip_R": rear(f("slip_stance_mean")) * 1e3,
        "fric95_F": front(f("fric_util_p95")),
        "fric95_R": rear(f("fric_util_p95")),
        "duty_F": front(f("duty")), "duty_R": rear(f("duty")),
        "rev_hz": rev,
        "act_rate": r["action_rate_mean"],
      })
      for grp, o in r["occupancy"].items():
        occ_rows.append({"preset": p, "bucket": bucket, "grp": grp, **o})

  hdr = ("preset     bucket        cmd(vx,wz)   vx_ach  falls f0Hz  Fshare "
         " GRFx F/R      zexc F/R mm  thigh F/R   calf F/R   ph(fl_br,R) "
         "|fl_fr|  slip F/R mm/s fric95 F/R  duty F/R  rev/s  actrate")
  print(hdr)
  print("-" * len(hdr))
  for r in rows:
    print(f"{r['preset']:10s} {r['bucket']:13s} "
          f"({r['cmd_vx']:+.2f},{r['cmd_wz']:+.1f}) "
          f"{r['vx']:+.3f} {r['falls']:5d} {r['f0']:.2f} {r['front_share']:6.3f} "
          f"{r['grfx_F']:+.2f}/{r['grfx_R']:+.2f} "
          f"{r['zexc_F']:5.1f}/{r['zexc_R']:5.1f}  "
          f"{r['thigh_F']:.2f}/{r['thigh_R']:.2f}  "
          f"{r['calf_F']:.2f}/{r['calf_R']:.2f}  "
          f"{r['ph_flbr']:+.3f},{r['R_flbr']:.2f} "
          f"{r['ph_flfr']:.3f}   "
          f"{r['slip_F']:4.0f}/{r['slip_R']:4.0f}   "
          f"{r['fric95_F']:.2f}/{r['fric95_R']:.2f} "
          f"{r['duty_F']:.2f}/{r['duty_R']:.2f} {r['rev_hz']:5.1f} "
          f"{r['act_rate']:.3f}")

  print("\nActuator-envelope occupancy (|qvel| rad/s; ID validity: no-load "
        f"slew ~{NOLOAD_SLEW}; tau cap 0.22 nominal)")
  print("dcviol% = samples delivering positive mechanical power ABOVE the "
        "DC-servo torque-speed line\n  (tau*qd>0 and |tau|/0.22 + "
        f"|qd|/{NOLOAD_SLEW} > 1) — physically undeliverable by a motor "
        "whose line passes\n  through (0.22 N*m stall, 4.4 rad/s no-load); "
        "the sim actuator only pays a fixed 0.05*qd joint damping.")
  print(f"{'preset':10s} {'bucket':13s} {'group':12s} "
        f"{'p50':>5s} {'p95':>5s} {'p99':>5s} "
        f"{'tau_p95':>7s} {'sat%':>5s} {'dcviol%':>8s} {'pwr_p95':>8s}")
  joint_groups = {
    "front_thigh": (1, 4), "rear_thigh": (7, 10),
    "front_calf": (2, 5), "rear_calf": (8, 11),
  }
  for p in presets:
    summary = json.loads((d / f"{p}_summary.json").read_text())
    for bucket, r in summary.items():
      npz = np.load(d / f"{p}_{bucket}.npz")
      clean = npz["clean"]
      qvel = npz["qvel"][:, clean]
      tau = npz["tau"][:, clean]
      for grp in ("front_thigh", "rear_thigh", "front_calf", "rear_calf"):
        ids = list(joint_groups[grp])
        qd, tq = qvel[:, :, ids], tau[:, :, ids]
        drive = tq * qd > 0  # delivering, not absorbing
        viol = float((drive
                      & (np.abs(tq) / 0.22 + np.abs(qd) / NOLOAD_SLEW > 1.0)
                      ).mean())
        pwr = np.maximum(tq * qd, 0.0)
        o = r["occupancy"][grp]
        print(f"{p:10s} {bucket:13s} {grp:12s} "
              f"{o['qvel_p50']:5.2f} {o['qvel_p95']:5.2f} {o['qvel_p99']:5.2f} "
              f"{o['tau_p95']:7.3f} {o['tau_sat_frac']:5.1%} "
              f"{viol:8.1%} {pctl(pwr, 95):8.3f}")


def pctl(x: np.ndarray, q: float) -> float:
  return float(np.percentile(x, q)) if x.size else float("nan")


if __name__ == "__main__":
  main()
