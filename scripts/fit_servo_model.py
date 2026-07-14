#!/usr/bin/env python3
"""Stage-1 servo-model fit from servo-ID capture sessions.

Fits the ToddlerBot actuator model (PD gains + deadband + piecewise
torque-speed clamp + MJCF damping/armature/frictionloss) plus ONE global
command delay to recorded excitation sessions, with PACE-style CMA-ES
(population = parallel replays, params normalized to [-1, 1], sigma0 0.5,
early stop on relative population score spread < 1e-2). The loaded
(stand) regime is replayed honestly: free base + ground plane, non-swept
joints PD-held at the recorded base pose, swept joint following the
recorded targets ZOH on the firmware clock. tau_max is PINNED (default
0.22 N*m) as the torque scale anchor — never co-fit it with
kp/damping/armature (spec pitfall).

Block-coordinate scheme (fit v2): --mode both runs the shared-class fit
FIRST, then per-joint fits with the other 11 joints HELD at the
shared-fit values (fit v1 held them at DEFAULT_PARAMS, which biased
every per-joint fit and broke the all-fitted deployment world). --mode
per-joint alone requires --held-from FIT.json (a previous schema-v2 fit
whose shared_class supplies the held values); there is no DEFAULT-held
fallback.

Local smoke (dev fixture, tiny budget):
  cd luwu_mjlab && PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python \\
    scripts/fit_servo_model.py ../runs/servo_id/2026-07-13_18-04-06_stand \\
    --mode both --generations 8 --popsize 8 --out /tmp/fit_smoke.json

Full fit (documented for the GPU box — CPU-parallel, ~64 pop x 150 gens):
  PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/fit_servo_model.py \\
    ../runs/servo_id/2026-07-13_18-07-30_stand \\
    ../runs/servo_id/2026-07-13_18-04-06_stand \\
    --mode both --generations 150 --popsize 64 --workers 24 \\
    --out ../runs/servo_id/fit_2026-07-13.json

Backends: --backend cpu (fork-pool plain MuJoCo, default) or
--backend warp (GPU-batched mujoco_warp: one generation = one batched
rollout of popsize x captures worlds, chunked to --max-envs; see
src/servo_id/replay_warp.py for semantics and CPU-agreement gates).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.servo_id import LEG_JOINTS
from src.servo_id.actuator_model import (
  DEFAULT_PARAMS,
  JOINT_PARAM_NAMES,
  PARAM_BOUNDS,
  normalize,
)
from src.servo_id.fitting import Evaluator, FitConfig, FitResult, run_cma_fit
from src.servo_id.loader import Session, TestCapture, load_session
from src.servo_id.replay import (
  build_replay_model,
  replay_capture,
  segment_metrics,
)

FIT_SCHEMA = "xgo-servo-fit/v2"
ALL_FIT_NAMES = [*JOINT_PARAM_NAMES, "delay_ms"]
VALIDATION_VARIANTS = ("baseline", "shared", "per_joint")


def eval_joints_for(capture: TestCapture) -> tuple[str, ...]:
  """Joints whose telemetry is scored: the excited leg joints."""
  return tuple(j for j in capture.excited if j in LEG_JOINTS)


def collect_captures(
  sessions: list[Session],
) -> list[tuple[TestCapture, tuple[str, ...]]]:
  out = []
  for sess in sessions:
    for cap in sess.tests:
      joints = eval_joints_for(cap)
      if joints:
        out.append((cap, joints))
  return out


def validation_table(
  captures: list[tuple[TestCapture, tuple[str, ...]]],
  evaluator: Evaluator,
  shared_values: dict[str, float],
  per_joint_values: dict[str, dict[str, float]],
  global_delay_ms: float,
) -> list[dict]:
  """Per-segment sim-vs-real RMSE for the three deployment variants.

  baseline   DEFAULT_PARAMS everywhere, DEFAULT delay;
  shared     the shared-fit set on all 12 servos, shared-fit delay;
  per_joint  ALL fitted sets applied at once (the deployment world);
             joints without a per-joint fit fall back to the shared set;
             the replay delay is the excited joint's own fitted delay.
  """
  from src.servo_id.actuator_model import ServoParams

  rm = evaluator.rm

  def mk_params(values_by_joint: dict[str, dict[str, float]]):
    plist = []
    for name in LEG_JOINTS:
      d = dict(DEFAULT_PARAMS)
      d.update(evaluator.cfg.fixed)
      d.update(
        {k: v for k, v in values_by_joint.get(name, {}).items()
         if k in JOINT_PARAM_NAMES or k == "delay_ms"}
      )
      plist.append(
        ServoParams(**{k: d[k] for k in (*JOINT_PARAM_NAMES, "delay_ms")})
      )
    return plist

  shared_by_joint = {j: shared_values for j in LEG_JOINTS}
  pj_by_joint = {
    j: per_joint_values.get(j, shared_values) for j in LEG_JOINTS
  }
  shared_delay = float(shared_values.get(
    "delay_ms", DEFAULT_PARAMS["delay_ms"]
  ))

  variant_params = {
    "baseline": mk_params({}),
    "shared": mk_params(shared_by_joint),
    "per_joint": mk_params(pj_by_joint),
  }

  rows: list[dict] = []
  for cap, joints in captures:
    assert len(joints) == 1, (
      f"{cap.path.name}: validation assumes exactly one excited leg "
      f"joint per capture, got {joints}"
    )
    (joint,) = joints
    variant_delay = {
      "baseline": DEFAULT_PARAMS["delay_ms"],
      "shared": shared_delay,
      "per_joint": float(
        per_joint_values.get(joint, {}).get("delay_ms", global_delay_ms)
      ),
    }
    metrics = {
      v: segment_metrics(
        replay_capture(rm, cap, variant_params[v], variant_delay[v]), cap
      )
      for v in VALIDATION_VARIANTS
    }
    for recs in zip(*(metrics[v] for v in VALIDATION_VARIANTS)):
      row = dict(recs[0])
      row.pop("rmse")
      row.pop("amp_ratio_sim", None)
      for v, rec in zip(VALIDATION_VARIANTS, recs):
        row[f"rmse_{v}"] = rec["rmse"]
        if "amp_ratio_sim" in rec:
          row[f"amp_ratio_sim_{v}"] = rec["amp_ratio_sim"]
      rows.append(row)
  return rows


def format_report(
  fit: dict,
  validation: list[dict],
) -> str:
  lines = []
  lines.append("Servo-model fit report")
  lines.append(f"created: {fit['created']}")
  lines.append(f"sessions: {[s['path'] for s in fit['sessions']]}")
  lines.append(f"settings: {json.dumps(fit['settings'])}")
  lines.append("")
  lines.append(f"global delay: {fit['global_delay_ms']:.1f} ms")
  lines.append(f"held-from (pass-2 held-joint values): {fit['held_from']}")
  lines.append("")
  if fit.get("per_joint"):
    lines.append("per-joint fits, pass 2, other joints held at the "
                 "shared-fit values (loss = MSE + 0.01*FFT-RMSE<10Hz):")
    hdr = (f"{'joint':<10s} {'loss_before':>12s} {'loss_after':>11s} "
           + " ".join(f"{n:>10s}" for n in ALL_FIT_NAMES))
    lines.append(hdr)
    for joint, rec in fit["per_joint"].items():
      vals = " ".join(
        f"{rec['params'].get(n, float('nan')):>10.4f}"
        for n in ALL_FIT_NAMES
      )
      lines.append(
        f"{joint:<10s} {rec['baseline_loss']:>12.3e} "
        f"{rec['loss']:>11.3e} {vals}"
      )
    lines.append("")
  if fit.get("shared_class"):
    rec = fit["shared_class"]
    lines.append("shared-class fit (one parameter set for all 12 servos):")
    lines.append(
      f"  loss {rec['baseline_loss']:.3e} -> {rec['loss']:.3e}; params "
      + json.dumps({k: round(v, 5) for k, v in rec["params"].items()})
    )
    lines.append("")
  lines.append(
    "per-segment sim-vs-real RMSE (rad); variants: baseline = current "
    "stack, shared = one fitted set on all 12 servos, per_joint = all "
    "fitted sets applied at once (deployment world):"
  )
  lines.append(
    f"{'file':<28s} {'joint':<10s} {'segment':<26s} "
    f"{'baseline':>8s} {'shared':>8s} {'per_jnt':>8s} {'amp r/pj':>12s}"
  )
  for row in validation:
    desc = row["desc"]
    if "freq_hz" in desc:
      seg = f"sine a={desc.get('amplitude', 0):.2f} f={desc['freq_hz']:.1f}"
    elif "target" in desc:
      seg = f"step to {desc['target']:+.2f}"
    else:
      seg = str(desc)[:26]
    amp = ""
    if "amp_ratio_real" in row:
      amp = (f"{row['amp_ratio_real']:.2f}/"
             f"{row.get('amp_ratio_sim_per_joint', float('nan')):.2f}")
    lines.append(
      f"{row['file']:<28s} {row['joint']:<10s} {seg:<26s} "
      f"{row['rmse_baseline']:>8.4f} {row['rmse_shared']:>8.4f} "
      f"{row['rmse_per_joint']:>8.4f} {amp:>12s}"
    )
  small = [
    r for r in validation
    if r.get("amp_cmd") == 0.08 and "amp_ratio_real" in r
  ]
  if small:
    lines.append("")
    lines.append(
      "small-amplitude (0.08 rad) attenuation check "
      "(real measured 0.77-0.91; the deadband term should reproduce it):"
    )
    for r in small:
      sims = " ".join(
        f"{v} {r.get(f'amp_ratio_sim_{v}', float('nan')):.2f}"
        for v in VALIDATION_VARIANTS
      )
      lines.append(
        f"  {r['file']} {r['joint']} f={r['desc'].get('freq_hz')}: "
        f"real {r['amp_ratio_real']:.2f} sim: {sims}"
      )
  lines.append("")
  lines.append("segment RMSE summary per variant (rad):")
  for v in VALIDATION_VARIANTS:
    vals = [r[f"rmse_{v}"] for r in validation]
    if vals:
      lines.append(
        f"  {v:<10s} mean {float(np.mean(vals)):.4f} "
        f"median {float(np.median(vals)):.4f} (n={len(vals)})"
      )
  return "\n".join(lines) + "\n"


def main() -> int:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("sessions", nargs="+", help="capture session directories")
  ap.add_argument("--out", required=True, help="output fit.json path")
  ap.add_argument("--report", default=None,
                  help="text report path (default: <out>.txt)")
  ap.add_argument("--mode", choices=("per-joint", "shared", "both"),
                  default="both",
                  help="both = shared fit first, then per-joint fits held "
                       "at the shared values; per-joint alone requires "
                       "--held-from")
  ap.add_argument("--held-from", default=None,
                  help="previous schema-v2 fit.json whose shared_class "
                       "supplies the held-joint values (required for "
                       "--mode per-joint)")
  ap.add_argument("--joints", default=None,
                  help="comma filter of excited joints to fit")
  ap.add_argument("--kinds", default=None,
                  help="comma filter of test kinds (steps,sine_grid,...)")
  ap.add_argument("--fit-params", default=",".join(ALL_FIT_NAMES),
                  help=f"comma subset of {','.join(ALL_FIT_NAMES)}")
  ap.add_argument("--generations", type=int, default=150)
  ap.add_argument("--popsize", type=int, default=64)
  ap.add_argument("--sigma0", type=float, default=0.5)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--workers", type=int, default=None)
  ap.add_argument("--tau-max", type=float, default=0.22,
                  help="PINNED stall torque (identifiability anchor)")
  ap.add_argument("--sim-dt", type=float, default=0.002)
  ap.add_argument("--settle", type=float, default=1.0)
  ap.add_argument("--skip-ms", type=float, default=300.0)
  ap.add_argument("--backend", choices=("cpu", "warp"), default="cpu",
                  help="population evaluator: fork-pool plain MuJoCo or "
                       "GPU-batched mujoco_warp")
  ap.add_argument("--max-envs", type=int, default=None,
                  help="warp backend: worlds per batched rollout chunk "
                       "(default from replay_warp.DEFAULT_MAX_ENVS)")
  args = ap.parse_args()

  fit_names = [s.strip() for s in args.fit_params.split(",") if s.strip()]
  for name in fit_names:
    if name not in PARAM_BOUNDS:
      ap.error(f"unknown fit param {name!r}")
  kinds = tuple(
    s.strip() for s in args.kinds.split(",")) if args.kinds else None
  joints = tuple(
    s.strip() for s in args.joints.split(",")) if args.joints else None

  shared_values: dict[str, float] | None = None
  held_from = "shared"
  inherited_shared_rec: dict | None = None
  # Fail fast: the per-joint pass NEVER runs against DEFAULT-held joints
  # (the v1 bug) — it needs shared values from this run or a previous fit.
  if args.mode == "per-joint":
    if not args.held_from:
      ap.error("--mode per-joint requires --held-from FIT.json (per-joint "
               "fits against DEFAULT-held joints are the v1 bug; hold "
               "them at a shared fit instead)")
    prev = json.loads(Path(args.held_from).read_text())
    if prev.get("schema") != FIT_SCHEMA:
      raise SystemExit(
        f"--held-from {args.held_from}: unsupported fit schema "
        f"{prev.get('schema')!r} (need {FIT_SCHEMA!r})"
      )
    if not prev.get("shared_class"):
      raise SystemExit(
        f"--held-from {args.held_from}: fit has no shared_class record"
      )
    shared_values = {
      k: v for k, v in prev["shared_class"]["params"].items()
      if k in PARAM_BOUNDS
    }
    held_from = str(args.held_from)
    print(f"held-joint values loaded from {held_from}")
    # Carry the pass-1 record forward so the pass-2 output json is a
    # self-contained fit artifact (backend may differ between passes).
    inherited_shared_rec = prev["shared_class"]

  sessions = []
  for sdir in args.sessions:
    sess = load_session(Path(sdir), kinds=kinds, joints=joints)
    regime = "LOADED (stand)" if sess.loaded else "unloaded (folded)"
    print(f"session {sess.path}: {regime}, {len(sess.tests)} tests")
    sessions.append(sess)
  captures = collect_captures(sessions)
  if not captures:
    print("no captures to fit", file=sys.stderr)
    return 1
  for cap, ej in captures:
    a = cap.alignment
    print(
      f"  {cap.path.name}: joints {ej}, cmd offset {a.offset_ms:.1f} ms, "
      f"receipt jitter std {a.jitter_std_ms:.2f} ms "
      f"(absmax {a.jitter_absmax_ms:.1f})"
    )

  cfg = FitConfig(
    fit_names=fit_names,
    fixed={"tau_max": args.tau_max},
    sim_dt=args.sim_dt,
    settle_s=args.settle,
    skip_ms=args.skip_ms,
    fixed_base=False,
  )

  def make_evaluator(caps, target_joint, held_values=None):
    if args.backend == "warp":
      from src.servo_id.replay_warp import DEFAULT_MAX_ENVS, WarpEvaluator

      return WarpEvaluator(
        caps, cfg, target_joint, held_values=held_values,
        max_envs=args.max_envs or DEFAULT_MAX_ENVS,
      )
    return Evaluator(caps, cfg, target_joint, held_values=held_values)

  per_joint: dict[str, dict] = {}
  per_joint_values: dict[str, dict[str, float]] = {}
  shared_rec: dict | None = inherited_shared_rec
  delays: list[float] = []

  def result_record(res: FitResult, evaluator) -> dict:
    params, _ = evaluator.params_from_values(res.values)
    col = (
      LEG_JOINTS.index(res.target_joint) if res.target_joint else 0
    )
    return {
      "params": params[col].as_dict(),
      "fitted_names": res.fit_names,
      "loss": res.loss,
      "baseline_loss": res.baseline_loss,
      "generations_run": res.generations_run,
      "early_stopped": res.early_stopped,
      "wall_s": round(res.wall_s, 1),
      "loss_trace": res.loss_trace,
    }

  # Pass 1: shared-class fit (unless supplied via --held-from above).
  if args.mode in ("shared", "both"):
    print(f"fitting shared class on {len(captures)} captures ...")
    evaluator = make_evaluator(captures, None)
    res = run_cma_fit(
      evaluator, args.generations, args.popsize, args.sigma0,
      seed=args.seed, workers=args.workers,
    )
    shared_rec = result_record(res, evaluator)
    shared_values = {
      k: v for k, v in shared_rec["params"].items() if k in PARAM_BOUNDS
    }

  assert shared_values is not None
  shared_delay = float(shared_values.get(
    "delay_ms", DEFAULT_PARAMS["delay_ms"]
  ))

  # Pass 2: per-joint fits with the other 11 joints held at the shared
  # values (block-coordinate descent). Warm-started at the shared solution
  # with a tighter sigma: pass 2 refines per-joint deltas, it does not
  # re-explore the whole cube.
  if args.mode in ("per-joint", "both"):
    held_values = {j: dict(shared_values) for j in LEG_JOINTS}
    x0_pass2 = normalize(
      np.array([shared_values.get(n, DEFAULT_PARAMS[n]) for n in fit_names]),
      fit_names,
    )
    sigma_pass2 = min(args.sigma0, 0.3)
    by_joint: dict[str, list] = {}
    for cap, ej in captures:
      for j in ej:
        by_joint.setdefault(j, []).append((cap, (j,)))
    for joint in sorted(by_joint, key=LEG_JOINTS.index):
      print(f"fitting {joint} on {len(by_joint[joint])} captures "
            "(others held at shared values) ...")
      evaluator = make_evaluator(by_joint[joint], joint,
                                 held_values=held_values)
      res = run_cma_fit(
        evaluator, args.generations, args.popsize, sigma_pass2,
        seed=args.seed, workers=args.workers, x0=x0_pass2,
      )
      per_joint[joint] = result_record(res, evaluator)
      # Full param set, not res.values: a restricted fit's values dict
      # holds only the fitted subset, and the validation table must see
      # the inherited shared values for the pinned parameters too.
      per_joint_values[joint] = {
        k: v for k, v in per_joint[joint]["params"].items()
        if k in PARAM_BOUNDS
      }
      if "delay_ms" in res.values:
        delays.append(res.values["delay_ms"])

  global_delay = float(np.median(delays)) if delays else shared_delay

  print("building validation table (three-variant replays) ...")
  evaluator = Evaluator(captures, cfg, target_joint=None)
  validation = validation_table(
    captures, evaluator, shared_values, per_joint_values, global_delay
  )

  fit = {
    "schema": FIT_SCHEMA,
    "created": datetime.now().isoformat(timespec="seconds"),
    "sessions": [
      {
        "path": str(s.path),
        "base_pose": s.base_pose_name,
        "loaded": s.loaded,
        "n_tests": len(s.tests),
      }
      for s in sessions
    ],
    "settings": {
      "mode": args.mode,
      "fit_names": fit_names,
      "generations": args.generations,
      "popsize": args.popsize,
      "sigma0": args.sigma0,
      "seed": args.seed,
      "tau_max_pinned": args.tau_max,
      "sim_dt": args.sim_dt,
      "settle_s": args.settle,
      "skip_ms": args.skip_ms,
      "regime": "loaded_free_base_contact",
      "bounds": {k: PARAM_BOUNDS[k] for k in fit_names},
    },
    "alignment": {
      cap.path.name: {
        "offset_ms": cap.alignment.offset_ms,
        "n_events": cap.alignment.n_events,
        "slope": cap.alignment.slope,
        "jitter_std_ms": cap.alignment.jitter_std_ms,
        "jitter_absmax_ms": cap.alignment.jitter_absmax_ms,
      }
      for cap, _ in captures
    },
    "per_joint": per_joint,
    "shared_class": shared_rec,
    "held_from": held_from,
    "global_delay_ms": global_delay,
    "validation": {
      "variants": list(VALIDATION_VARIANTS),
      "segments": validation,
    },
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(fit, indent=2))
  report = format_report(fit, validation)
  report_path = Path(args.report) if args.report else out.with_suffix(
    out.suffix + ".txt"
  )
  report_path.write_text(report)
  print(report)
  print(f"fit written to {out}\nreport written to {report_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
