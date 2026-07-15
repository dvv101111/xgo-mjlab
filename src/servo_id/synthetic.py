"""Synthetic capture-session generation from a KNOWN parameter set.

Produces a session directory in the exact ``xgo-servo-id-session/v2``
format (manifest.json + per-test npz with every key the real capture tool
writes), by rolling out the same replay backend under truth parameters and
then re-sampling the trajectory the way the firmware telemetry does:

* telemetry rows on a ~10 ms push clock,
* per-servo position refresh only every ``refresh_ms`` (values repeat
  between refreshes -> exercises the dedupe path),
* commands stamped on a separate host clock; firmware receipt times
  (``cmd_ms``) carry bounded jitter -> exercises clock alignment,
* the truth command delay is applied between receipt and actuation.

Used by the loader unit tests and by the parameter-recovery acceptance
test (scripts/fit_servo_model.py --selftest).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from src.servo_id import CANONICAL_JOINTS, LEG_JOINTS
from src.servo_id.actuator_model import ServoParams
from src.servo_id.loader import AlignmentDiag, JointSamples, Segment, TestCapture
from src.servo_id.replay import ReplayModel, build_replay_model, replay_capture

STAND_POSE_15 = np.array([0.0, -0.90, 0.28] * 4 + [0.0, 0.0, 0.0])


@dataclass
class SyntheticSpec:
  joint: str = "fl_thigh"
  rate_hz: float = 100.0
  step_amps: tuple = (0.15, 0.3)
  step_hold_s: float = 0.8
  sine_amps: tuple = (0.1, 0.25)
  sine_freqs: tuple = (1.0, 3.0)
  sine_dwell_s: float = 2.5
  sine_gap_s: float = 0.3
  fw_offset_ms: float = 50_000.0  # firmware clock at host command 0
  receipt_jitter_ms: float = 1.5
  refresh_ms: float = 30.0  # per-servo position refresh period
  telem_dt_ms: float = 10.0
  noise_rad: float = 0.0  # optional measurement noise on q
  seed: int = 0


def _steps_offsets(spec: SyntheticSpec) -> tuple[np.ndarray, list[dict]]:
  hold = int(round(spec.step_hold_s * spec.rate_hz))
  offs, segs = [], []
  i = 0
  for amp in spec.step_amps:
    for target in (amp, 0.0, -amp, 0.0):
      offs.append(np.full(hold, target))
      segs.append(
        {"i0": i, "i1": i + hold, "target": float(target),
         "amplitude": float(amp)}
      )
      i += hold
  return np.concatenate(offs), segs


def _sine_offsets(spec: SyntheticSpec) -> tuple[np.ndarray, list[dict]]:
  dwell = int(round(spec.sine_dwell_s * spec.rate_hz))
  gap = int(round(spec.sine_gap_s * spec.rate_hz))
  offs, segs = [], []
  i = 0
  for amp in spec.sine_amps:
    for freq in spec.sine_freqs:
      t = np.arange(dwell) / spec.rate_hz
      offs.append(amp * np.sin(2.0 * np.pi * freq * t))
      segs.append(
        {"i0": i, "i1": i + dwell, "amplitude": float(amp),
         "amplitude_requested": float(amp), "freq_hz": float(freq)}
      )
      i += dwell
      offs.append(np.zeros(gap))
      i += gap
  return np.concatenate(offs), segs


def _synthetic_capture(
  kind: str,
  offsets: np.ndarray,
  segments: list[dict],
  spec: SyntheticSpec,
) -> tuple[TestCapture, dict]:
  """Ideal (truth-aligned) TestCapture for the forward rollout + meta."""
  n = offsets.shape[0]
  jcol = CANONICAL_JOINTS.index(spec.joint)
  cmd_pose = np.tile(STAND_POSE_15, (n, 1))
  cmd_pose[:, jcol] += offsets
  cmd_time = spec.fw_offset_ms + np.arange(n) * (1000.0 / spec.rate_hz)
  meta = {
    "label": spec.joint,
    "kind": kind,
    "joints": {
      spec.joint: {
        "kind": kind, "rate_hz": spec.rate_hz, "segments": segments,
        "params": {}, "send_on_change": False,
      }
    },
    "base_pose_rad": STAND_POSE_15.tolist(),
    "rate_hz": spec.rate_hz,
    "orientation": "synthetic",
    "payload_g": 0.0,
    "temp_c": [None] * 15,
    "vbat_start_mv": 7400,
    "vbat_end_mv": 7400,
    "overruns": 0,
  }
  capture = TestCapture(
    path=Path(f"synthetic_{kind}"),
    label=spec.joint,
    kind=kind,
    rate_hz=spec.rate_hz,
    base_pose=STAND_POSE_15.copy(),
    excited=(spec.joint,),
    cmd_time_ms=cmd_time,
    cmd_pose=cmd_pose,
    cmd_sent=np.ones(n, dtype=bool),
    samples={},  # filled after the rollout
    segments=[
      Segment(spec.joint, cmd_time[s["i0"]],
              cmd_time[min(s["i1"], n - 1)], s)
      for s in segments
    ],
    alignment=AlignmentDiag(spec.fw_offset_ms, n, 1.0, 0.0, 0.0, 1),
    gravity_body=None,
    meta=meta,
  )
  return capture, meta


def _telemetry_arrays(
  capture: TestCapture,
  traj,
  spec: SyntheticSpec,
  rng: np.random.Generator,
) -> dict[str, np.ndarray]:
  """Re-sample the sim trajectory the way firmware telemetry does."""
  n_cmd = capture.cmd_time_ms.shape[0]
  # Telemetry drain starts before the first command's receipt (the real
  # capture's first rows carry a stale settle-command cmd_ms), so command
  # 0's receipt is observable as a cmd_ms increase for the aligner.
  t_first = capture.cmd_time_ms[0] - 2.0 * spec.telem_dt_ms
  t_last = capture.cmd_time_ms[-1] + 300.0  # capture tail
  t_ms = np.arange(t_first, t_last, spec.telem_dt_ms).astype(np.int64)
  m = t_ms.shape[0]

  sample_t = np.zeros((m, 15), dtype=np.int64)
  q = np.zeros((m, 15), dtype=float)
  for j, name in enumerate(CANONICAL_JOINTS):
    phase = float(rng.uniform(0.0, spec.refresh_ms))
    refresh = t_first - phase + spec.refresh_ms * np.arange(
      int(np.ceil((t_last - t_first + spec.refresh_ms) / spec.refresh_ms))
      + 1
    )
    idx = np.searchsorted(refresh, t_ms.astype(float), side="right") - 1
    s_t = refresh[np.clip(idx, 0, None)]
    sample_t[:, j] = np.rint(s_t).astype(np.int64)
    if name in LEG_JOINTS:
      col = LEG_JOINTS.index(name)
      qj = traj.at(sample_t[:, j].astype(float), col)
    else:
      qj = np.full(m, STAND_POSE_15[j])
    if spec.noise_rad > 0.0:
      # One noise draw per DISTINCT sample, repeated between refreshes.
      uniq, inv = np.unique(sample_t[:, j], return_inverse=True)
      qj = qj + rng.normal(0.0, spec.noise_rad, uniq.shape[0])[inv]
    q[:, j] = qj
  q_age = (t_ms[:, None] - sample_t).astype(np.int32)

  receipt = capture.cmd_time_ms + rng.uniform(
    0.0, spec.receipt_jitter_ms, n_cmd
  )
  receipt = np.rint(receipt).astype(np.int64)
  receipt = np.maximum.accumulate(receipt)
  cmd_idx = np.searchsorted(receipt, t_ms, side="right") - 1
  cmd_ms = receipt[np.clip(cmd_idx, 0, None)]
  cmd_ms[cmd_idx < 0] = receipt[0] - 1000  # stale pre-test command

  dt_s = np.diff(sample_t.astype(float), axis=0) / 1000.0
  qd_fd = np.zeros_like(q)
  with np.errstate(divide="ignore", invalid="ignore"):
    qd_fd[1:] = np.where(dt_s > 0, np.diff(q, axis=0) / np.where(
      dt_s > 0, dt_s, 1.0), 0.0)

  host_t0 = 1000.0  # arbitrary host monotonic origin, seconds
  return {
    "cmd_t_host": host_t0
    + (capture.cmd_time_ms - capture.cmd_time_ms[0]) / 1000.0,
    "cmd_pose": capture.cmd_pose,
    "cmd_sent": capture.cmd_sent,
    "t_ms": t_ms,
    "sample_t_ms": sample_t,
    "q": q,
    "q_age_ms": q_age,
    "cmd_ms": cmd_ms,
    "qd_fd": qd_fd,
    "qd_meas": qd_fd.copy(),
    "imu_t_ms": t_ms[:n_cmd] if m >= n_cmd else t_ms,
    "imu_acc": np.tile([0.0, 0.0, -9.81], (n_cmd, 1)),
    "imu_gyro": np.zeros((n_cmd, 3)),
    "imu_vbat_mv": np.full(n_cmd, 7400, dtype=np.int32),
  }


def generate_synthetic_session(
  out_dir: Path,
  params_true: ServoParams,
  spec: SyntheticSpec | None = None,
  kinds: tuple[str, ...] = ("steps", "sine_grid"),
  rm: ReplayModel | None = None,
) -> Path:
  """Write a synthetic stand-regime session under ``out_dir``.

  ``params_true`` (including its ``delay_ms``) is applied to ALL 12
  servos. Returns the created session directory.
  """
  spec = spec or SyntheticSpec()
  rng = np.random.default_rng(spec.seed)
  rm = rm or build_replay_model(fixed_base=False)
  session = Path(out_dir) / "synthetic_stand"
  session.mkdir(parents=True, exist_ok=True)

  tests_meta = []
  for kind in kinds:
    if kind == "steps":
      offsets, segments = _steps_offsets(spec)
    elif kind == "sine_grid":
      offsets, segments = _sine_offsets(spec)
    else:
      raise ValueError(f"unknown synthetic kind {kind!r}")
    capture, meta = _synthetic_capture(kind, offsets, segments, spec)
    # Truth rollout: samples are not needed for the forward pass, but the
    # replay window is derived from them -> seed with the command span.
    capture.samples = {
      name: JointSamples(
        t_ms=np.array(
          [capture.cmd_time_ms[0], capture.cmd_time_ms[-1] + 250.0],
          dtype=np.int64,
        ),
        q=np.array([STAND_POSE_15[CANONICAL_JOINTS.index(name)]] * 2),
      )
      for name in LEG_JOINTS
    }
    traj = replay_capture(
      rm, capture, [params_true] * 12, params_true.delay_ms
    )
    arrays = _telemetry_arrays(capture, traj, spec, rng)
    fname = f"{spec.joint}__{kind}.npz"
    np.savez_compressed(
      session / fname, meta_json=np.array(json.dumps(meta)), **arrays
    )
    tests_meta.append(
      {
        "file": fname, "label": spec.joint, "kind": kind,
        "duration_s": round(offsets.shape[0] / spec.rate_hz, 2),
        "n_cmd": int(offsets.shape[0]),
        "n_telem": int(arrays["t_ms"].shape[0]),
        "vbat_start_mv": 7400, "vbat_end_mv": 7400,
      }
    )

  manifest = {
    "schema": "xgo-servo-id-session/v2",
    "created": datetime.now().isoformat(timespec="seconds"),
    "synthetic": True,
    "params_true": params_true.as_dict(),
    "base_pose": "stand",
    "base_pose_source": "synthetic",
    "base_pose_rad": STAND_POSE_15.tolist(),
    "rate_hz": spec.rate_hz,
    "orientation": "synthetic",
    "payload_g": 0.0,
    "planned_tests": len(tests_meta),
    "skips": [],
    "vbat_start_mv": 7400,
    "vbat_end_mv": 7400,
    "tests": tests_meta,
    "aborted": None,
  }
  (session / "manifest.json").write_text(json.dumps(manifest, indent=2))
  return session
