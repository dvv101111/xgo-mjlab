"""Capture-session loading for servo system ID.

Input format: ``xgo-servo-id-session/v2`` directories written by the parent
repo's ``tools/servo_id_capture.py`` — a ``manifest.json`` plus one npz per
(joint, test). v1 sessions were captured on the v2.3 firmware (broken
per-servo refresh cadence, dropped commands) and are invalid by design:
they are rejected, never migrated. Everything this module returns lives on
the FIRMWARE clock (ms), the only clock shared by commands and telemetry:

* Telemetry frames are fresh-sweep-gated on the v2.4 firmware: one frame
  per completed coherent full-15 read sweep, ~75-80 Hz under load, all 15
  servos sharing one sweep timestamp window. ``sample_t_ms[:, j]`` is the
  firmware time the value in ``q[:, j]`` was actually sampled
  (``t_ms - pos_age``); a servo that failed a sweep keeps its stale value,
  so per-joint streams are still DEDUPED on ``sample_t_ms``
  (see :func:`dedupe_joint_samples`).
* Commands are stamped on the HOST clock (``cmd_t_host``, monotonic s).
  Each telemetry row carries ``cmd_ms``, the firmware receipt time of the
  most recently received command. :func:`align_commands` reconstructs the
  constant host->firmware offset from those receipt events (the two clocks
  tick at the same rate to ~1e-6 over a <60 s capture; verified slope
  9.999993 ms per 10 ms host tick on the dev fixture) and places every
  command on the firmware clock. Residual receipt jitter (measured std
  ~0.8 ms, absmax ~5 ms on the fixture) is reported, not modeled; it is DR
  territory. The fitted global delay is therefore "firmware receipt ->
  visible actuation", the same convention as the measured 54 ms median
  onset latency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.servo_id import CANONICAL_JOINTS

GRAVITY = 9.81
# IMU accel convention (verified on the stand dev fixture: mean acc =
# (0.01, 0.36, -9.78) with the robot upright): the accelerometer reports
# the gravity VECTOR in the body frame, i.e. -z when upright. A fixed-base
# replay must use gravity = GRAVITY * unit(mean acc).
IMU_ACC_MIN_NORM = 8.0
IMU_ACC_MAX_NORM = 11.5


class CaptureFormatError(RuntimeError):
  """The npz / manifest does not match the expected capture contract."""


@dataclass
class JointSamples:
  """Deduped telemetry for one canonical joint (firmware clock)."""

  t_ms: np.ndarray  # (M,) int64, strictly increasing sample times
  q: np.ndarray  # (M,) float64 rad


@dataclass
class AlignmentDiag:
  """Host->firmware command-clock alignment diagnostics."""

  offset_ms: float  # firmware = host_ms + offset
  n_events: int  # distinct cmd_ms receipt events used
  slope: float  # fw ms per host ms (should be ~1.0)
  jitter_std_ms: float  # residual receipt jitter around the offset
  jitter_absmax_ms: float
  max_index_gap: int  # largest command-index gap between events


@dataclass
class Segment:
  """One excitation segment (from meta_json), on the firmware clock."""

  joint: str
  t0_ms: float
  t1_ms: float
  desc: dict  # raw segment dict (amplitude / freq_hz / target / ...)


@dataclass
class TestCapture:
  """One (joint, test) npz, fully aligned to the firmware clock."""

  path: Path
  label: str
  kind: str
  rate_hz: float
  base_pose: np.ndarray  # (15,) rad
  excited: tuple[str, ...]  # canonical names of excited joints
  cmd_time_ms: np.ndarray  # (n_cmd,) float64, fw clock, uniform grid
  cmd_pose: np.ndarray  # (n_cmd, 15) rad, absolute targets
  cmd_sent: np.ndarray  # (n_cmd,) bool
  samples: dict[str, JointSamples]  # canonical joint -> deduped stream
  segments: list[Segment]
  alignment: AlignmentDiag
  gravity_body: np.ndarray | None  # (3,) m/s^2 in the base frame
  meta: dict = field(repr=False, default_factory=dict)
  # Replay regime (2026-07-15), stamped from the session manifest by
  # load_session: fixed_base selects the bench replay world (welded base,
  # no floor, gravity from the IMU); stand sessions keep the free base.
  fixed_base: bool = False

  @property
  def temp_c(self) -> list | None:
    """Per-servo temperature (deg C, canonical order, None per failed
    read) taken right before the test, or None for captures without it."""
    return self.meta.get("temp_c")


@dataclass
class Session:
  """One capture-session directory."""

  path: Path
  manifest: dict
  base_pose_name: str  # "stand" (loaded), "folded"/"midrange" (free legs)
  orientation: str  # free-form physical-setup label from the capture CLI
  payload_g: float  # weighed mass attached to the feet, grams
  tests: list[TestCapture]

  @property
  def loaded(self) -> bool:
    """True when the robot stood on the floor (ground-contact regime)."""
    return self.base_pose_name not in ("folded", "midrange")


def dedupe_joint_samples(
  sample_t_ms: np.ndarray,
  q: np.ndarray,
  q_age_ms: np.ndarray | None = None,
) -> JointSamples:
  """Dedupe one joint's telemetry column on its firmware sample times.

  Positions repeat between servo refreshes (~20-34 ms); keep exactly one
  row per distinct ``sample_t_ms``, dropping NaN positions and rows whose
  age reads as unknown (255 sentinel in the capture tool).
  """
  t = np.asarray(sample_t_ms, dtype=np.int64)
  qv = np.asarray(q, dtype=float)
  ok = np.isfinite(qv)
  if q_age_ms is not None:
    ok &= np.asarray(q_age_ms) < 255
  t, qv = t[ok], qv[ok]
  if t.size == 0:
    return JointSamples(t_ms=t, q=qv)
  order = np.argsort(t, kind="stable")
  t, qv = t[order], qv[order]
  keep = np.concatenate(([True], np.diff(t) > 0))
  return JointSamples(t_ms=t[keep], q=qv[keep])


def align_commands(
  cmd_ms: np.ndarray,
  cmd_t_host: np.ndarray,
  rate_hz: float,
) -> tuple[np.ndarray, AlignmentDiag]:
  """Place the host command stream on the firmware clock.

  ``cmd_ms[i]`` is the firmware receipt time of the most recent command at
  telemetry row i. Receipt events (distinct increasing values, minus the
  stale pre-test entry) land on the host's uniform command grid; the event
  -> command-index assignment is the cumulative round against the first
  event, which does not accumulate error. The single constant offset is
  the median residual; everything else is jitter (reported).

  Returns (cmd_time_ms, diag) with cmd_time_ms[k] = offset + host_ms[k].

  Convention note: the first observed receipt event is assumed to be
  command 0 (the capture tool's first telemetry rows carry a stale
  settle-command cmd_ms, so command 0's receipt shows up as an increase).
  If a capture ever starts draining late, the assignment shifts by one
  command (~10 ms) — a constant absorbed by the fitted global delay.
  """
  cmd_ms = np.asarray(cmd_ms, dtype=np.int64)
  host_ms = (np.asarray(cmd_t_host, dtype=float)
             - float(cmd_t_host[0])) * 1000.0
  n_cmd = host_ms.shape[0]
  dt_nom = 1000.0 / rate_hz

  # Receipt events: strictly-increasing changes within the stream. The
  # first row's cmd_ms may predate the capture (settle command) and is
  # excluded by taking only values where cmd_ms increases.
  inc = np.concatenate(([False], np.diff(cmd_ms) > 0))
  events = np.unique(cmd_ms[inc])
  if events.size < max(4, n_cmd // 50):
    raise CaptureFormatError(
      f"too few command receipt events ({events.size}) to align clocks"
    )
  k = np.round((events - events[0]) / dt_nom).astype(int)
  k = np.clip(k, 0, n_cmd - 1)
  slope = float(np.polyfit(k, events.astype(float), 1)[0]) / dt_nom
  resid = events - host_ms[k]
  offset = float(np.median(resid))
  jitter = resid - offset
  diag = AlignmentDiag(
    offset_ms=offset,
    n_events=int(events.size),
    slope=slope,
    jitter_std_ms=float(np.std(jitter)),
    jitter_absmax_ms=float(np.max(np.abs(jitter))),
    max_index_gap=int(np.max(np.diff(k))) if k.size > 1 else 0,
  )
  if abs(slope - 1.0) > 5e-4:
    raise CaptureFormatError(
      f"command clock slope {slope:.6f} deviates from 1.0 — clocks are "
      "not tick-compatible; refusing to align"
    )
  return offset + host_ms, diag


def gravity_from_imu(imu_acc: np.ndarray) -> np.ndarray | None:
  """Mean-accelerometer gravity vector in the base frame (see module doc).

  Returns None when the reading is implausible (norm far from g, e.g. the
  robot was moving hard or the IMU misbehaved).
  """
  acc = np.asarray(imu_acc, dtype=float)
  if acc.ndim != 2 or acc.shape[0] < 10:
    return None
  mean = np.nanmean(acc, axis=0)
  norm = float(np.linalg.norm(mean))
  if not (IMU_ACC_MIN_NORM <= norm <= IMU_ACC_MAX_NORM):
    return None
  return GRAVITY * mean / norm


def _segments_from_meta(
  meta: dict, cmd_time_ms: np.ndarray
) -> list[Segment]:
  """Segment windows (fw clock) from the meta_json joints dict.

  ``i0``/``i1`` index the command grid (i1 exclusive); the window end is
  clamped to the last command of the grid.
  """
  n = cmd_time_ms.shape[0]
  segs: list[Segment] = []
  for joint, jmeta in meta.get("joints", {}).items():
    for s in jmeta.get("segments", []):
      i0 = int(s["i0"])
      i1 = min(int(s["i1"]), n - 1)
      if i0 >= n or i1 <= i0:
        continue
      segs.append(
        Segment(
          joint=joint,
          t0_ms=float(cmd_time_ms[i0]),
          t1_ms=float(cmd_time_ms[i1]),
          desc={k: v for k, v in s.items() if k not in ("i0", "i1")},
        )
      )
  return segs


def load_test(path: Path) -> TestCapture:
  """Load one capture npz and align everything to the firmware clock."""
  path = Path(path)
  with np.load(path, allow_pickle=False) as d:
    required = ("meta_json", "cmd_t_host", "cmd_pose", "cmd_sent", "t_ms",
                "sample_t_ms", "q", "q_age_ms", "cmd_ms", "qd_meas")
    missing = [k for k in required if k not in d.files]
    if missing:
      raise CaptureFormatError(f"{path.name}: missing npz keys {missing}")
    meta = json.loads(str(d["meta_json"]))
    q = d["q"]
    if q.ndim != 2 or q.shape[1] != len(CANONICAL_JOINTS):
      raise CaptureFormatError(
        f"{path.name}: q has shape {q.shape}, expected (N, 15)"
      )
    rate_hz = float(meta["rate_hz"])
    cmd_time_ms, diag = align_commands(d["cmd_ms"], d["cmd_t_host"], rate_hz)
    samples = {
      name: dedupe_joint_samples(
        d["sample_t_ms"][:, j], q[:, j], d["q_age_ms"][:, j]
      )
      for j, name in enumerate(CANONICAL_JOINTS)
    }
    gravity = (
      gravity_from_imu(d["imu_acc"]) if "imu_acc" in d.files else None
    )
    excited = tuple(meta.get("joints", {}).keys())
    return TestCapture(
      path=path,
      label=str(meta.get("label", path.stem)),
      kind=str(meta.get("kind", "unknown")),
      rate_hz=rate_hz,
      base_pose=np.asarray(meta["base_pose_rad"], dtype=float),
      excited=excited,
      cmd_time_ms=cmd_time_ms,
      cmd_pose=np.asarray(d["cmd_pose"], dtype=float),
      cmd_sent=np.asarray(d["cmd_sent"], dtype=bool),
      samples=samples,
      segments=_segments_from_meta(meta, cmd_time_ms),
      alignment=diag,
      gravity_body=gravity,
      meta=meta,
    )


def load_session(
  session_dir: Path,
  kinds: tuple[str, ...] | None = None,
  joints: tuple[str, ...] | None = None,
) -> Session:
  """Load a session directory (manifest + the test npz files it lists).

  ``kinds`` / ``joints`` optionally filter tests by excitation kind and by
  excited-joint label. Missing npz files (aborted sessions) are skipped
  with a warning on stderr.
  """
  import sys

  session_dir = Path(session_dir)
  manifest_path = session_dir / "manifest.json"
  if not manifest_path.exists():
    raise CaptureFormatError(f"{session_dir}: no manifest.json")
  manifest = json.loads(manifest_path.read_text())
  if manifest.get("schema") != "xgo-servo-id-session/v2":
    raise CaptureFormatError(
      f"{session_dir}: schema {manifest.get('schema')!r} is not "
      "xgo-servo-id-session/v2 (v1 sessions are v2.3-firmware-poisoned "
      "and invalid by design; recapture on v2.4)"
    )
  tests: list[TestCapture] = []
  for rec in manifest.get("tests", []):
    if rec["kind"] == "decay":
      # Torque-off pendulum captures carry passive telemetry only (no
      # command stream to clock-align); they are analyzed by dedicated
      # tooling, not the fit loader.
      continue
    if kinds and rec["kind"] not in kinds:
      continue
    if joints and rec["label"] not in joints and rec["label"] not in (
      "all_legs", "multi"
    ):
      continue
    npz = session_dir / rec["file"]
    if not npz.exists():
      print(f"WARNING: {npz} listed in manifest but missing; skipped",
            file=sys.stderr)
      continue
    tests.append(load_test(npz))
  sess = Session(
    path=session_dir,
    manifest=manifest,
    base_pose_name=str(manifest.get("base_pose", "unknown")),
    orientation=str(manifest["orientation"]),
    payload_g=float(manifest["payload_g"]),
    tests=tests,
  )
  # Stamp the replay regime from the manifest: bench sessions
  # (folded/midrange base pose, robot fixtured, legs free) replay
  # fixed-base with IMU gravity; stand sessions keep the free base +
  # floor world.
  for cap in sess.tests:
    cap.fixed_base = not sess.loaded
  return sess
