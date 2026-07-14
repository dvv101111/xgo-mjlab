"""Loader unit tests: dedupe, clock alignment, segment slicing, sessions.

Run: PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m unittest \\
       tests.test_loader -v
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.servo_id.actuator_model import ServoParams
from src.servo_id.loader import (
  align_commands,
  dedupe_joint_samples,
  load_session,
)
from src.servo_id.synthetic import (
  SyntheticSpec,
  generate_synthetic_session,
)


class TestDedupe(unittest.TestCase):
  def test_repeats_collapse(self):
    # Telemetry pushes every 10 ms; the servo refreshed every 30 ms, so
    # sample times (and values) repeat between refreshes.
    sample_t = np.array([100, 100, 100, 130, 130, 130, 160, 160])
    q = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
    age = np.zeros(8, dtype=np.int32)
    s = dedupe_joint_samples(sample_t, q, age)
    np.testing.assert_array_equal(s.t_ms, [100, 130, 160])
    np.testing.assert_array_equal(s.q, [1.0, 2.0, 3.0])

  def test_nan_and_unknown_age_dropped(self):
    sample_t = np.array([100, 130, 160, 190])
    q = np.array([1.0, np.nan, 3.0, 4.0])
    age = np.array([0, 0, 255, 0], dtype=np.int32)
    s = dedupe_joint_samples(sample_t, q, age)
    np.testing.assert_array_equal(s.t_ms, [100, 190])
    np.testing.assert_array_equal(s.q, [1.0, 4.0])

  def test_out_of_order_sorted(self):
    sample_t = np.array([130, 100, 160])
    q = np.array([2.0, 1.0, 3.0])
    s = dedupe_joint_samples(sample_t, q)
    np.testing.assert_array_equal(s.t_ms, [100, 130, 160])
    np.testing.assert_array_equal(s.q, [1.0, 2.0, 3.0])


class TestAlignment(unittest.TestCase):
  def _make_stream(self, n_cmd=400, rate=100.0, offset=54321.0, seed=0,
                   jitter=2.0):
    """Host command grid + telemetry cmd_ms with known offset/jitter."""
    rng = np.random.default_rng(seed)
    host_t = 500.0 + np.arange(n_cmd) / rate  # seconds
    receipt = offset + (host_t - host_t[0]) * 1000.0 + rng.uniform(
      0.0, jitter, n_cmd
    )
    receipt = np.maximum.accumulate(np.rint(receipt)).astype(np.int64)
    # telemetry rows every 10 ms carrying the latest receipt; first row
    # predates the stream (stale settle command).
    t_rows = offset - 5.0 + 10.0 * np.arange(n_cmd + 20)
    idx = np.searchsorted(receipt, t_rows, side="right") - 1
    cmd_ms = np.where(idx >= 0, receipt[np.clip(idx, 0, None)],
                      int(offset) - 1500)
    return host_t, cmd_ms.astype(np.int64), receipt

  def test_offset_recovery(self):
    host_t, cmd_ms, receipt = self._make_stream()
    cmd_time, diag = align_commands(cmd_ms, host_t, 100.0)
    # Reconstructed command times must match true receipts to within the
    # injected jitter (constant-offset model absorbs the median).
    err = cmd_time - receipt
    self.assertLess(float(np.max(np.abs(err - np.median(err)))), 3.0)
    self.assertLess(abs(diag.slope - 1.0), 1e-4)
    self.assertGreater(diag.n_events, 300)

  def test_coalesced_commands(self):
    # Two commands arriving within one telemetry period must not derail
    # the index assignment (cumulative round, no error accumulation).
    host_t, cmd_ms, receipt = self._make_stream(jitter=6.0, seed=2)
    cmd_time, diag = align_commands(cmd_ms, host_t, 100.0)
    err = cmd_time - receipt
    self.assertLess(float(np.max(np.abs(err - np.median(err)))), 8.0)


class TestSyntheticSession(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.tmp = tempfile.TemporaryDirectory()
    truth = ServoParams(
      kp=8.0, kd=0.12, damping=0.10, armature=0.002,
      frictionloss=0.02, qd_knee=6.0, qd_max=11.0, deadband=0.02,
      delay_ms=50.0,
    )
    cls.spec = SyntheticSpec(
      step_amps=(0.3,), sine_amps=(0.25,), sine_freqs=(1.0, 3.0), seed=7
    )
    cls.session_dir = generate_synthetic_session(
      Path(cls.tmp.name), truth, cls.spec
    )

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()

  def test_session_loads(self):
    sess = load_session(self.session_dir)
    self.assertTrue(sess.loaded)  # stand regime
    self.assertEqual(len(sess.tests), 2)
    kinds = {t.kind for t in sess.tests}
    self.assertEqual(kinds, {"steps", "sine_grid"})

  def test_segments_sliced(self):
    sess = load_session(self.session_dir)
    sine = next(t for t in sess.tests if t.kind == "sine_grid")
    # 1 amp x 2 freqs
    self.assertEqual(len(sine.segments), 2)
    for seg in sine.segments:
      self.assertEqual(seg.joint, "fl_thigh")
      self.assertGreater(seg.t1_ms, seg.t0_ms)
      self.assertIn("freq_hz", seg.desc)
    # segment windows lie inside the command window
    self.assertGreaterEqual(sine.segments[0].t0_ms, sine.cmd_time_ms[0])
    self.assertLessEqual(sine.segments[-1].t1_ms, sine.cmd_time_ms[-1])

  def test_dedupe_applied(self):
    sess = load_session(self.session_dir)
    sine = next(t for t in sess.tests if t.kind == "sine_grid")
    s = sine.samples["fl_thigh"]
    # ~30 ms refresh vs 10 ms telemetry -> roughly 1/3 of rows survive.
    n_rows = sine.meta["vbat_start_mv"] and s.t_ms.size  # deduped count
    self.assertGreater(n_rows, 50)
    self.assertTrue(np.all(np.diff(s.t_ms) > 0))
    dt = np.diff(s.t_ms)
    self.assertAlmostEqual(float(np.median(dt)), 30.0, delta=1.0)

  def test_alignment_close_to_truth(self):
    sess = load_session(self.session_dir)
    for t in sess.tests:
      # Truth firmware offset is fw_offset_ms; recovered offset must be
      # within the injected receipt jitter.
      self.assertLess(
        abs(t.alignment.offset_ms - self.spec.fw_offset_ms),
        self.spec.receipt_jitter_ms + 1.0,
      )
      self.assertLess(t.alignment.jitter_std_ms, 2.0)

  def test_kind_filter(self):
    sess = load_session(self.session_dir, kinds=("steps",))
    self.assertEqual([t.kind for t in sess.tests], ["steps"])


if __name__ == "__main__":
  unittest.main()
