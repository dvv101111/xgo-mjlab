"""Acceptance test: parameter recovery from a synthetic capture.

Generates a session from KNOWN servo parameters (including a NONZERO
deadband) through the same replay backend + telemetry re-sampling (dedupe
pattern, receipt jitter), then runs the CMA-ES fitter on 4 free
parameters (kp, damping, frictionloss, deadband — the rest pinned at
truth) with a CI-sized budget and asserts recovery. This proves the whole
chain inverts: loader alignment/dedupe -> ZOH replay -> loss ->
optimizer. The full-budget real fit uses 150 gens x 64 pop over all 9
parameters (documented in scripts/fit_servo_model.py).

Also holds the focused unit test for the deadband torque law itself.

Run: PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m unittest \\
       tests.test_fit_synthetic -v
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.servo_id.actuator_model import ServoParams
from src.servo_id.fitting import Evaluator, FitConfig, run_cma_fit
from src.servo_id.loader import load_session
from src.servo_id.synthetic import SyntheticSpec, generate_synthetic_session

TRUTH = ServoParams(
  kp=8.0, kd=0.12, damping=0.12, armature=0.002,
  frictionloss=0.02, qd_knee=6.0, qd_max=11.0, deadband=0.02,
  delay_ms=50.0,
)
FIT_NAMES = ["kp", "damping", "frictionloss", "deadband"]
PINNED = {
  "tau_max": 0.22, "kd": TRUTH.kd, "armature": TRUTH.armature,
  "qd_knee": TRUTH.qd_knee, "qd_max": TRUTH.qd_max,
  "delay_ms": TRUTH.delay_ms,
}
DEADBAND_TOL = 0.01  # rad, loose gate for the CI-sized CMA budget
GENERATIONS = 24
POPSIZE = 32


class TestDeadbandTorqueLaw(unittest.TestCase):
  """pd_clamped_torque: zero inside the band, shifted-linear outside."""

  KP = np.full(5, 10.0)
  KD = np.zeros(5)  # isolate the position-error path
  DB = np.full(5, 0.02)
  QD = np.zeros(5)
  TAU_MAX = 10.0  # large: keep the clamp inactive for this test
  QD_KNEE = np.full(5, 6.0)
  QD_MAX = np.full(5, 11.0)

  def _tau(self, err: np.ndarray) -> np.ndarray:
    from src.servo_id.actuator_model import pd_clamped_torque

    return pd_clamped_torque(
      np.zeros_like(err), self.QD, err, self.KP, self.KD, self.DB,
      self.TAU_MAX, self.QD_KNEE, self.QD_MAX,
    )

  def test_zero_torque_inside_band(self):
    err = np.array([0.0, 0.005, -0.005, 0.02, -0.02])
    np.testing.assert_array_equal(self._tau(err), np.zeros(5))

  def test_shifted_linear_outside_band(self):
    err = np.array([0.03, -0.03, 0.05, -0.05, 0.1])
    expected = self.KP * np.sign(err) * (np.abs(err) - self.DB)
    np.testing.assert_allclose(self._tau(err), expected, rtol=0, atol=1e-12)

  def test_zero_deadband_recovers_plain_pd(self):
    err = np.array([0.03, -0.03, 0.0, 0.05, -0.1])
    from src.servo_id.actuator_model import pd_clamped_torque

    tau = pd_clamped_torque(
      np.zeros_like(err), self.QD, err, self.KP, self.KD,
      np.zeros(5), self.TAU_MAX, self.QD_KNEE, self.QD_MAX,
    )
    np.testing.assert_allclose(tau, self.KP * err, rtol=0, atol=1e-12)


class TestSyntheticRecovery(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.tmp = tempfile.TemporaryDirectory()
    spec = SyntheticSpec(
      step_amps=(0.3,), sine_amps=(0.25,), sine_freqs=(1.0, 3.0), seed=11
    )
    session_dir = generate_synthetic_session(Path(cls.tmp.name), TRUTH,
                                             spec)
    sess = load_session(session_dir)
    captures = [(cap, ("fl_thigh",)) for cap in sess.tests]
    cfg = FitConfig(fit_names=list(FIT_NAMES), fixed=dict(PINNED))
    # Shared mode: truth params were applied to ALL 12 servos during
    # generation, and in the loaded regime the held joints' parameters
    # couple into the swept joint through the free base (per-joint mode
    # would score the candidate against a body held by default servos).
    cls.evaluator = Evaluator(captures, cfg, target_joint=None)

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()

  def test_truth_params_score_near_zero(self):
    # Plumbing check: evaluating the truth values must land far below
    # the current-stack baseline (residual = telemetry re-sampling and
    # receipt-jitter noise only).
    truth_loss = self.evaluator.eval_values(
      {"kp": TRUTH.kp, "damping": TRUTH.damping,
       "frictionloss": TRUTH.frictionloss, "deadband": TRUTH.deadband}
    )
    baseline = self.evaluator.eval_values({})
    self.assertLess(truth_loss, 2e-5)
    self.assertLess(truth_loss, baseline / 20.0)

  def test_cma_recovers_parameters(self):
    res = run_cma_fit(
      self.evaluator,
      generations=GENERATIONS,
      popsize=POPSIZE,
      seed=0,
      workers=min(POPSIZE, os.cpu_count() or 2),
      verbose=False,
    )
    self.assertLess(res.loss, res.baseline_loss / 10.0)
    v = res.values
    self.assertLess(abs(v["kp"] - TRUTH.kp), 0.20 * TRUTH.kp,
                    msg=f"kp {v['kp']:.3f} vs truth {TRUTH.kp}")
    self.assertLess(abs(v["damping"] - TRUTH.damping),
                    0.30 * TRUTH.damping,
                    msg=f"damping {v['damping']:.4f} vs {TRUTH.damping}")
    self.assertLess(abs(v["frictionloss"] - TRUTH.frictionloss), 0.015,
                    msg=f"frictionloss {v['frictionloss']:.4f} "
                        f"vs {TRUTH.frictionloss}")
    self.assertLess(abs(v["deadband"] - TRUTH.deadband), DEADBAND_TOL,
                    msg=f"deadband {v['deadband']:.4f} "
                        f"vs {TRUTH.deadband}")
    print(
      "\nsynthetic recovery: "
      + ", ".join(
        f"{k} {v[k]:.4f} (truth "
        f"{getattr(TRUTH, k):.4f})" for k in FIT_NAMES
      )
      + f"; loss {res.loss:.2e} (baseline {res.baseline_loss:.2e})"
    )


if __name__ == "__main__":
  unittest.main()
