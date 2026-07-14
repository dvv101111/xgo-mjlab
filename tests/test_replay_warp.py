"""Warp (GPU-batched) replay backend tests. Skip cleanly without CUDA.

Two acceptance gates for src/servo_id/replay_warp.py:

1. CPU-vs-warp loss agreement on REAL captures: 3 random candidate
   vectors x 2 captures (steps + sine_grid) of the 2026-07-13 stand
   session must agree within 2% relative per (candidate, capture) item.
   The random candidates span ALL JOINT_PARAM_NAMES, so nonzero
   deadband values are exercised through both backends.
   Physics engines differ (fp32 mujoco_warp vs fp64 MuJoCo, batched
   Newton solver); measured agreement at port time was <=0.13%, the 2%
   gate leaves headroom for engine version drift.
2. Synthetic parameter recovery (same budget and tolerances as the CPU
   test in tests/test_fit_synthetic.py) through the warp backend.

Run: PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m unittest \\
       tests.test_replay_warp -v
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
  from src.servo_id.replay_warp import warp_available

  _HAS_GPU = warp_available()
except Exception:  # warp import/driver failure counts as "no GPU"
  _HAS_GPU = False

REAL_SESSION = Path(
  "/home/dvv/dev/bots/Quadruped-robot/runs/servo_id/"
  "2026-07-13_18-07-30_stand"
)
REL_TOL = 0.02  # documented CPU-vs-warp loss agreement gate


@unittest.skipUnless(_HAS_GPU, "no CUDA device for the warp backend")
class TestWarpMatchesCpuOnRealCaptures(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not REAL_SESSION.exists():
      raise unittest.SkipTest(f"real session not found: {REAL_SESSION}")
    from src.servo_id.actuator_model import JOINT_PARAM_NAMES
    from src.servo_id.fitting import Evaluator, FitConfig
    from src.servo_id.loader import load_test
    from src.servo_id.replay_warp import WarpEvaluator

    cls.captures = [
      (load_test(REAL_SESSION / "fl_hip__steps.npz"), ("fl_hip",)),
      (load_test(REAL_SESSION / "fl_hip__sine_grid.npz"), ("fl_hip",)),
    ]
    cls.fit_names = [*JOINT_PARAM_NAMES, "delay_ms"]
    cfg = FitConfig(fit_names=list(cls.fit_names),
                    fixed={"tau_max": 0.22})
    cls.cpu = Evaluator(cls.captures, cfg, target_joint="fl_hip")
    cls.warp = WarpEvaluator(cls.captures, cfg, target_joint="fl_hip",
                             max_envs=8)

  def test_losses_match_cpu_within_tolerance(self):
    from src.servo_id.replay import capture_loss, replay_capture
    from src.servo_id.replay_warp import WorkItem, evaluate_items

    rng = np.random.default_rng(3)
    zs = [rng.uniform(-1.0, 1.0, len(self.fit_names)) for _ in range(3)]

    items, joints_per_item, cpu_losses = [], [], []
    for z in zs:
      values = self.cpu.values_from_vector(z)
      params, delay = self.cpu.params_from_values(values)
      for cap, joints in self.captures:
        traj = replay_capture(self.cpu.rm, cap, params, delay)
        cpu_losses.append(capture_loss(traj, cap, joints))
        items.append(WorkItem(cap, params, delay))
        joints_per_item.append(joints)

    warp_losses = evaluate_items(self.warp.batch, items, joints_per_item)
    for i, (lc, lw) in enumerate(zip(cpu_losses, warp_losses)):
      rel = abs(lw - lc) / lc
      self.assertLess(
        rel, REL_TOL,
        msg=f"item {i}: cpu {lc:.6e} warp {lw:.6e} rel {rel:.4f}",
      )

  def test_eval_population_matches_eval_values(self):
    # The batched population path and the single-candidate path must be
    # the same computation (chunking/idle-world handling must not leak).
    rng = np.random.default_rng(7)
    zs = [rng.uniform(-1.0, 1.0, len(self.fit_names)) for _ in range(2)]
    pop = self.warp.eval_population(zs)
    singles = [self.warp.eval_values(self.warp.values_from_vector(z))
               for z in zs]
    np.testing.assert_allclose(pop, singles, rtol=1e-3)


@unittest.skipUnless(_HAS_GPU, "no CUDA device for the warp backend")
class TestWarpSyntheticRecovery(unittest.TestCase):
  """tests/test_fit_synthetic.py rerun through the warp backend."""

  @classmethod
  def setUpClass(cls):
    from src.servo_id.fitting import FitConfig, run_cma_fit  # noqa: F401
    from src.servo_id.loader import load_session
    from src.servo_id.replay_warp import WarpEvaluator
    from src.servo_id.synthetic import SyntheticSpec, generate_synthetic_session
    from tests.test_fit_synthetic import FIT_NAMES, PINNED, TRUTH

    cls.tmp = tempfile.TemporaryDirectory()
    spec = SyntheticSpec(
      step_amps=(0.3,), sine_amps=(0.25,), sine_freqs=(1.0, 3.0), seed=11
    )
    session_dir = generate_synthetic_session(Path(cls.tmp.name), TRUTH,
                                             spec)
    sess = load_session(session_dir)
    captures = [(cap, ("fl_thigh",)) for cap in sess.tests]
    cfg = FitConfig(fit_names=list(FIT_NAMES), fixed=dict(PINNED))
    cls.evaluator = WarpEvaluator(captures, cfg, target_joint=None,
                                  max_envs=64)

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()

  def test_cma_recovers_parameters_warp(self):
    from src.servo_id.fitting import run_cma_fit
    from tests.test_fit_synthetic import (
      DEADBAND_TOL,
      FIT_NAMES,
      GENERATIONS,
      POPSIZE,
      TRUTH,
    )

    res = run_cma_fit(
      self.evaluator, generations=GENERATIONS, popsize=POPSIZE, seed=0,
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
      "\nwarp synthetic recovery: "
      + ", ".join(
        f"{k} {v[k]:.4f} (truth {getattr(TRUTH, k):.4f})"
        for k in FIT_NAMES
      )
      + f"; loss {res.loss:.2e} (baseline {res.baseline_loss:.2e})"
    )


if __name__ == "__main__":
  unittest.main()
