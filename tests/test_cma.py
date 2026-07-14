"""Sanity checks for the vendored minimal CMA-ES.

Run: PYTHONPATH=. .venv/bin/python -m unittest tests.test_cma -v
"""

import unittest

import numpy as np

from src.servo_id.cma import CmaEs


class TestCmaEs(unittest.TestCase):
  def test_sphere(self):
    target = np.array([0.3, -0.5, 0.1, 0.7])
    es = CmaEs(x0=np.zeros(4), sigma0=0.5, popsize=16, seed=3)
    for _ in range(60):
      x = es.ask()
      f = np.sum((x - target) ** 2, axis=1)
      es.tell(x, f)
    self.assertLess(float(np.max(np.abs(es.mean - target))), 1e-3)

  def test_rosenbrock_2d(self):
    # Optimum (1, 1) sits on the box corner region; scaled into bounds.
    es = CmaEs(x0=np.zeros(2), sigma0=0.5, popsize=24, seed=0)
    for _ in range(150):
      x = es.ask()
      a, b = x[:, 0], x[:, 1]
      f = (1.0 - a) ** 2 + 100.0 * (b - a**2) ** 2
      es.tell(x, f)
    self.assertLess(float(np.max(np.abs(es.mean - 1.0))), 5e-2)

  def test_bounds_respected(self):
    es = CmaEs(x0=np.zeros(3), sigma0=1.5, popsize=32, seed=1)
    x = es.ask()
    self.assertTrue(np.all(x >= -1.0) and np.all(x <= 1.0))


if __name__ == "__main__":
  unittest.main()
