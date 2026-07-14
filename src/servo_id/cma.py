"""Minimal dependency-free CMA-ES (ask/tell), for box-bounded problems.

Standard (mu/mu_w, lambda) CMA-ES per Hansen's tutorial (arXiv 1604.00772):
cumulative step-size adaptation + rank-1/rank-mu covariance update. The
search space is the normalized parameter cube; sampled points are clipped
to the box and the update uses the clipped points, which is adequate here
(PACE uses the same box + sigma0 0.5 + population-spread early stop).

Not vendored from PACE (their optimizer wraps the ``cmaes`` pip package,
which is not in this venv); validated in tests/test_cma.py on sphere and
Rosenbrock.
"""

from __future__ import annotations

import numpy as np


class CmaEs:
  """Ask/tell CMA-ES minimizer on [-1, 1]^n (bounds optional)."""

  def __init__(
    self,
    x0: np.ndarray,
    sigma0: float = 0.5,
    popsize: int | None = None,
    bounds: tuple[float, float] | None = (-1.0, 1.0),
    seed: int = 0,
  ):
    self.n = int(np.asarray(x0).size)
    self.mean = np.asarray(x0, dtype=float).copy()
    self.sigma = float(sigma0)
    self.bounds = bounds
    self.rng = np.random.default_rng(seed)

    n = self.n
    self.popsize = popsize or 4 + int(3 * np.log(n))
    self.mu = self.popsize // 2
    w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
    self.weights = w / w.sum()
    self.mu_eff = 1.0 / np.sum(self.weights**2)

    self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
    self.d_sigma = (
      1.0
      + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0)
      + self.c_sigma
    )
    self.c_c = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
    self.c_1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
    self.c_mu = min(
      1.0 - self.c_1,
      2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
      / ((n + 2.0) ** 2 + self.mu_eff),
    )
    self.chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))

    self.p_sigma = np.zeros(n)
    self.p_c = np.zeros(n)
    self.cov = np.eye(n)
    self._decompose()
    self.generation = 0

  def _decompose(self) -> None:
    self.cov = (self.cov + self.cov.T) / 2.0
    eigval, eigvec = np.linalg.eigh(self.cov)
    eigval = np.maximum(eigval, 1e-20)
    self._B = eigvec
    self._D = np.sqrt(eigval)
    self._inv_sqrt = eigvec @ np.diag(1.0 / self._D) @ eigvec.T

  def ask(self) -> np.ndarray:
    """Sample a (popsize, n) population, clipped to the box."""
    z = self.rng.standard_normal((self.popsize, self.n))
    x = self.mean + self.sigma * (z * self._D) @ self._B.T
    if self.bounds is not None:
      x = np.clip(x, self.bounds[0], self.bounds[1])
    return x

  def tell(self, x: np.ndarray, fitness: np.ndarray) -> None:
    """Rank by fitness (lower = better) and update the distribution."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(fitness)
    x_mu = x[order[: self.mu]]
    y_mu = (x_mu - self.mean) / self.sigma
    y_w = self.weights @ y_mu

    self.mean = self.mean + self.sigma * y_w

    self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + np.sqrt(
      self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff
    ) * (self._inv_sqrt @ y_w)
    norm_ps = np.linalg.norm(self.p_sigma)
    self.sigma *= float(
      np.exp((self.c_sigma / self.d_sigma) * (norm_ps / self.chi_n - 1.0))
    )

    h_sigma = float(
      norm_ps
      / np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
      < (1.4 + 2.0 / (self.n + 1.0)) * self.chi_n
    )
    self.p_c = (1.0 - self.c_c) * self.p_c + h_sigma * np.sqrt(
      self.c_c * (2.0 - self.c_c) * self.mu_eff
    ) * y_w

    delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
    rank_mu = (y_mu * self.weights[:, None]).T @ y_mu
    self.cov = (
      (1.0 - self.c_1 - self.c_mu) * self.cov
      + self.c_1 * (np.outer(self.p_c, self.p_c) + delta_h * self.cov)
      + self.c_mu * rank_mu
    )
    self._decompose()
    self.generation += 1
