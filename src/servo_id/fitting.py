"""CMA-ES fitting orchestration (PACE-style, multiprocess population).

Two population backends, selected by the evaluator object passed to
:func:`run_cma_fit`:

  cpu   ``Evaluator`` — members evaluated in parallel worker processes
        (fork), each owning its own MjData; the compiled model is
        inherited from the parent. The model is tiny (18 DoF) at
        ~25 us/step, so a population evaluation is CPU-bound and
        embarrassingly parallel.
  warp  ``replay_warp.WarpEvaluator`` — exposes ``eval_population``, so a
        whole generation (popsize x captures work items) is one batched
        mujoco_warp rollout on the GPU; no fork pool is created.

Modes (block-coordinate scheme since fit v2):
  shared     pass 1: one CMA-ES run; a single parameter set applied to
             all 12 servos (same servo model everywhere, ToddlerBot
             consistency assumption) + the global delay. Scored on every
             capture.
  per-joint  pass 2: one CMA-ES run per excited joint over that joint's
             captures; the other 11 joints hold the base pose under the
             pass-1 shared-fit values (``held_values``). Fits the joint's
             8 model params + a delay; the cross-joint median delay is
             reported as the global delay.
             Fit v1 ran per-joint with the other joints under DEFAULT
             (current stack) parameters — through the free base the held
             joints' true-vs-default error couples into the swept joint's
             response, so each fit absorbed a bias that broke the
             all-fitted deployment world (validated 5x worse than the
             shared fit). Per-joint therefore now REQUIRES held values
             from a shared fit; there is no DEFAULT-held fallback.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field

import numpy as np

from src.servo_id import LEG_JOINTS
from src.servo_id.actuator_model import (
  DEFAULT_PARAMS,
  SERVO_PARAM_NAMES,
  ServoParams,
  denormalize,
)
from src.servo_id.cma import CmaEs
from src.servo_id.loader import TestCapture
from src.servo_id.replay import (
  LOSS_SKIP_MS,
  SETTLE_S,
  SIM_DT,
  ReplayModel,
  build_replay_model,
  capture_loss,
  replay_capture,
)

EARLY_STOP_SPREAD = 1e-2  # relative population score spread (PACE epsilon)
EARLY_STOP_MIN_GENS = 10
# Plateau stop: v1 traces reached within 1% of their final loss by gen
# 17-70 (median ~40) yet ran the full 150-generation budget.
PLATEAU_GENS = 30
PLATEAU_RTOL = 0.01


@dataclass
class FitConfig:
  fit_names: list[str]  # subset of SERVO_PARAM_NAMES (+ "delay_ms")
  fixed: dict[str, float] = field(default_factory=dict)  # pinned overrides
  sim_dt: float = SIM_DT
  settle_s: float = SETTLE_S
  skip_ms: float = LOSS_SKIP_MS


class Evaluator:
  """Maps a normalized CMA vector -> mean replay loss over captures.

  ``target_joint`` None means shared mode (params applied to all 12
  servos); otherwise only that joint gets the candidate parameters and
  the rest are HELD: each held joint uses DEFAULT_PARAMS updated with
  ``held_values[joint]`` when present (pass-1 shared-fit values in the
  block-coordinate scheme). The candidate delay applies globally — held
  joints have constant targets, so the delay is irrelevant to them.

  The replay regime is selected PER CAPTURE from its session manifest
  (2026-07-15): one ReplayModel per distinct fixed_base value among the
  captures — free base + floor for stand sessions, welded base + IMU
  gravity for bench sessions.
  """

  def __init__(
    self,
    captures: list[tuple[TestCapture, tuple[str, ...]]],
    cfg: FitConfig,
    target_joint: str | None,
    held_values: dict[str, dict[str, float]] | None = None,
  ):
    self.captures = captures
    self.cfg = cfg
    self.target_joint = target_joint
    self.held_values = held_values
    regimes = sorted({cap.fixed_base for cap, _ in captures})
    self.rms: dict[bool, ReplayModel] = {
      fixed_base: build_replay_model(fixed_base=fixed_base)
      for fixed_base in regimes
    }
    self._data_pid: int | None = None
    self._data: dict[bool, object] = {}

  @property
  def rm(self) -> ReplayModel:
    """The single replay model (single-regime capture sets only)."""
    if len(self.rms) != 1:
      raise ValueError(
        f"evaluator spans {len(self.rms)} replay regimes "
        f"{sorted(self.rms)}; there is no single model"
      )
    return next(iter(self.rms.values()))

  def rm_for(self, capture: TestCapture) -> ReplayModel:
    return self.rms[capture.fixed_base]

  def _mjdata(self, rm: ReplayModel):
    import mujoco

    pid = os.getpid()
    if self._data_pid != pid:
      self._data = {}
      self._data_pid = pid
    if rm.fixed_base not in self._data:
      self._data[rm.fixed_base] = mujoco.MjData(rm.model)
    return self._data[rm.fixed_base]

  def params_from_values(
    self, values: dict[str, float]
  ) -> tuple[list[ServoParams], float]:
    """Physical param dict (fit names only) -> 12 ServoParams + delay."""
    base = dict(DEFAULT_PARAMS)
    base.update(self.cfg.fixed)
    fitted = dict(base)
    fitted.update({k: v for k, v in values.items() if k != "delay_ms"})
    delay = float(values.get("delay_ms", base["delay_ms"]))
    fitted["delay_ms"] = delay
    base["delay_ms"] = delay

    def mk(d: dict[str, float]) -> ServoParams:
      return ServoParams(
        **{k: d[k] for k in (*SERVO_PARAM_NAMES, "delay_ms")}
      )

    if self.target_joint is None:
      params = [mk(fitted)] * 12
    else:
      held = self.held_values or {}

      def merged(name: str) -> dict[str, float]:
        d = dict(base)
        d.update(
          {k: v for k, v in held.get(name, {}).items()
           if k in SERVO_PARAM_NAMES}
        )
        return d

      # The target inherits its held (shared) values for parameters NOT
      # in the fitted subset — a restricted pass-2 fit (e.g. deadband +
      # delay only) must refine the shared solution, not DEFAULT_PARAMS.
      tgt = merged(self.target_joint)
      tgt.update({k: v for k, v in values.items() if k != "delay_ms"})
      tgt["delay_ms"] = delay
      params = [
        mk(tgt) if name == self.target_joint else mk(merged(name))
        for name in LEG_JOINTS
      ]
    return params, delay

  def values_from_vector(self, z: np.ndarray) -> dict[str, float]:
    phys = denormalize(z, self.cfg.fit_names)
    return dict(zip(self.cfg.fit_names, phys.tolist()))

  def eval_values(self, values: dict[str, float]) -> float:
    params, delay = self.params_from_values(values)
    losses = []
    for capture, joints in self.captures:
      rm = self.rm_for(capture)
      traj = replay_capture(
        rm, capture, params, delay,
        sim_dt=self.cfg.sim_dt, settle_s=self.cfg.settle_s,
        data=self._mjdata(rm),
      )
      losses.append(
        capture_loss(traj, capture, joints, skip_ms=self.cfg.skip_ms)
      )
    losses = [x for x in losses if np.isfinite(x)]
    return float(np.mean(losses)) if losses else float("inf")

  def eval_vector(self, z: np.ndarray) -> float:
    return self.eval_values(self.values_from_vector(z))


# Module-level singleton inherited by fork()ed pool workers.
_EVALUATOR: Evaluator | None = None


def _set_evaluator(ev: Evaluator) -> None:
  global _EVALUATOR
  _EVALUATOR = ev


def _worker_eval(z: np.ndarray) -> float:
  assert _EVALUATOR is not None
  return _EVALUATOR.eval_vector(z)


@dataclass
class FitResult:
  target_joint: str | None
  fit_names: list[str]
  values: dict[str, float]  # fitted physical values
  loss: float
  baseline_loss: float
  loss_trace: list[dict]  # per-gen {gen, best, mean, spread, sigma}
  generations_run: int
  early_stopped: bool
  wall_s: float


def run_cma_fit(
  evaluator: Evaluator,
  generations: int,
  popsize: int,
  sigma0: float = 0.5,
  seed: int = 0,
  workers: int | None = None,
  spread_tol: float = EARLY_STOP_SPREAD,
  verbose: bool = True,
  x0: np.ndarray | None = None,
  plateau_gens: int = PLATEAU_GENS,
  plateau_rtol: float = PLATEAU_RTOL,
) -> FitResult:
  """CMA-ES over the evaluator's normalized parameter cube.

  ``x0`` warm-starts the search mean (normalized coords; e.g. the pass-1
  shared solution for pass-2 per-joint fits). Stops early when the
  population spread collapses (PACE) or when the best-seen loss has not
  improved by ``plateau_rtol`` (relative) within ``plateau_gens``
  generations.
  """
  t_start = time.time()
  names = evaluator.cfg.fit_names
  baseline_loss = evaluator.eval_values({})  # DEFAULT/current-stack model

  if x0 is None:
    x0 = np.zeros(len(names))
  es = CmaEs(
    x0=np.clip(np.asarray(x0, dtype=float), -1.0, 1.0),
    sigma0=sigma0, popsize=popsize, seed=seed,
  )
  # Backend dispatch: an evaluator exposing eval_population (the warp/GPU
  # backend) evaluates a whole generation in one batched call; otherwise
  # fall back to the fork-pool CPU batch.
  batched = callable(getattr(evaluator, "eval_population", None))
  workers = workers or max(1, (os.cpu_count() or 2) - 1)
  pool = None
  if not batched:
    _set_evaluator(evaluator)
    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=min(workers, popsize))

  best_z: np.ndarray | None = None
  best_loss = float("inf")
  best_gen = 0
  plateau_ref = float("inf")
  trace: list[dict] = []
  early = False
  try:
    for gen in range(generations):
      x = es.ask()
      if batched:
        losses = np.asarray(evaluator.eval_population(list(x)), dtype=float)
      else:
        assert pool is not None
        losses = np.asarray(
          pool.map(_worker_eval, list(x), chunksize=1), dtype=float
        )
      es.tell(x, losses)
      i_best = int(np.argmin(losses))
      if losses[i_best] < best_loss:
        best_loss = float(losses[i_best])
        best_z = x[i_best].copy()
      lo, hi = float(np.min(losses)), float(np.max(losses))
      spread = (hi - lo) / max(lo, 1e-12)
      trace.append(
        {
          "gen": gen, "best": lo, "mean": float(np.mean(losses)),
          "spread": spread, "sigma": float(es.sigma),
        }
      )
      if verbose:
        label = evaluator.target_joint or "shared"
        print(
          f"  [{label}] gen {gen + 1}/{generations} "
          f"best {lo:.3e} mean {np.mean(losses):.3e} "
          f"spread {spread:.3f} sigma {es.sigma:.3f}",
          flush=True,
        )
      if best_loss < plateau_ref * (1.0 - plateau_rtol):
        plateau_ref = best_loss
        best_gen = gen
      if gen + 1 >= EARLY_STOP_MIN_GENS and spread < spread_tol:
        early = True
        break
      if plateau_gens and gen - best_gen >= plateau_gens:
        if verbose:
          print(
            f"  [{evaluator.target_joint or 'shared'}] plateau stop at "
            f"gen {gen + 1} (no >{plateau_rtol:.0%} improvement in "
            f"{plateau_gens} gens)",
            flush=True,
          )
        early = True
        break
  finally:
    if pool is not None:
      pool.close()
      pool.join()

  # PACE reports the distribution mean; with small budgets the best-seen
  # candidate can beat it, so evaluate both and keep the winner.
  mean_loss = evaluator.eval_vector(es.mean)
  if mean_loss <= best_loss:
    best_loss, best_z = mean_loss, es.mean.copy()
  assert best_z is not None
  values = evaluator.values_from_vector(best_z)
  return FitResult(
    target_joint=evaluator.target_joint,
    fit_names=list(names),
    values=values,
    loss=best_loss,
    baseline_loss=baseline_loss,
    loss_trace=trace,
    generations_run=len(trace),
    early_stopped=early,
    wall_s=time.time() - t_start,
  )
