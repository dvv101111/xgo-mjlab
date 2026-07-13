"""Local actuator extensions for the XGO velocity tasks.

Per-servo response-delay DR (2026-07-12 sim-fidelity fix #2): mjlab's
``DelayedActuator`` draws ONE lag per environment, so all 12 servos of a
robot share the same command delay. Real hobby serial-bus servos have
per-servo, load-dependent lag, and it is the DIFFERENTIAL lag between the
loaded and unloaded legs that desynchronizes gaits on hardware (front/rear
desync seen on the Sprint/Agile deploys). ``PerServoDelayedActuator`` keeps
the exact update policy of mjlab's ``DelayBuffer`` (min/max lag, hold
probability, periodic staggered refresh) but draws the lag INDEPENDENTLY
per (env, servo).

Opt-in: the default robot config keeps mjlab's ``DelayedActuatorCfg``
untouched; presets switch via ``enable_per_servo_delay`` in
``config/xgolite/sim_fidelity.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.delayed_actuator import DelayedActuator, DelayedActuatorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity

__all__ = (
  "PerServoDelayBuffer",
  "PerServoDelayedActuator",
  "PerServoDelayedActuatorCfg",
)


class PerServoDelayBuffer:
  """DelayBuffer variant with an independent lag per (env, target).

  Mirrors ``mjlab.utils.buffers.DelayBuffer`` semantics exactly, except the
  lag tensor has shape ``(batch_size, num_targets)`` instead of
  ``(batch_size,)``:

  - the refresh TICK is per env (``update_period`` steps, staggered by a
    per-env phase offset) — matching the shared-battery / shared-bus timing
    of the real robot;
  - on a refresh tick each target independently redraws its lag from
    ``[min_lag, max_lag]``, each keeping its previous lag with probability
    ``hold_prob`` (temporal correlation per servo);
  - after ``reset`` a row starts at lag 0 and its history is backfilled
    with the first appended value (same as ``CircularBuffer``).
  """

  def __init__(
    self,
    min_lag: int,
    max_lag: int,
    batch_size: int,
    num_targets: int,
    device: str,
    hold_prob: float = 0.0,
    update_period: int = 0,
    per_env_phase: bool = True,
  ) -> None:
    if min_lag < 0:
      raise ValueError(f"min_lag must be >= 0, got {min_lag}")
    if max_lag < min_lag:
      raise ValueError(f"max_lag ({max_lag}) must be >= min_lag ({min_lag})")
    if not 0.0 <= hold_prob <= 1.0:
      raise ValueError(f"hold_prob must be in [0, 1], got {hold_prob}")
    if update_period < 0:
      raise ValueError(f"update_period must be >= 0, got {update_period}")

    self.min_lag = min_lag
    self.max_lag = max_lag
    self.batch_size = batch_size
    self.num_targets = num_targets
    self.device = device
    self.hold_prob = hold_prob
    self.update_period = update_period
    self.per_env_phase = per_env_phase

    self._max_len = max_lag + 1 if max_lag > 0 else 1
    self._buffer: torch.Tensor | None = None  # (max_len, batch, num_targets)
    self._pointer: int = -1
    self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
    self._current_lags = torch.zeros(
      (batch_size, num_targets), dtype=torch.long, device=device
    )
    self._step_count = torch.zeros(batch_size, dtype=torch.long, device=device)
    # Cached index helpers for the per-(env, target) gather.
    self._env_idx = (
      torch.arange(batch_size, device=device)
      .unsqueeze(1)
      .expand(batch_size, num_targets)
    )
    self._tgt_idx = (
      torch.arange(num_targets, device=device)
      .unsqueeze(0)
      .expand(batch_size, num_targets)
    )

    if update_period > 0 and per_env_phase:
      self._phase_offsets = torch.randint(
        0, update_period, (batch_size,), dtype=torch.long, device=device
      )
    else:
      self._phase_offsets = torch.zeros(batch_size, dtype=torch.long, device=device)

  @property
  def is_initialized(self) -> bool:
    return self._buffer is not None

  @property
  def current_lags(self) -> torch.Tensor:
    """Current lag per (env, target). Shape: (batch_size, num_targets)."""
    return self._current_lags

  def set_lags(
    self,
    lags: torch.Tensor,
    batch_ids: Sequence[int] | torch.Tensor | slice | None = None,
  ) -> None:
    """Set lag values for specified environments.

    Args:
      lags: Lag values. Scalar, ``(num_batch_ids,)`` (broadcast across
        targets, DelayBuffer-compatible) or ``(num_batch_ids, num_targets)``.
      batch_ids: Batch indices to set, or None to set all.
    """
    idx = slice(None) if batch_ids is None else batch_ids
    lags = lags.to(device=self.device, dtype=torch.long)
    if lags.ndim == 1:
      lags = lags.unsqueeze(-1)
    self._current_lags[idx] = lags.clamp(self.min_lag, self.max_lag)

  def reset(
    self, batch_ids: Sequence[int] | torch.Tensor | slice | None = None
  ) -> None:
    idx = slice(None) if batch_ids is None else batch_ids
    self._num_pushes[idx] = 0
    self._current_lags[idx] = 0
    self._step_count[idx] = 0
    if self._buffer is not None:
      self._buffer[:, idx] = 0.0
    if self.update_period > 0 and self.per_env_phase:
      new_phases = torch.randint(
        0,
        self.update_period,
        (self.batch_size,),
        dtype=torch.long,
        device=self.device,
      )
      self._phase_offsets[idx] = new_phases[idx]

  def append(self, data: torch.Tensor) -> None:
    """Append a new frame. Shape: (batch_size, num_targets)."""
    if data.shape[0] != self.batch_size:
      raise ValueError(f"Expected batch size {self.batch_size}, got {data.shape[0]}")
    data = data.to(self.device)

    if self._buffer is None:
      self._pointer = -1
      self._buffer = torch.empty(
        (self._max_len, *data.shape), dtype=data.dtype, device=self.device
      )

    self._pointer = (self._pointer + 1) % self._max_len
    self._buffer[self._pointer] = data

    # Backfill entire history with the first frame for new/reset rows.
    is_first_push = self._num_pushes == 0
    if torch.any(is_first_push):
      self._buffer[:, is_first_push] = data[is_first_push]

    self._num_pushes += 1

  def compute(self) -> torch.Tensor:
    """Return the delayed frame for the current step. Shape: (batch, targets)."""
    if self._buffer is None:
      raise RuntimeError("Buffer not initialized. Call append() first.")

    self._update_lags()

    # Clamp lags to the valid history per env (backfill makes older frames
    # identical anyway, matching DelayBuffer behavior).
    available = (
      torch.minimum(
        self._num_pushes,
        torch.full_like(self._num_pushes, self._max_len),
      )
      - 1
    ).clamp_min(0)
    valid_lags = torch.minimum(self._current_lags, available.unsqueeze(-1))
    idx = torch.remainder(self._pointer - valid_lags, self._max_len)
    return self._buffer[idx, self._env_idx, self._tgt_idx]

  def _update_lags(self) -> None:
    if self.update_period > 0:
      phase_adjusted = (self._step_count + self._phase_offsets) % self.update_period
      should_update = phase_adjusted == 0  # (batch,)
    else:
      should_update = torch.ones(
        self.batch_size, dtype=torch.bool, device=self.device
      )

    candidate = torch.randint(
      self.min_lag,
      self.max_lag + 1,
      (self.batch_size, self.num_targets),
      dtype=torch.long,
      device=self.device,
    )
    update_mask = should_update.unsqueeze(-1)
    if self.hold_prob > 0.0:
      should_sample = (
        torch.rand(
          (self.batch_size, self.num_targets),
          dtype=torch.float32,
          device=self.device,
        )
        >= self.hold_prob
      )
      update_mask = update_mask & should_sample

    self._current_lags = torch.where(update_mask, candidate, self._current_lags)
    self._step_count += 1


@dataclass(kw_only=True)
class PerServoDelayedActuatorCfg(DelayedActuatorCfg):
  """DelayedActuatorCfg whose lags are drawn independently per servo."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> PerServoDelayedActuator:
    base_actuator = self.base_cfg.build(entity, target_ids, target_names)
    return PerServoDelayedActuator(self, base_actuator)


class PerServoDelayedActuator(DelayedActuator):
  """DelayedActuator with an independent delay per (env, servo).

  Identical to the parent except ``initialize`` builds
  ``PerServoDelayBuffer`` instances; ``compute``/``reset``/``set_lags`` are
  inherited (the buffer API is interchangeable).
  """

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    self._base_actuator.initialize(mj_model, model, data, device)

    self._target_ids = self._base_actuator._target_ids
    self._ctrl_ids = self._base_actuator._ctrl_ids
    self._global_ctrl_ids = self._base_actuator._global_ctrl_ids

    targets = (
      (self.cfg.delay_target,)
      if isinstance(self.cfg.delay_target, str)
      else self.cfg.delay_target
    )
    num_targets = len(self._base_actuator._target_ids_list)

    for target in targets:
      self._delay_buffers[target] = PerServoDelayBuffer(  # type: ignore[assignment]
        min_lag=self.cfg.delay_min_lag,
        max_lag=self.cfg.delay_max_lag,
        batch_size=data.nworld,
        num_targets=num_targets,
        device=device,
        hold_prob=self.cfg.delay_hold_prob,
        update_period=self.cfg.delay_update_period,
        per_env_phase=self.cfg.delay_per_env_phase,
      )
