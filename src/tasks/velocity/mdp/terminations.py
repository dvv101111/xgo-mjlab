from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)

def sustained_illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 5.0,
) -> torch.Tensor:
  """Terminate only on contact held across the sensor's FULL history.

  Terrain variant of ``illegal_contact`` (v21b, 2026-07-15): on procedural
  terrain, spawn landings and step-edge brushes produce single-sample
  force spikes on leg geoms that ``illegal_contact``'s any-over-history
  logic turns into instant episode death (the v21b v1 collapse: eplen
  ~7-70 for 1300 iters, 332 illegal-contact terminations per window).
  Requiring every history sample above threshold (~80 ms at 4 slots /
  50 Hz) keeps genuine falls terminal — a fallen robot RESTS on the
  sensor geoms — while transients become the ``nonfoot_contact`` penalty's
  job.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  assert data.force_history is not None, (
    "sustained_illegal_contact needs a contact sensor with history_length > 1")
  force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
  return (force_mag > force_threshold).all(dim=-1).any(dim=-1)  # [B]
