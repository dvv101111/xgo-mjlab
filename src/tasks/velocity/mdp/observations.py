from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def piecewise_linear(
  x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor
) -> torch.Tensor:
  """Piecewise-linear interpolation over ascending knots, clamped outside.

  Same knot-interpolation pattern as the pose_height_band lookup in
  ``velocity_command.py``; factored here because the scheduled phase clock
  and its check script both need it.
  """
  idx = torch.clamp(torch.searchsorted(xs, x.contiguous()), 1, len(xs) - 1)
  x0, x1 = xs[idx - 1], xs[idx]
  y0, y1 = ys[idx - 1], ys[idx]
  w = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0)
  return y0 + w * (y1 - y0)


class joint_vel_control_rate_rel:
  """``joint_vel_rel`` measured at CONTROL rate, not physics rate.

  v19 v3/v3_ext postmortem (2026-07-14): the measured servo plant is a
  stiff torque-saturated relay that dithers at the 500 Hz physics rate
  (see ``rewards.joint_acc_control_rate_l2`` for the full story). mjlab's
  ``joint_vel_rel`` reads the instantaneous physics-rate ``joint_vel``,
  so on this plant every observation frame carries ~0.5-1 rad/s of
  structured dither noise the policy can neither control nor exploit —
  and which the DEPLOYED stack does not have: the hardware loop feeds the
  policy finite-differenced telemetry positions (locomotion.py: "the
  policy obs stays on the finite-difference qd"). Both v19 v3 runs
  plateaued below the curriculum gate with this obs; this term is the
  obs-side analog of the control-rate joint-acc reward fix.

  Semantics: (joint_pos - prev_joint_pos) / step_dt across CONTROL steps
  (50 Hz), minus default_joint_vel — on smooth plants this equals the
  instantaneous velocity; under physics-rate dither the >25 Hz content
  aliases down bounded by the dither's sub-milliradian position
  amplitude (~0.05 rad/s) instead of its velocity amplitude. It is also
  exactly the signal the deploy loop computes from telemetry.

  First step after an env reset returns 0 (no previous position; deploy
  boots from a standstill where qd is 0 too).
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._prev_q: torch.Tensor | None = None
    self._valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._valid[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if self._prev_q is None:
      self._prev_q = q.clone()
    qd = (q - self._prev_q) / env.step_dt
    self._prev_q = q.clone()
    qd = qd * self._valid.float().unsqueeze(1)
    self._valid[:] = True
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    return qd - default_joint_vel[:, asset_cfg.joint_ids]


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    # 0.05: below the lateral command range so sidestep keeps a gait clock.
    # Twist slice [:3] only: with the v14 pose channels the height command
    # (~0.116) is always in the vector, so a full norm would never read
    # "standing" and the gait clock would tick during pose_hold episodes.
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) < 0.05
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase


class phase_scheduled:
  """Speed-scheduled per-env gait phase clock (XGOLite-V21A, 2026-07-15).

  Replaces the global episode clock of ``phase`` with a per-env stateful
  accumulator whose stepping frequency is a deterministic function of the
  commanded twist:

    speed_equiv = ||cmd_xy|| + wz_equiv * |cmd_wz|
    f = piecewise_linear(speed_equiv, freq_knots)   [Hz, clamped at the ends]
    phase <- (phase + f * step_dt) mod 1            each control step

  Rationale (docs/research/v21-terrain-gait-litreview-2026-07-15.md):
  nobody in 2023-2026 rough-terrain work keeps a fixed unmodulatable clock —
  the field schedules or modulates it (WTW arXiv:2212.03238 commands
  f in 1.5-4.0 Hz; Singh arXiv:2504.13619 learns a phase increment; PGTT
  arXiv:2510.18348 randomizes U[1,3] Hz). At our Froude number (~0.1-0.2
  across the envelope) dynamic similarity puts the robot in the WALK band,
  so the schedule anchors LOW near stand-adjacent speeds and rises to the
  validated 2.5 Hz trot clock at the envelope edge; the low band also
  shrinks the 41-120 ms actuation-latency fraction of the cycle (lit review
  section 2.1 verdict: "argue for the lower band, not higher").

  Contract kept from ``phase`` (the deploy obs frame is frozen at 49 dims):
  output is the same 2-dim (sin, cos); when the twist norm (slice [:3]) is
  below ``stand_norm`` the accumulator FREEZES (does not advance) and the
  output is zeroed exactly like the global-clock term. Resets to phase 0.

  Deploy parity: the schedule is exported as ONNX metadata
  ``phase_schedule`` (rl/runner.py); the hardware loop replays
  "phase += f(cmd) * dt per 50 Hz tick, reset to 0, freeze+zero below
  stand_norm". The gait rewards read this term's ``phase``/``freq`` buffers
  so reward and observation share one clock.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    knots = sorted(tuple(k) for k in cfg.params["freq_knots"])
    if len(knots) < 2:
      raise ValueError("phase_scheduled needs at least 2 freq_knots.")
    self._knot_speed = torch.tensor(
      [k[0] for k in knots], dtype=torch.float32, device=env.device
    )
    self._knot_freq = torch.tensor(
      [k[1] for k in knots], dtype=torch.float32, device=env.device
    )
    # Per-env accumulator state; ``freq`` holds the last scheduled frequency
    # (0 while frozen) for the gait rewards and the check script.
    self.phase = torch.zeros(env.num_envs, device=env.device)
    self.freq = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.phase[env_ids] = 0.0
    self.freq[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    freq_knots: tuple[tuple[float, float], ...],
    wz_equiv: float,
    stand_norm: float,
  ) -> torch.Tensor:
    del freq_knots  # Consumed in __init__.
    command = env.command_manager.get_command(command_name)
    speed = (
      torch.linalg.norm(command[:, :2], dim=1) + wz_equiv * command[:, 2].abs()
    )
    f = piecewise_linear(speed, self._knot_speed, self._knot_freq)
    # Twist slice [:3] only (see ``phase``): the v14 pose channels must not
    # keep the clock ticking during pose_hold episodes.
    active = torch.linalg.norm(command[:, :3], dim=1) >= stand_norm
    self.freq = torch.where(active, f, torch.zeros_like(f))
    self.phase = torch.where(
      active, (self.phase + f * env.step_dt) % 1.0, self.phase
    )
    angle = self.phase * (2.0 * math.pi)
    out = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)
    return torch.where(active.unsqueeze(1), out, torch.zeros_like(out))

