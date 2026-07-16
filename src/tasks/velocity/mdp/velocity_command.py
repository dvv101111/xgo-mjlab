from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

from .rewards import body_pitch_from_gravity

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    # Grid-adaptive command curriculum (Margolis et al., "Walk These Ways",
    # RSS 2022 / IJRR 2024 RewardThresholdCurriculum). When enabled, the
    # (vx, wz) pair is drawn from a persistent cell-weight grid instead of
    # the box-uniform draw; vy keeps its independent uniform draw. The
    # weights are updated by the `command_grid_adaptive` curriculum term
    # (curriculums.py) from episodic tracking performance.
    self.grid_enabled = cfg.grid_curriculum is not None
    if self.grid_enabled:
      gc = cfg.grid_curriculum
      assert gc is not None
      # The exclusive resample lottery (axis_focus / slow_vx / fast_vx /
      # turn_at_speed) shapes exposure by hand; the grid owns (vx, wz)
      # exposure entirely, so the two mechanisms must not compose. This is
      # checked here (not only in __post_init__) because presets mutate the
      # cfg after construction.
      if (
        cfg.axis_focus_probs is not None
        or cfg.slow_vx_prob > 0.0
        or cfg.fast_vx_prob > 0.0
        or cfg.turn_at_speed_prob > 0.0
      ):
        raise ValueError(
          "grid_curriculum is incompatible with the focus-mode lottery "
          "(axis_focus_probs/slow_vx/fast_vx/turn_at_speed); clear them."
        )
      if cfg.heading_command:
        raise ValueError(
          "grid_curriculum is incompatible with heading_command (heading "
          "overrides the sampled wz, breaking cell attribution)."
        )
      # The grid snapshots ranges.lin_vel_x / ranges.ang_vel_z at build time;
      # stage-based range mutation (commands_vel) must not be combined with it.
      vx_lo, vx_hi = cfg.ranges.lin_vel_x
      wz_lo, wz_hi = cfg.ranges.ang_vel_z
      self._grid_n_vx = max(1, round((vx_hi - vx_lo) / gc.vx_cell_size))
      self._grid_n_wz = max(1, round((wz_hi - wz_lo) / gc.wz_cell_size))
      # Actual cell sizes: exact tiling of the range (equals the cfg cell
      # size when the range is an integer multiple of it).
      self._grid_vx_lo = vx_lo
      self._grid_wz_lo = wz_lo
      self._grid_vx_size = (vx_hi - vx_lo) / self._grid_n_vx
      self._grid_wz_size = (wz_hi - wz_lo) / self._grid_n_wz
      vx_centers = vx_lo + (torch.arange(self._grid_n_vx, device=self.device) + 0.5) * self._grid_vx_size
      wz_centers = wz_lo + (torch.arange(self._grid_n_wz, device=self.device) + 0.5) * self._grid_wz_size
      # Seed region: weight 1.0 for cells whose CENTER falls inside the
      # known-trackable band, 0.0 elsewhere (unlocked later by the
      # curriculum term, +0.2 per passed episode on the cell + 4-neighbors).
      seed_vx = (vx_centers >= gc.seed_lin_vel_x[0]) & (vx_centers <= gc.seed_lin_vel_x[1])
      seed_wz = (wz_centers >= gc.seed_ang_vel_z[0]) & (wz_centers <= gc.seed_ang_vel_z[1])
      self.grid_seed_mask = seed_vx.unsqueeze(1) & seed_wz.unsqueeze(0)
      if not bool(self.grid_seed_mask.any()):
        raise ValueError(
          "grid_curriculum seed region contains no cell center; widen "
          "seed_lin_vel_x/seed_ang_vel_z or shrink the cell sizes."
        )
      self.grid_weights = torch.where(
        self.grid_seed_mask,
        torch.ones((), device=self.device),
        torch.zeros((), device=self.device),
      ).float()
      # Cell the env's CURRENT command belongs to (last resample), -1 when
      # the episode must not gate cell unlocks (twist zeroed by the 0.05
      # stand gate / pose_hold, or a rel_standing_envs standing episode).
      self.grid_cell_index = torch.full(
        (self.num_envs,), -1, dtype=torch.long, device=self.device
      )
      # Last logged seed-region tracking fraction (kept across curriculum
      # calls with no attributable envs so the log stays continuous).
      self.grid_seed_tracking_last = 0.0
      if gc.unlock_min_vel_ratio is not None and not (
        0.0 < gc.unlock_min_vel_ratio <= 1.0
      ):
        raise ValueError(
          f"grid_curriculum.unlock_min_vel_ratio must be in (0, 1], got "
          f"{gc.unlock_min_vel_ratio} (ratios are clipped to [0, 1])."
        )

    # v21b-v3 achieved-velocity ratio gate (opt-in; see GridCurriculumCfg.
    # unlock_min_vel_ratio). When armed, per-env EPISODE SUMS of the achieved
    # base twist are accumulated once per control step in _update_metrics
    # (the exact root_link velocities the tracking rewards read) and zeroed
    # on episode reset — the same hooks/semantics as the reward manager's
    # _episode_sums the frac gates divide. command_grid_adaptive divides by
    # episode_length_buf to get the episode-mean achieved velocity.
    self.grid_ratio_enabled = (
      self.grid_enabled
      and cfg.grid_curriculum is not None
      and cfg.grid_curriculum.unlock_min_vel_ratio is not None
    )
    if self.grid_ratio_enabled:
      self.grid_achieved_lin_sum = torch.zeros(
        self.num_envs, 2, device=self.device
      )
      self.grid_achieved_ang_sum = torch.zeros(self.num_envs, device=self.device)

    # v14 body-pose channels: when pose_mode_probs is set, the command grows
    # from [vx, vy, wz] to [vx, vy, wz, body_pitch, base_height].
    self.pose_enabled = cfg.pose_mode_probs is not None
    if self.pose_enabled:
      if cfg.ranges.body_pitch is None or cfg.ranges.base_height is None:
        raise ValueError(
          "pose_mode_probs is set but ranges.body_pitch/base_height are None."
        )
      if cfg.nominal_pose is None:
        raise ValueError("pose_mode_probs is set but nominal_pose is None.")

    command_dim = 5 if self.pose_enabled else 3
    self.vel_command_b = torch.zeros(self.num_envs, command_dim, device=self.device)
    if self.pose_enabled:
      # Start at the nominal pose so pre-resample observations don't read a
      # nonsense height-0 command.
      assert self.cfg.nominal_pose is not None
      self.vel_command_b[:, 3] = self.cfg.nominal_pose[0]
      self.vel_command_b[:, 4] = self.cfg.nominal_pose[1]
    if self.pose_enabled and cfg.pose_height_band is not None:
      band = torch.tensor(cfg.pose_height_band, device=self.device)
      self._band_pitch = band[:, 0].contiguous()  # ascending knots
      self._band_lo = band[:, 1].contiguous()
      self._band_hi = band[:, 2].contiguous()
    else:
      self._band_pitch = None
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
    if self.pose_enabled:
      self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    if self.grid_ratio_enabled:
      # Zero the achieved-velocity episode sums for the envs that reset —
      # mirrors the reward manager zeroing its _episode_sums in the same
      # _reset_idx pass (AFTER command_grid_adaptive has read them).
      assert isinstance(env_ids, torch.Tensor)
      self.grid_achieved_lin_sum[env_ids] = 0.0
      self.grid_achieved_ang_sum[env_ids] = 0.0
    return super().reset(env_ids)

  def _update_metrics(self) -> None:
    if self.grid_ratio_enabled:
      # Episode sums of the achieved base twist (same source tensors the
      # tracking rewards read); command_grid_adaptive turns them into
      # episode means. Accumulated once per control step, like the reward
      # episode sums (one-step window offset: the manager computes commands
      # after _reset_idx, so the sum includes the post-reset sample and
      # excludes the final pre-reset one — same sample count, negligible
      # for an episode-mean).
      self.grid_achieved_lin_sum += self.robot.data.root_link_lin_vel_b[:, :2]
      self.grid_achieved_ang_sum += self.robot.data.root_link_ang_vel_b[:, 2]
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )
    if self.pose_enabled:
      pitch_meas = body_pitch_from_gravity(self.robot.data.projected_gravity_b)
      self.metrics["error_pitch"] += (
        torch.abs(self.vel_command_b[:, 3] - pitch_meas) / max_command_step
      )
      self.metrics["error_height"] += (
        torch.abs(self.vel_command_b[:, 4] - self.robot.data.root_link_pos_w[:, 2])
        / max_command_step
      )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    if self.grid_enabled:
      # Grid-adaptive draw: cell ~ Categorical(grid_weights), then uniform
      # inside the cell. vy keeps the independent box-uniform draw. The
      # focus-mode lottery below is structurally off (enforced in __init__).
      flat_w = self.grid_weights.reshape(-1)
      cell = torch.multinomial(flat_w, len(env_ids), replacement=True)
      ix = torch.div(cell, self._grid_n_wz, rounding_mode="floor").float()
      iz = torch.remainder(cell, self._grid_n_wz).float()
      u_vx = torch.rand(len(env_ids), device=self.device)
      u_wz = torch.rand(len(env_ids), device=self.device)
      self.vel_command_b[env_ids, 0] = self._grid_vx_lo + (ix + u_vx) * self._grid_vx_size
      self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
      self.vel_command_b[env_ids, 2] = self._grid_wz_lo + (iz + u_wz) * self._grid_wz_size
    else:
      self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
      self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
      self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    if self.cfg.axis_focus_probs is not None:
      # Axis-focused episodes: force pure-rotation / pure-lateral /
      # backward-only commands on a fraction of resamples. Independent
      # uniform sampling gives pure-lateral ~1% exposure, so minority
      # directions never get trained without this.
      p_rot, p_lat, p_back = self.cfg.axis_focus_probs
      u = torch.rand(len(env_ids), device=self.device)
      pure_rot = u < p_rot
      pure_lat = (u >= p_rot) & (u < p_rot + p_lat)
      back_only = (u >= p_rot + p_lat) & (u < p_rot + p_lat + p_back)
      self.vel_command_b[env_ids[pure_rot], 0:2] = 0.0
      self.vel_command_b[env_ids[pure_lat], 0] = 0.0
      self.vel_command_b[env_ids[pure_lat], 2] = 0.0
      self.vel_command_b[env_ids[back_only], 0] = -self.vel_command_b[
        env_ids[back_only], 0
      ].abs()
      self.vel_command_b[env_ids[back_only], 1:3] = 0.0
      if self.cfg.slow_vx_prob > 0.0:
        # v17 stiction-regime focus: pure-vx episodes with |vx| drawn from
        # slow_vx_band (signed uniformly). The 2026-07-11 ladders put the
        # worst uncorrected heading drift at the SLOW dwells (true |vx|
        # 0.06 -> +0.05..+0.09 rad/s left yaw on 3 of 4 corrections-OFF
        # sessions) — the regime where stiction dominates — yet uniform
        # sampling over (-0.40, 0.45) gives the |vx| 0.06-0.12 sim band
        # only ~5% exposure after the focus modes above. Draws share the
        # same u as the other focus modes, so probabilities stay exclusive.
        # Band must sit ABOVE the 0.05 stand gate below or the episodes
        # degrade to standing.
        p0 = p_rot + p_lat + p_back
        slow = (u >= p0) & (u < p0 + self.cfg.slow_vx_prob)
        slow_ids = env_ids[slow]
        lo, hi = self.cfg.slow_vx_band
        mag = torch.empty(len(slow_ids), device=self.device).uniform_(lo, hi)
        sign = torch.where(
          torch.rand(len(slow_ids), device=self.device) < 0.5, -1.0, 1.0
        )
        self.vel_command_b[slow_ids, 0] = mag * sign
        self.vel_command_b[slow_ids, 1:3] = 0.0
      # Aggressive-preset focus modes (2026-07-11), same exclusive lottery:
      # fat sampling at high |vx| (fast_vx) and mixed vx+wz turning at speed
      # (turn_at_speed). Uniform sampling over a sprint-wide vx range gives
      # the top-speed band only ~15-20% exposure and vx-with-strong-wz pairs
      # ~8%; these modes concentrate gradient where the preset's objective
      # actually lives.
      p_used = sum(self.cfg.axis_focus_probs) + self.cfg.slow_vx_prob
      if self.cfg.fast_vx_prob > 0.0:
        assert self.cfg.fast_vx_band_fwd is not None
        assert self.cfg.fast_vx_band_back is not None
        fast = (u >= p_used) & (u < p_used + self.cfg.fast_vx_prob)
        fast_ids = env_ids[fast]
        fwd = torch.rand(len(fast_ids), device=self.device) < self.cfg.fast_vx_fwd_frac
        vx_fwd = torch.empty(len(fast_ids), device=self.device).uniform_(
          *self.cfg.fast_vx_band_fwd
        )
        vx_back = torch.empty(len(fast_ids), device=self.device).uniform_(
          *self.cfg.fast_vx_band_back
        )
        self.vel_command_b[fast_ids, 0] = torch.where(fwd, vx_fwd, vx_back)
        self.vel_command_b[fast_ids, 1:3] = 0.0
        p_used += self.cfg.fast_vx_prob
      if self.cfg.turn_at_speed_prob > 0.0:
        turn = (u >= p_used) & (u < p_used + self.cfg.turn_at_speed_prob)
        turn_ids = env_ids[turn]
        self.vel_command_b[turn_ids, 0] = torch.empty(
          len(turn_ids), device=self.device
        ).uniform_(*self.cfg.turn_vx_band)
        self.vel_command_b[turn_ids, 1] = 0.0
        wz_mag = torch.empty(len(turn_ids), device=self.device).uniform_(
          *self.cfg.turn_wz_band
        )
        wz_sign = torch.where(
          torch.rand(len(turn_ids), device=self.device) < 0.5, -1.0, 1.0
        )
        self.vel_command_b[turn_ids, 2] = wz_mag * wz_sign
    lateral_focus = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
    if self.cfg.lateral_focus_prob > 0.0:
      # v21a pure-lateral focus episodes, GRID-COMPATIBLE (unlike the
      # exclusive axis_focus lottery, which the grid sampler forbids): with
      # probability lateral_focus_prob the (vx, wz) draw — grid or uniform —
      # is overridden by a pure-lateral command, |vy| ~ U(lateral_focus_band),
      # sign uniform. Rationale: the grid owns (vx, wz) but vy is an
      # independent box draw, so pure-lateral commands (the deployable
      # strafe) are measure-zero without a focus mode; the v21 lit review
      # (section 2.5) makes relaxed-offset lateral gaits a headline goal.
      # These envs are excluded from grid cell attribution below (they
      # demonstrate no (vx, wz) competence). No RNG is consumed when the
      # probability is 0 — pre-v21a presets sample bit-identically.
      lateral_focus = (
        torch.rand(len(env_ids), device=self.device) < self.cfg.lateral_focus_prob
      )
      lat_ids = env_ids[lateral_focus]
      mag = torch.empty(len(lat_ids), device=self.device).uniform_(
        *self.cfg.lateral_focus_band
      )
      sign = torch.where(
        torch.rand(len(lat_ids), device=self.device) < 0.5, -1.0, 1.0
      )
      self.vel_command_b[lat_ids, 0] = 0.0
      self.vel_command_b[lat_ids, 1] = mag * sign
      self.vel_command_b[lat_ids, 2] = 0.0
    if self.pose_enabled:
      # v14 pose-mode mix, applied at resample like axis_focus. Modes
      # (probabilities in cfg.pose_mode_probs, summing to 1):
      #   nominal:    pose = nominal_pose exactly, twist as sampled above
      #               (incl. axis_focus) -- keeps the v13b twist curriculum.
      #   pose_hold:  pose ~ U(ranges), twist forced to ZERO -- learn to
      #               reach and hold a body pose statically.
      #   posed_walk: pose ~ U(ranges), twist scaled by 0.5 -- walk while
      #               holding a non-nominal pose (slower: shorter legs when
      #               crouched can't cover the full twist range).
      assert self.cfg.pose_mode_probs is not None
      assert self.cfg.nominal_pose is not None
      assert self.cfg.ranges.body_pitch is not None
      assert self.cfg.ranges.base_height is not None
      self.vel_command_b[env_ids, 3] = r.uniform_(*self.cfg.ranges.body_pitch)
      self.vel_command_b[env_ids, 4] = r.uniform_(*self.cfg.ranges.base_height)
      p_nominal, p_hold, _ = self.cfg.pose_mode_probs
      u = torch.rand(len(env_ids), device=self.device)
      nominal = u < p_nominal
      pose_hold = (u >= p_nominal) & (u < p_nominal + p_hold)
      posed_walk = u >= p_nominal + p_hold
      self.vel_command_b[env_ids[nominal], 3] = self.cfg.nominal_pose[0]
      self.vel_command_b[env_ids[nominal], 4] = self.cfg.nominal_pose[1]
      self.vel_command_b[env_ids[pose_hold], :3] = 0.0
      self.vel_command_b[env_ids[posed_walk], :3] *= 0.5
      if self._band_pitch is not None:
        # v15: clamp the height command into the pitch-conditioned standable
        # band (the workspace is not a rectangle; see cfg.pose_height_band).
        pit = self.vel_command_b[env_ids, 3]
        idx = torch.clamp(
          torch.searchsorted(self._band_pitch, pit.contiguous()),
          1, len(self._band_pitch) - 1)
        p0, p1 = self._band_pitch[idx - 1], self._band_pitch[idx]
        w = ((pit - p0) / (p1 - p0)).clamp(0.0, 1.0)
        lo = self._band_lo[idx - 1] + w * (self._band_lo[idx] - self._band_lo[idx - 1])
        hi = self._band_hi[idx - 1] + w * (self._band_hi[idx] - self._band_hi[idx - 1])
        self.vel_command_b[env_ids, 4] = torch.maximum(
          lo, torch.minimum(hi, self.vel_command_b[env_ids, 4]))
    # 0.05 stand threshold: must sit BELOW the vy range (+/-0.08) or every
    # pure-lateral episode is zeroed into a standing episode (v11 bug).
    # Twist slice [:3] ONLY: with the height channel (~0.116) always in the
    # vector, a full-vector norm would never read "standing", and zeroing the
    # full row would erase the pose command.
    # Gate ordering: applied AFTER the posed_walk 0.5 scaling so the invariant
    # "an active twist command never has norm in (0, 0.05)" holds for every
    # gate (phase clock, gait, pose regimes, deploy STAND_CMD_NORM). Cost:
    # ~29% of posed_walk episodes degrade to pose_hold -- all pure-lateral
    # focus draws (0.5 * 0.08 = 0.04 < 0.05) plus slow-rotation/backward
    # tails -- so the effective mix is ~50/32/18, and posed walking is
    # forward/turn dominated. Accepted: 0.04 m/s posed lateral is below the
    # hardware's useful range anyway.
    self.vel_command_b[env_ids, :3] *= (torch.norm(self.vel_command_b[env_ids, :3], dim=1) > 0.05).unsqueeze(1)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    if self.grid_enabled:
      # Cell attribution for the curriculum term: match each env to the cell
      # CONTAINING its final post-mutation (vx, wz) — after posed_walk 0.5
      # scaling and the 0.05 norm gate — so unlock credit lands where
      # competence was actually demonstrated. Envs whose twist ended up
      # zeroed (norm gate / pose_hold) or that were drawn as standing envs
      # (rel_standing_envs; twist zeroed every step in _update_command) get
      # -1: standing tracks a zero command trivially and must not unlock
      # cells. Episodes spanning several resamples are attributed to the
      # LAST drawn cell only (approximation: episodic reward sums cannot be
      # split per command dwell).
      vx_f = self.vel_command_b[env_ids, 0]
      wz_f = self.vel_command_b[env_ids, 2]
      ix = torch.clamp(
        ((vx_f - self._grid_vx_lo) / self._grid_vx_size).floor().long(),
        0, self._grid_n_vx - 1,
      )
      iz = torch.clamp(
        ((wz_f - self._grid_wz_lo) / self._grid_wz_size).floor().long(),
        0, self._grid_n_wz - 1,
      )
      cell = ix * self._grid_n_wz + iz
      zeroed = torch.norm(self.vel_command_b[env_ids, :3], dim=1) == 0.0
      # lateral_focus episodes track vy only ((vx, wz) forced to zero) and
      # must not unlock (vx, wz) cells any more than standing envs do.
      cell[zeroed | self.is_standing_env[env_ids] | lateral_focus] = -1
      self.grid_cell_index[env_ids] = cell

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    # Zero the twist slice only: standing envs still track their pose command.
    self.vel_command_b[standing_env_ids, :3] = 0.0

  # GUI.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  init_velocity_prob: float = 0.0
  # (p_pure_rotation, p_pure_lateral, p_backward_only) applied at resample;
  # None = plain independent uniform sampling (upstream behavior).
  axis_focus_probs: tuple[float, float, float] | None = None
  # v17: probability of a pure slow-vx episode (stiction regime), drawn from
  # the same exclusive resample lottery as axis_focus_probs (their sum plus
  # this must stay <= 1). |vx| ~ U(slow_vx_band), sign uniform, vy = wz = 0.
  # Requires axis_focus_probs to be set. The band lower edge must exceed the
  # 0.05 stand gate.
  slow_vx_prob: float = 0.0
  slow_vx_band: tuple[float, float] = (0.06, 0.12)
  # Aggressive-preset focus modes (2026-07-11), drawn from the same exclusive
  # lottery (axis_focus + slow_vx + fast_vx + turn_at_speed must sum <= 1;
  # both require axis_focus_probs to be set).
  # fast_vx: pure-vx episode with |vx| in the TOP band of the range — fat
  # sampling at sprint speeds. Forward with prob fast_vx_fwd_frac, else
  # backward (band given as (lo, hi) with lo < hi, negative for backward).
  fast_vx_prob: float = 0.0
  fast_vx_band_fwd: tuple[float, float] | None = None
  fast_vx_band_back: tuple[float, float] | None = None
  fast_vx_fwd_frac: float = 0.65
  # turn_at_speed: forward vx from turn_vx_band combined with a strong yaw
  # command (|wz| from turn_wz_band, sign uniform), vy zero — trains turning
  # while moving instead of the pure-rotation / pure-vx split.
  turn_at_speed_prob: float = 0.0
  turn_vx_band: tuple[float, float] = (0.15, 0.45)
  turn_wz_band: tuple[float, float] = (0.5, 1.5)
  # v21a pure-lateral focus (2026-07-15), grid-COMPATIBLE: applied AFTER the
  # (vx, wz) draw (grid or uniform), overriding it with vx = wz = 0 and
  # |vy| ~ U(lateral_focus_band) (sign uniform) on a lateral_focus_prob
  # fraction of resamples. Unlike axis_focus_probs this composes with
  # grid_curriculum: the affected envs are excluded from cell attribution.
  # Band lower edge must exceed the 0.05 stand gate. 0.0 = off (default,
  # no RNG consumed — pre-v21a presets sample bit-identically).
  lateral_focus_prob: float = 0.0
  lateral_focus_band: tuple[float, float] = (0.08, 0.20)
  # v14 body-pose channels (opt-in; all None = upstream 3-dim behavior).
  # (p_nominal, p_pose_hold, p_posed_walk) applied at resample; must sum to 1.
  # When set, the command is [vx, vy, wz, body_pitch, base_height] and
  # ranges.body_pitch / ranges.base_height / nominal_pose are required.
  pose_mode_probs: tuple[float, float, float] | None = None
  # (body_pitch [rad, positive = nose up], base_height [m, root z above the
  # floor plane]) used verbatim in nominal-mode episodes.
  nominal_pose: tuple[float, float] | None = None
  # v15: the standable (pitch, height) workspace is NOT a rectangle — nose-up
  # needs extended rear legs (no deep crouch), near-level pitch allows the
  # full height range. Knots of (pitch, h_lo, h_hi) [floor convention],
  # piecewise-linearly interpolated; sampled heights are clamped into the
  # band for the sampled pitch. Derived from Quadruped-robot
  # src/xgo/openfw/body.py leg IK (12% width margin per pitch).
  pose_height_band: tuple[tuple[float, float, float], ...] | None = None

  @dataclass
  class GridCurriculumCfg:
    """Grid-adaptive command curriculum (Margolis et al., RSS 2022).

    Rescaled RewardThresholdCurriculum: the (vx, wz) plane is tiled into
    cells; sampling draws a cell proportionally to a persistent weight
    grid, then uniform inside the cell. Weights start at 1.0 inside the
    seed region (v17's known-trackable band) and 0.0 elsewhere; the
    ``command_grid_adaptive`` curriculum term adds +0.2 to a cell and its
    4-connected neighbors whenever an episode attributed to that cell
    passes BOTH tracking thresholds (weights clipped to [0, 1]). This is
    the validated fix for the tight-tolerance from-scratch collapse
    ("converges to jittering in place", IJRR 2024 no-curriculum ablation).
    vy is NOT gridded (kept independent uniform: its +-0.08 range is a
    single cell wide anyway). Incompatible with the focus-mode lottery and
    heading_command; the grid snapshots ranges.lin_vel_x/ang_vel_z at env
    build time.
    """

    # Nominal cell sizes; the actual size is range_span / round(span/size)
    # so the range tiles exactly. 0.05 m/s makes the SLOW band (0.05-0.15)
    # its own pair of cells that must individually earn competence.
    vx_cell_size: float = 0.05
    wz_cell_size: float = 0.25
    # Initial weight-1.0 region (cells selected by center). Defaults:
    # v17's known-trackable band.
    seed_lin_vel_x: tuple[float, float] = (-0.15, 0.25)
    seed_ang_vel_z: tuple[float, float] = (-0.5, 0.5)
    # v21b-v3 achieved-velocity AND-gate (opt-in; None = pre-existing
    # frac-only behavior, bit-identical for every task that leaves it
    # unset). The v21b 10k postmortem (docs/research/
    # v19-postmortem-v4-plant-2026-07-15.md, gate-metric fix): idle
    # episodes on lethal terrain pass the frac gates on cells nobody
    # actually tracks (the v19-v5 idle-inversion), unlocking 100% of the
    # grid by iter 2000 while the policy stands still. When set, a cell
    # unlock ADDITIONALLY requires the cell's mean achieved/commanded
    # velocity ratio >= this threshold on EVERY commanded axis:
    #   - linear (episodes with |cmd_xy| >= 0.05): episode-mean of
    #     dot(achieved_v_xy, cmd_xy) / |cmd_xy|^2, clipped to [0, 1]
    #     (directional — reverse motion scores 0);
    #   - angular (episodes with |wz| >= 0.1): analogous on wz;
    #   - cells commanding neither axis (stand cells) are exempt (frac
    #     gates only).
    # Ratios are averaged per cell over the episodes attributed to it in
    # the reset batch (same attribution/reset hooks as the frac gates).
    unlock_min_vel_ratio: float | None = None

  # None = off: the sampler is bit-identical to the pre-grid code (all grid
  # code is behind `if self.grid_enabled` and consumes no RNG when off).
  grid_curriculum: GridCurriculumCfg | None = None

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None
    # v14 pose channels; sampled uniformly in pose_hold / posed_walk modes.
    body_pitch: tuple[float, float] | None = None
    base_height: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
    if self.grid_curriculum is not None:
      # Same checks as UniformVelocityCommand.__init__ (which re-validates
      # because presets mutate cfgs after construction).
      if (
        self.axis_focus_probs is not None
        or self.slow_vx_prob > 0.0
        or self.fast_vx_prob > 0.0
        or self.turn_at_speed_prob > 0.0
      ):
        raise ValueError(
          "grid_curriculum is incompatible with the focus-mode lottery "
          "(axis_focus_probs/slow_vx/fast_vx/turn_at_speed); clear them."
        )
      if self.heading_command:
        raise ValueError("grid_curriculum is incompatible with heading_command.")
      if self.grid_curriculum.vx_cell_size <= 0 or self.grid_curriculum.wz_cell_size <= 0:
        raise ValueError("grid_curriculum cell sizes must be positive.")
      if self.grid_curriculum.unlock_min_vel_ratio is not None and not (
        0.0 < self.grid_curriculum.unlock_min_vel_ratio <= 1.0
      ):
        raise ValueError(
          f"grid_curriculum.unlock_min_vel_ratio must be in (0, 1], got "
          f"{self.grid_curriculum.unlock_min_vel_ratio}."
        )
    if self.slow_vx_prob or self.fast_vx_prob or self.turn_at_speed_prob:
      if self.axis_focus_probs is None:
        raise ValueError(
          "slow_vx/fast_vx/turn_at_speed probs require axis_focus_probs."
        )
      total = (
        sum(self.axis_focus_probs)
        + self.slow_vx_prob
        + self.fast_vx_prob
        + self.turn_at_speed_prob
      )
      if total > 1.0 + 1e-6:
        raise ValueError(
          f"focus-mode probabilities must sum to <= 1, got {total}."
        )
    if self.slow_vx_prob:
      if self.slow_vx_band[0] <= 0.05 or self.slow_vx_band[1] <= self.slow_vx_band[0]:
        raise ValueError(
          f"slow_vx_band must be ascending and sit above the 0.05 stand "
          f"gate, got {self.slow_vx_band}."
        )
    if self.lateral_focus_prob:
      if not (0.0 < self.lateral_focus_prob <= 1.0):
        raise ValueError(
          f"lateral_focus_prob must be in (0, 1], got {self.lateral_focus_prob}."
        )
      if (
        self.lateral_focus_band[0] <= 0.05
        or self.lateral_focus_band[1] <= self.lateral_focus_band[0]
      ):
        raise ValueError(
          f"lateral_focus_band must be ascending and sit above the 0.05 "
          f"stand gate, got {self.lateral_focus_band}."
        )
    if self.fast_vx_prob:
      if self.fast_vx_band_fwd is None or self.fast_vx_band_back is None:
        raise ValueError("fast_vx_prob requires both fast_vx bands.")
      if not (
        self.fast_vx_band_fwd[0] < self.fast_vx_band_fwd[1]
        and self.fast_vx_band_back[0] < self.fast_vx_band_back[1]
      ):
        raise ValueError("fast_vx bands must be ascending (lo, hi) tuples.")
    if self.pose_mode_probs is not None:
      if len(self.pose_mode_probs) != 3 or abs(sum(self.pose_mode_probs) - 1.0) > 1e-6:
        raise ValueError(
          f"pose_mode_probs must be 3 probabilities summing to 1, got "
          f"{self.pose_mode_probs}."
        )
      if self.ranges.body_pitch is None or self.ranges.base_height is None:
        raise ValueError(
          "pose_mode_probs is set but ranges.body_pitch/base_height are None."
        )
      if self.nominal_pose is None:
        raise ValueError("pose_mode_probs is set but nominal_pose is None.")
