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

  def _update_metrics(self) -> None:
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
