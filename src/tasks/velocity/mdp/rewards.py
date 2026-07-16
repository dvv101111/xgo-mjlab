from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Axis "commanded" thresholds shared by the velocity-progress rewards and the
# grid-curriculum achieved-velocity ratio gate (curriculums.py imports these —
# they live here because rewards.py sits at the bottom of the mdp import
# graph: velocity_command -> rewards, curriculums -> velocity_command).
# Linear matches the project-wide 0.05 stand gate; angular 0.1 matches the
# terrain curriculum's min_cmd_norm-scale guard band for near-zero yaw
# commands.
_RATIO_LIN_CMD_MIN = 0.05
_RATIO_ANG_CMD_MIN = 0.1


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_linear_velocity_adaptive(
  env: ManagerBasedRlEnv,
  std: float,
  std_gain: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """track_linear_velocity with sigma scaled by command magnitude.

  A fixed sigma cannot serve both ends of the command range: tight makes
  large commands flatline exp() at the initial error (no gradient, the
  policy gives up); loose makes small commands nearly free to ignore.
  sigma_eff = std + std_gain * |cmd_xy| keeps both regimes shaped.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  std_eff = std + std_gain * torch.norm(command[:, :2], dim=1)
  return torch.exp(-lin_vel_error / std_eff**2)


def track_linear_velocity_relative(
  env: ManagerBasedRlEnv,
  rel: float,
  std_min: float,
  std_max: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """track_linear_velocity with sigma PROPORTIONAL to the command magnitude.

  XGOLite-Precision (2026-07-11 speed-accuracy analysis, R2): a fixed sigma
  prices tracking error in absolute m/s, so the same RELATIVE shortfall is
  ~10x cheaper at cmd 0.08 than at cmd 0.30 — the measured v17 slow-band
  deficits (vx 0.08 -> 0.059) forfeited only ~0.065 weight-units. A sigma
  proportional to ||cmd|| prices relative error uniformly across the range:

    sigma_eff = clip(rel * ||cmd_xy||, std_min, std_max)

  and for ||cmd_xy|| < 0.05 (standing / pure rotation, the STAND_CMD_NORM
  gate) sigma_eff = std_max so stand behavior keeps the v17 shaping.
  r = exp(-(||cmd_xy - v_xy||^2 + 2 vz^2) / sigma_eff^2). The additive-sigma
  ``track_linear_velocity_adaptive`` above solves the opposite problem
  (keeping a gradient at SPRINT commands); this one tightens the slow band.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  cmd_norm = torch.norm(command[:, :2], dim=1)
  std_eff = torch.clamp(rel * cmd_norm, min=std_min, max=std_max)
  std_eff = torch.where(cmd_norm < 0.05, torch.full_like(std_eff, std_max), std_eff)
  return torch.exp(-lin_vel_error / std_eff**2)


def track_angular_velocity_adaptive(
  env: ManagerBasedRlEnv,
  std: float,
  std_gain: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """track_angular_velocity with sigma scaled by |commanded yaw rate|."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  std_eff = std + std_gain * command[:, 2].abs()
  return torch.exp(-ang_vel_error / std_eff**2)


def track_lateral_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Tight-sigma reward on the lateral (body-y) velocity component only.

  v13b lateral fix: with the adaptive sigma (std 0.15 + 0.5*|cmd|), standing
  still already earns ~84% of the tracking reward at vy=0.08, so lateral
  asymptoted at ~0.012 m/s of the ~0.045 physical ceiling. A dedicated tight
  sigma restores the gradient (standing -> 0.08, full capability -> 0.61).
  Always active (no command gate).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  vy_error = torch.square(command[:, 1] - asset.data.root_link_lin_vel_b[:, 1])
  return torch.exp(-vy_error / std**2)


def track_yaw_zero(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Tight-sigma reward on zero body yaw rate when yaw is uncommanded.

  v16 drift fix (exact mirror of the v13b lateral fix): at wz_cmd = 0 the
  adaptive angular sigma is 0.2 rad/s and body_ang_vel penalizes xy only,
  so the measured 0.05-0.07 rad/s parasitic yaw under vx load cost only
  ~6-12% of one term. This term makes that band expensive: at std 0.07 a
  0.05 rad/s bias forfeits ~40% of the reward. Gated to |wz_cmd| below the
  threshold so commanded turning is untouched.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  wz = asset.data.root_link_ang_vel_b[:, 2]
  gate = (command[:, 2].abs() < command_threshold).float()
  return torch.exp(-torch.square(wz) / std**2) * gate


def velocity_progress_lin(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Directional velocity-progress ratio on the commanded xy twist (V21B).

  Anti-idle term (v21b-v4 postmortem): on the terrain task, standing still
  is the dominant local optimum — the relative-sigma tracking family pays
  generous partial credit at ZERO velocity for slow commands (measured
  0.489 raw standing at cmd 0.08, ~0.42 mean over the seed band) and
  ``stand_still`` only applies at |cmd| <= 0.1, so nothing priced idling
  under a move command. This term pays for MOVING in the commanded
  direction and nothing else:

    r = clip(dot(v_xy_b, cmd_xy) / max(|cmd_xy|^2, eps), 0, 1)

  i.e. the achieved fraction of the commanded velocity along the command:
  standing scores 0, reverse motion clips to 0, overshoot saturates at 1
  (no incentive to outrun the command — precision stays priced by the
  tracking terms). Envs below the ``_RATIO_LIN_CMD_MIN`` stand gate score 0
  (standing episodes carry no progress signal). Same per-step math as the
  grid unlock ratio gate's episode mean (``command_grid_adaptive``), same
  velocity source as the tracking rewards (base-frame root linear velocity).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_xy = command[:, :2]
  v_xy = asset.data.root_link_lin_vel_b[:, :2]
  cmd_sq = torch.sum(cmd_xy * cmd_xy, dim=1)
  ratio = torch.clamp(
    torch.sum(v_xy * cmd_xy, dim=1) / cmd_sq.clamp(min=1e-12), 0.0, 1.0
  )
  return ratio * (cmd_sq >= _RATIO_LIN_CMD_MIN**2).float()


def velocity_progress_ang(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``velocity_progress_lin`` for the commanded yaw rate (V21B).

    r = clip(wz_b * cmd_wz / max(cmd_wz^2, eps), 0, 1)

  zeroed for |cmd_wz| < ``_RATIO_ANG_CMD_MIN`` (0.1). Directional: rotating
  against the command scores 0; overshoot saturates at 1. Same yaw-rate
  source as the angular tracking rewards (base-frame root angular velocity).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_wz = command[:, 2]
  wz = asset.data.root_link_ang_vel_b[:, 2]
  ratio = torch.clamp(
    wz * cmd_wz / (cmd_wz * cmd_wz).clamp(min=1e-12), 0.0, 1.0
  )
  return ratio * (cmd_wz.abs() >= _RATIO_ANG_CMD_MIN).float()


def body_pitch_from_gravity(projected_gravity_b: torch.Tensor) -> torch.Tensor:
  """Body pitch about the body Y axis from projected gravity, in rad.

  Sign convention (v14 command contract): positive = nose UP. For a body
  pitched nose-down by theta, projected gravity in the base frame is
  (sin(theta), 0, -cos(theta)), so atan2(-g_x, -g_z) = -theta.
  """
  return torch.atan2(-projected_gravity_b[:, 0], -projected_gravity_b[:, 2])


def track_body_pitch(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking the commanded body pitch (command channel 3)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  pitch_meas = body_pitch_from_gravity(asset.data.projected_gravity_b)
  pitch_error = torch.square(command[:, 3] - pitch_meas)
  return torch.exp(-pitch_error / std**2)


def track_base_height(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking the commanded base height (command channel 4).

  Uses absolute root z: valid on flat terrain only (floor plane at z=0).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  height_error = torch.square(command[:, 4] - asset.data.root_link_pos_w[:, 2])
  return torch.exp(-height_error / std**2)


def terrain_height_under_base(
  env: ManagerBasedRlEnv, sensor_name: str
) -> torch.Tensor:
  """Mean terrain height (world z) under the base from the terrain scan.

  Averages the ray-hit z over all valid rays of the base-centered grid scan
  (V21B: 0.32 x 0.20 m, comparable to the support polygon), which reads as
  the local ground plane height under the robot — exact on flat/rough tiles,
  the mid-slope height on ramps. Rays that miss (distance < 0) are excluded;
  if every ray misses (cannot happen over generator terrain, kept for
  robustness) the flat-world floor z=0 is returned. [B]
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  hit_z = sensor.data.hit_pos_w[..., 2]  # [B, N]
  valid = sensor.data.distances >= 0  # [B, N]
  count = valid.sum(dim=1)
  mean_z = (hit_z * valid).sum(dim=1) / count.clamp(min=1)
  return torch.where(count > 0, mean_z, torch.zeros_like(mean_z))


def terrain_height_under_feet(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  foot_pos_w: torch.Tensor,
  foot_exclusion_radius: float = 0.015,
) -> torch.Tensor:
  """Terrain height (world z) under each foot from the base terrain scan.

  Nearest-valid-ray lookup: for each foot, the terrain height is the hit z
  of the closest scan ray in the xy plane. Rays whose hit lands within
  ``foot_exclusion_radius`` (xy) of ANY foot center are excluded first —
  rays from the base can strike a foot top (pad radius 6 mm, ray spacing
  20 mm), and using such a hit as "ground" would zero the measured
  clearance of a swinging foot. Feet with no valid ray fall back to the
  scan-mean ground height under the base.

  foot_pos_w: [B, F, 3] world foot positions. Returns [B, F].
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  hit = sensor.data.hit_pos_w  # [B, N, 3]
  valid = sensor.data.distances >= 0  # [B, N]

  # xy distance of every ray hit to every foot: [B, F, N].
  d2 = torch.sum(
    torch.square(hit[:, None, :, :2] - foot_pos_w[:, :, None, :2]), dim=-1
  )
  on_any_foot = (d2 < foot_exclusion_radius**2).any(dim=1)  # [B, N]
  usable = valid & ~on_any_foot  # [B, N]

  big = torch.finfo(d2.dtype).max
  d2_masked = torch.where(usable.unsqueeze(1), d2, torch.full_like(d2, big))
  nearest = torch.argmin(d2_masked, dim=-1)  # [B, F]
  terrain_z = torch.gather(hit[..., 2], 1, nearest)  # [B, F]

  has_ray = usable.any(dim=1, keepdim=True).expand_as(terrain_z)
  fallback = terrain_height_under_base(env, sensor_name).unsqueeze(1)
  return torch.where(has_ray, terrain_z, fallback)


def track_base_height_terrain(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``track_base_height`` made terrain-relative (V21B).

  The flat-task term compares the commanded base height (command channel 4)
  against ABSOLUTE root z, which is wrong the moment env origins leave z=0
  (generator terrain rows sit at arbitrary heights). Here the height is
  measured relative to the local terrain surface under the robot (scan-mean
  ground height from the terrain_scan raycast) — the same "root z above the
  floor" contract the command was calibrated on, evaluated against the
  floor that is actually underfoot. Flat tasks are untouched (they keep
  ``track_base_height``).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  rel_height = (
    asset.data.root_link_pos_w[:, 2] - terrain_height_under_base(env, sensor_name)
  )
  height_error = torch.square(command[:, 4] - rel_height)
  return torch.exp(-height_error / std**2)


def feet_clearance_terrain(
  env: ManagerBasedRlEnv,
  target_height: float,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  foot_exclusion_radius: float = 0.015,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``feet_clearance`` made terrain-relative (V21B).

  The flat-task term prices |foot_z - target| in ABSOLUTE world z (floor at
  z=0 assumed); on generator terrain that misprices every foot by the local
  ground height. Here each foot's height is measured relative to the
  terrain under that foot (nearest-ray lookup on the terrain_scan, see
  ``terrain_height_under_feet``). Same cost form as upstream: per-foot
  |rel_height - target| weighted by the foot's xy speed, summed, gated on
  the commanded twist. Flat tasks keep ``feet_clearance``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids]  # [B, F, 3]
  terrain_z = terrain_height_under_feet(
    env, sensor_name, foot_pos, foot_exclusion_radius
  )  # [B, F]
  rel_height = foot_pos[..., 2] - terrain_z  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  delta = torch.abs(rel_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def body_orientation_cmd_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """body_orientation_l2 made pitch-command-aware (v14).

  The flat-orientation penalty would fight every non-zero pitch command, so
  pitch is penalized toward the COMMANDED pitch (command channel 3) while
  roll is still penalized toward 0. For small angles the magnitude matches
  the upstream term (pitch_err^2 + g_y^2 ~ pitch_err^2 + roll^2), so the
  existing weight carries over.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  # Same body_ids branch as body_orientation_l2.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
  else:
    projected_gravity_b = asset.data.projected_gravity_b

  pitch_meas = body_pitch_from_gravity(projected_gravity_b)
  pitch_err_sq = torch.square(command[:, 3] - pitch_meas)
  roll_sq = torch.square(projected_gravity_b[:, 1])
  return pitch_err_sq + roll_sq


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            # Gate on the twist slice (indices 0..2) only; the v14 pose
            # channels must not count toward the "is moving" norm.
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


def _scheduled_phase_term(env: ManagerBasedRlEnv):
  """The actor's scheduled phase-clock obs term (single source of truth).

  The V21A gait rewards must score contacts against the SAME per-env phase
  the policy observes; duplicating the accumulator here would desynchronize
  on resets (the obs term is computed once more per episode, for the
  initial observation). Reading the actor instance also gives the correct
  temporal pairing: at reward time the buffer holds the phase the policy
  saw when it produced the action being scored.
  """
  from .observations import phase_scheduled

  func = env.observation_manager.get_term_cfg("actor", "phase").func
  if not isinstance(func, phase_scheduled):
    raise TypeError(
      "V21A gait rewards require the actor 'phase' obs term to be "
      f"mdp.observations.phase_scheduled, got {type(func).__name__}."
    )
  return func


def gait_member_offsets(
  command: torch.Tensor,
  trot_offsets: tuple[float, ...],
  walk_offsets: tuple[float, ...],
  trot_duty: float,
  walk_duty: float,
  lat_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Per-env gait member from the commanded twist (V21A deploy-parity rule).

  Lateral-dominant command (|vy| > |vx| AND |vy| > lat_threshold) selects
  the lateral-sequence WALK member (4-beat, duty > 0.5); everything else
  keeps the diagonal trot (duty 0.5). Deterministic in the command the
  policy already observes — no new command dims. Returns
  (offsets [B, 4], duty [B], walk_mask [B]).
  """
  device, dtype = command.device, command.dtype
  walk_mask = (command[:, 1].abs() > command[:, 0].abs()) & (
    command[:, 1].abs() > lat_threshold
  )
  trot_t = torch.tensor(trot_offsets, device=device, dtype=dtype)
  walk_t = torch.tensor(walk_offsets, device=device, dtype=dtype)
  offsets = torch.where(
    walk_mask.unsqueeze(1), walk_t.unsqueeze(0), trot_t.unsqueeze(0)
  )
  duty = torch.where(
    walk_mask,
    torch.full_like(command[:, 0], walk_duty),
    torch.full_like(command[:, 0], trot_duty),
  )
  return offsets, duty, walk_mask


def orc_phase_weight(leg_phase: torch.Tensor, duty: torch.Tensor) -> torch.Tensor:
  """Smooth stance/swing weight for the ORC-family contact reward.

  Duty-aware member of the ``-F_GRF * sin(phase)`` family (Zhang/Heim/Jeon/
  Kim, arXiv:2402.08662): the leg phase (stance-first convention, stance =
  [0, duty)) is warped so stance occupies [0, 0.5) and swing [0.5, 1), then
  weighted by sin(2*pi*warped) — +1 at mid-stance, -1 at mid-swing, 0 at
  the touchdown/liftoff transitions. Multiplied by a saturated GRF this
  rewards force carried in stance and penalizes force in swing.

  leg_phase: [B, N] in [0, 1); duty: [B] or [B, 1] in (0, 1).
  """
  if duty.dim() == 1:
    duty = duty.unsqueeze(1)
  in_stance = leg_phase < duty
  warped = torch.where(
    in_stance,
    0.5 * leg_phase / duty,
    0.5 + 0.5 * (leg_phase - duty) / (1.0 - duty),
  )
  return torch.sin(2.0 * torch.pi * warped)


def orc_contact_reward(
  base_phase: torch.Tensor,
  offsets: torch.Tensor,
  duty: torch.Tensor,
  force_norm: torch.Tensor,
  force_scale: float,
) -> torch.Tensor:
  """Pure ORC contact score (unit-testable core of ``gait_contact_orc``).

  base_phase [B], offsets [B, N], duty [B], force_norm [B, N] (N per-foot
  GRF magnitudes). Force is normalized by ``force_scale`` and saturated at
  1 — a graded signal, not the boolean contact match of ``feet_gait``.
  Returns the per-env mean over feet, in [-1, 1].
  """
  leg_phase = (base_phase.unsqueeze(1) + offsets) % 1.0
  f_sat = (force_norm / force_scale).clamp(0.0, 1.0)
  return (orc_phase_weight(leg_phase, duty) * f_sat).mean(dim=1)


def gait_contact_orc(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  trot_offsets: tuple[float, ...],
  walk_offsets: tuple[float, ...],
  trot_duty: float,
  walk_duty: float,
  lat_threshold: float,
  force_scale: float,
  command_threshold: float,
) -> torch.Tensor:
  """ORC-style phase-contact reward (replaces ``feet_gait`` in V21A).

  arXiv:2402.08662's ORC ablation: a phase OBSERVATION alone yields
  "unbalanced gaits (2-3 legs)" (the FreeGait rear-shuffle); balanced
  4-leg use emerges only with the phase-based CONTACT reward on. This is
  that reward, on the scheduled per-env clock: saturated per-foot GRF
  weighted by the stance/swing phase weight (see ``orc_phase_weight``),
  with per-env offsets/duty from the speed/direction gait rule
  (``gait_member_offsets``). Gated off below the same 0.05 twist norm as
  ``feet_gait`` (which is left untouched — other tasks use it).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force_norm = torch.norm(sensor.data.force, dim=-1)  # [B, N]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  offsets, duty, _ = gait_member_offsets(
    command, trot_offsets, walk_offsets, trot_duty, walk_duty, lat_threshold
  )
  base_phase = _scheduled_phase_term(env).phase
  reward = orc_contact_reward(base_phase, offsets, duty, force_norm, force_scale)
  # Same gate as feet_gait: twist slice only, linear norm + |wz|.
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  active = ((linear_norm + angular_norm) > command_threshold).float()
  return reward * active


class gait_lr_symmetry:
  """Left-right morphological-symmetry gait reward (V21A).

  Ding et al. (arXiv:2403.10723, 50 Hz Go2): dropping the morphological-
  symmetry term raises gait-consistency error 0.2 -> 0.4 and energy
  30-50% ("prevents limping"); Su et al. (arXiv:2403.17320): symmetry
  methods give better gait quality than an unconstrained baseline. V21A
  disables the HARD mirror loss/augmentation (they assume the trot's
  (sin,cos) -> (-sin,-cos) phase equivalence, invalid for the walk
  member), so this REWARD supplies the left-right symmetry pressure.

  Form: in both V21A gait members the left-right leg pairs are exactly
  half a gait cycle apart (trot fl-fr = 0.5; walk fl-fr = bl-br = 0.5),
  so the mirrored joint vector from half a period ago should equal the
  current one: r = exp(-mean((q_t - P q_{t-T/2})^2) / std^2), with P the
  fl<->fr / bl<->br permutation (sign +1: the generated XML uses mirrored
  hip axes — see rl/symmetry.py). The lag T/2 = 0.5 / f(cmd) follows the
  scheduled clock per env. Gated off while standing (frozen clock) and
  until the ring buffer holds a half period; after a mid-dwell resample
  the lag uses the new frequency immediately (transient inaccuracy of at
  most half a cycle, negligible over a 3-8 s dwell).
  """

  # (fl, fr, bl, br) x (hip, thigh, calf) model order; mirror = swap
  # fl<->fr, bl<->br with unchanged joint angles (rl/symmetry.py).
  _JOINT_PERM = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    joint_ids = cfg.params["asset_cfg"].joint_ids
    n_joints = asset.data.joint_pos[:, joint_ids].shape[1]
    if n_joints != len(self._JOINT_PERM):
      raise ValueError(
        f"gait_lr_symmetry expects {len(self._JOINT_PERM)} leg joints, "
        f"got {n_joints}."
      )
    self._capacity = int(cfg.params.get("buffer_capacity", 32))
    self._buf = torch.zeros(
      env.num_envs, self._capacity, n_joints, device=env.device
    )
    self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._ptr = 0
    self._perm = torch.tensor(
      self._JOINT_PERM, dtype=torch.long, device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._count[env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
    buffer_capacity: int = 32,
  ) -> torch.Tensor:
    del buffer_capacity  # Consumed in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    self._buf[:, self._ptr] = q
    self._count += 1

    freq = _scheduled_phase_term(env).freq  # 0 while the clock is frozen.
    lag = torch.round(0.5 / (freq.clamp(min=1e-6) * env.step_dt)).long()
    lag = lag.clamp(1, self._capacity - 1)
    valid = (freq > 0.0) & (self._count > lag)

    idx = (self._ptr - lag) % self._capacity
    q_past = self._buf[torch.arange(env.num_envs, device=env.device), idx]
    err = torch.mean(torch.square(q - q_past[:, self._perm]), dim=1)
    reward = torch.exp(-err / std**2)

    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    active = (linear_norm + angular_norm) > command_threshold

    self._ptr = (self._ptr + 1) % self._capacity
    return reward * (valid & active).float()


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def _pose_command_deviation(
  command: torch.Tensor,
  nominal_pose: tuple[float, float],
  height_scale: float,
) -> torch.Tensor:
  """Rad-equivalent deviation of the pose command from the nominal pose.

  dev = |pitch_cmd - pitch_nom| + height_scale * |h_cmd - h_nom|. The height
  scale converts meters to a pitch-comparable magnitude (the full +/-0.019 m
  height range maps to ~0.38 at scale 20, close to the 0.436 max pitch).
  """
  return (command[:, 3] - nominal_pose[0]).abs() + height_scale * (
    command[:, 4] - nominal_pose[1]
  ).abs()


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
    pose_relax_gain: float = 0.0,
    pose_dev_height_scale: float = 20.0,
    nominal_pose: tuple[float, float] | None = None,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    # Speed regime from the twist slice [:3] only (indices 0..2): with the
    # v14 pose channels appended, a full-vector norm (height ~0.116 always
    # present) would never classify an env as standing.
    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    if pose_relax_gain > 0.0 and command.shape[1] >= 5:
      # v14 conflict fix: the stand-default joint targets are wrong for a
      # commanded crouch/pitch, and at std_standing tightness this term would
      # fight the pose-tracking rewards. Scaling the stds up with the pose
      # command's deviation from nominal (rather than gating the term off)
      # keeps the smoothness/symmetry regularization while letting the legs
      # move as far as the commanded pose requires: at max pitch (0.436 rad,
      # gain 8) stds grow ~4.5x, turning a hard prior into a loose one.
      assert nominal_pose is not None, "pose_relax_gain > 0 needs nominal_pose."
      dev = _pose_command_deviation(command, nominal_pose, pose_dev_height_scale)
      std = std * (1.0 + pose_relax_gain * dev.unsqueeze(1))

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        pose_dev_threshold: float | None = None,
        pose_dev_height_scale: float = 20.0,
        nominal_pose: tuple[float, float] | None = None,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            if pose_dev_threshold is not None and command.shape[1] >= 5:
                # v14 conflict fix: this term drags joints to the stand
                # default whenever the twist is small -- which includes every
                # pose_hold episode, where the whole point is to HOLD a
                # non-default crouch. Only apply it when the commanded pose
                # is (near-)nominal; posed standing is regularized by the
                # relaxed `pose` term and action_rate instead.
                assert nominal_pose is not None, (
                    "pose_dev_threshold needs nominal_pose."
                )
                dev = _pose_command_deviation(
                    command, nominal_pose, pose_dev_height_scale
                )
                scale = scale * (dev < pose_dev_threshold).float()
            reward *= scale
    return reward


class joint_acc_control_rate_l2:
  """``joint_acc_l2`` measured at CONTROL rate, not physics rate.

  v19 v1/v2 postmortem (2026-07-14): the measured servo plant (kp 39.7,
  kd 0.007, damping 0.011, armature 0, torque clamp +/-0.22 N*m) is a
  stiff torque-saturated relay with almost no dissipation. At the 500 Hz
  physics rate it dithers in a sub-milliradian limit cycle around the
  target (the real servo's audible buzz; force sign flips on ~19% of
  control steps, instantaneous qacc RMS ~900 rad/s^2 while standing
  STILL). That dither is invisible at the 100 Hz telemetry the Stage-1
  fit scored — physically-damped parameter variants replay the captures
  3x WORSE, so the fit is right at gait frequencies and unconstrained
  above them.

  mjlab's ``joint_acc_l2`` reads the instantaneous ``qacc`` of the last
  physics substep, so on this plant it taxes the policy ~-0.8/step for a
  phenomenon it cannot influence; with is_terminated at -200 that made
  dying (+bootstrap 0) cheaper than living and both v19 runs collapsed
  into the 5-step suicide attractor (Episode_Reward/joint_acc_l2 was
  -660 of the -704 mean episode reward at v2 iter 50).

  Fix: finite-difference the joint velocity across CONTROL steps (50 Hz)
  — the same acceleration the old soft plant effectively exposed and the
  same signal the deployed observation stack could ever see. On smooth
  plants this equals qacc; under physics-rate dither the >25 Hz content
  aliases down bounded by the dither's velocity amplitude (~2 orders of
  magnitude smaller penalty) instead of scaling with the 500 Hz rate.

  First step after an env reset returns 0 (no previous velocity).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._prev_qd: torch.Tensor | None = None
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
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if self._prev_qd is None:
      self._prev_qd = qd.clone()
    acc = (qd - self._prev_qd) / env.step_dt
    self._prev_qd = qd.clone()
    cost = torch.sum(torch.square(acc), dim=1) * self._valid.float()
    self._valid[:] = True
    return cost

