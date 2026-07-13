from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def command_grid_adaptive(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
  lin_reward_name: str = "track_linear_velocity",
  ang_reward_name: str = "track_angular_velocity",
  gamma_lin: float = 0.8,
  gamma_ang: float = 0.7,
  weight_step: float = 0.2,
) -> dict[str, torch.Tensor]:
  """Grid-adaptive command curriculum update (Margolis et al., RSS 2022).

  Invoked by the curriculum manager at the TOP of ``_reset_idx`` with the
  resetting env_ids — before the reward manager's episode sums are zeroed,
  before the command manager resamples, and before ``episode_length_buf``
  is cleared — so the ending episode's sums, elapsed steps, last-drawn cell
  and standing flags are all still readable here.

  Episodic tracking fraction (0..1 of the max attainable): mjlab's reward
  manager accumulates ``_episode_sums[name] += raw * weight * dt`` per step
  (``dt`` only when ``scale_rewards_by_dt``, the default), and the tracking
  terms' raw value is ``exp(-err^2/sigma^2) in (0, 1]``, so

      frac = episode_sum / (elapsed_steps * weight * dt)

  with ``elapsed_steps = episode_length_buf[env]`` (the ACTUAL episode
  length — normalizing by max length would punish early terminations twice).
  An env whose episode passes BOTH ``frac_lin >= gamma_lin`` and
  ``frac_ang >= gamma_ang`` adds ``weight_step`` to its command cell and
  the cell's 4-connected neighbors; weights are clipped to [0, 1]. Target
  cells are deduplicated within one call (several passing envs on the same
  cell count once — matches the numpy fancy-indexing semantics of the
  reference RewardThresholdCurriculum).

  Envs attributed to no cell (index -1: twist zeroed by the stand gate /
  pose_hold, or standing envs) never gate unlocks. Multi-resample episodes
  attribute to the last drawn cell (see the sampler).
  """
  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, UniformVelocityCommand)
  assert command_term.grid_enabled, (
    f"command_grid_adaptive requires grid_curriculum on command '{command_name}'."
  )
  weights = command_term.grid_weights
  n_vx, n_wz = command_term._grid_n_vx, command_term._grid_n_wz

  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]

  if len(env_ids) > 0:
    reward_manager = env.reward_manager
    dt_scale = env.step_dt if reward_manager._scale_by_dt else 1.0
    steps = env.episode_length_buf[env_ids].clamp(min=1).float()

    def tracking_fraction(name: str) -> torch.Tensor:
      weight = reward_manager.get_term_cfg(name).weight
      assert weight > 0.0, f"tracking term '{name}' must have positive weight."
      return reward_manager._episode_sums[name][env_ids] / (steps * weight * dt_scale)

    frac_lin = tracking_fraction(lin_reward_name)
    frac_ang = tracking_fraction(ang_reward_name)
    cells = command_term.grid_cell_index[env_ids]
    valid = cells >= 0
    passed = valid & (frac_lin >= gamma_lin) & (frac_ang >= gamma_ang)

    if bool(passed.any()):
      pc = cells[passed]
      ix = torch.div(pc, n_wz, rounding_mode="floor")
      iz = torch.remainder(pc, n_wz)
      targets = []
      for dx, dz in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, nz = ix + dx, iz + dz
        in_bounds = (nx >= 0) & (nx < n_vx) & (nz >= 0) & (nz < n_wz)
        targets.append((nx * n_wz + nz)[in_bounds])
      target_cells = torch.unique(torch.cat(targets))
      flat = weights.reshape(-1)
      flat[target_cells] = torch.clamp(flat[target_cells] + weight_step, 0.0, 1.0)

    # Seed-region mean tracking: min(lin, ang) — the binding unlock metric —
    # over attributable envs whose cell sits in the seed region. Falls back
    # to the previous value when this reset batch has none.
    seed_flat = command_term.grid_seed_mask.reshape(-1)
    in_seed = valid & seed_flat[cells.clamp(min=0)]
    if bool(in_seed.any()):
      command_term.grid_seed_tracking_last = torch.minimum(frac_lin, frac_ang)[
        in_seed
      ].mean().item()

  return {
    "mean_cell_weight": weights.mean(),
    "frac_cells_unlocked": (weights >= 0.5).float().mean(),
    "seed_tracking": torch.tensor(command_term.grid_seed_tracking_last),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
