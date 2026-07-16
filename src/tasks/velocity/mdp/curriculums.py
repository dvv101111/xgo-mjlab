from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

# Axis "commanded" thresholds for the achieved-velocity ratio gate
# (GridCurriculumCfg.unlock_min_vel_ratio) — single source shared with the
# v21b velocity-progress rewards; defined in rewards.py (the bottom of the
# mdp import graph) to avoid an import cycle through velocity_command.
from .rewards import _RATIO_ANG_CMD_MIN, _RATIO_LIN_CMD_MIN  # noqa: F401
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


def _episode_vel_ratios(
  command_term: UniformVelocityCommand,
  env_ids: torch.Tensor,
  steps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Per-episode directional achieved/commanded velocity ratios.

  Single source for the ratio math shared by the grid unlock AND-gate
  (``command_grid_adaptive``) and the terrain curriculum's ratio gates
  (``terrain_levels_reward_gated``); the v21b progress rewards mirror the
  same expression per step (rewards.py). Episode-mean achieved velocity =
  the command term's achieved sums / elapsed ``steps`` (the accumulators
  are still un-zeroed at ``_reset_idx`` time):

    lin ratio = clip(dot(mean_v_xy, cmd_xy) / |cmd_xy|^2, 0, 1)
    ang ratio = clip(mean_wz * cmd_wz / cmd_wz^2, 0, 1)

  Directional: reverse motion scores 0; overshoot saturates at 1. Returns
  (lin_ratio, ang_ratio, lin_commanded, ang_commanded) with the commanded
  masks at the shared axis thresholds (|cmd_xy| >= 0.05, |cmd_wz| >= 0.1).
  """
  mean_lin = command_term.grid_achieved_lin_sum[env_ids] / steps.unsqueeze(-1)
  mean_ang = command_term.grid_achieved_ang_sum[env_ids] / steps
  cmd = command_term.vel_command_b[env_ids]
  cmd_xy, cmd_wz = cmd[:, :2], cmd[:, 2]
  lin_sq = (cmd_xy * cmd_xy).sum(dim=-1)
  lin_commanded = lin_sq >= _RATIO_LIN_CMD_MIN**2
  ang_commanded = cmd_wz.abs() >= _RATIO_ANG_CMD_MIN
  lin_ratio = torch.clamp(
    (mean_lin * cmd_xy).sum(dim=-1) / lin_sq.clamp(min=1e-12), 0.0, 1.0
  )
  ang_ratio = torch.clamp(
    mean_ang * cmd_wz / (cmd_wz * cmd_wz).clamp(min=1e-12), 0.0, 1.0
  )
  return lin_ratio, ang_ratio, lin_commanded, ang_commanded


def terrain_levels_reward_gated(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
  lin_reward_name: str = "track_linear_velocity",
  ang_reward_name: str = "track_angular_velocity",
  gamma_lin: float = 0.70,
  gamma_ang: float = 0.55,
  demote_frac: float = 0.5,
  min_cmd_norm: float = 0.1,
  min_episode_frac: float = 0.5,
  promote_min_vel_ratio: float | None = None,
  demote_below_vel_ratio: float | None = None,
) -> torch.Tensor:
  """Reward-gated terrain-level curriculum (V21B).

  ``terrain_levels_vel`` (above) promotes on DISTANCE walked (> half a
  terrain tile) and demotes when distance < 0.5 * ||cmd|| * T. Both rules
  fail at 577 g scale and small commands (lit review section 1.5): the
  half-tile crossing is calibrated to mid-size robots, and the
  velocity-scaled demotion silently stalls for low-speed bins (the Isaac
  Lab #969/#1492/#1685 coupling bug). HIM (arXiv:2312.11460) replaces the
  promotion gate with a fraction of the velocity-tracking REWARD; this term
  transplants that rule onto the same episodic tracking fractions the grid
  curriculum already gates on (``command_grid_adaptive``: episode reward
  sum / (elapsed_steps * weight * dt), in (0, 1]):

  - PROMOTE an env when frac_lin >= gamma_lin AND frac_ang >= gamma_ang
    (defaults = the v18/v20-calibrated competent-episode stochastic means,
    the same 0.70/0.55 the command grid unlocks on) AND the episode lasted
    at least ``min_episode_frac`` of the max length (an env that tracked
    for a second and then fell must not move up).
  - DEMOTE an env when frac_lin < gamma_lin * demote_frac — clearly
    failing on its current terrain — but ONLY when its commanded twist
    norm was at least ``min_cmd_norm`` (the low-speed demotion guard:
    standing / near-zero-command episodes carry no evidence about terrain
    difficulty, so they HOLD; this is the explicit fix for the coupling
    bug above, and it also covers the stand-injection episodes whose
    command the sampler zeroes).

  Composition with the command grid: both terms run from the same
  ``_reset_idx`` hook and read the same (still-unzeroed) episode sums;
  this one only moves ``terrain.env_origins`` while the grid only moves
  command-cell weights — no shared state. update_env_origins keeps the
  legged_gym anti-forgetting rule (envs promoted past the top row are
  reassigned to a random row).

  RATIO GATES (opt-in, v21b-v5 postmortem): the frac gates above score
  relative-sigma REWARD fractions, on which STANDING at slow commands
  (~0.49 partial credit) outscores honest ratio-0.5-0.6 walking (~0.26).
  In the v5 run the honest walker never passed gamma_lin and the frac
  demote gate (0.70 * 0.5 = 0.35 > its ~0.26) demoted every env to row 0
  by iter 500 — pinning the terrain-level-coupled hazard severity at
  s ~ 0 forever. When ``promote_min_vel_ratio``/``demote_below_vel_ratio``
  are set (both together; v21b wires 0.5/0.25), episodes that command at
  least one twist axis are judged on the same honest directional
  achieved/commanded velocity ratio as the grid unlock gate and the
  progress rewards (``_episode_vel_ratios``; needs the command term's
  achieved accumulators, i.e. ``grid_curriculum.unlock_min_vel_ratio``):

  - PROMOTE: ``moving`` and ``long_enough`` (guards unchanged) AND ratio
    >= ``promote_min_vel_ratio`` on EVERY commanded axis (lin gated at
    |cmd_xy| >= 0.05, ang at |cmd_wz| >= 0.1).
  - DEMOTE: ``moving`` AND ratio < ``demote_below_vel_ratio`` on ANY
    commanded axis. The frac demote is NOT kept alongside: its threshold
    (gamma_lin * demote_frac = 0.35) sits ABOVE an honest ratio-0.5
    walker's frac (~0.26), so any frac-based demotion would keep
    demoting exactly the policies this recalibration protects.
  - Episodes whose twist norm passes ``min_cmd_norm`` but command NEITHER
    axis above its ratio threshold keep the frac gates (stand-adjacent
    cells behave as before); true stand episodes still HOLD via the
    ``moving`` guard.

  Defaults ``None`` keep the frac-gated behavior bit-identical.

  Returns the mean terrain level (logged as Curriculum/terrain_levels).
  """
  if (promote_min_vel_ratio is None) != (demote_below_vel_ratio is None):
    raise ValueError(
      "promote_min_vel_ratio and demote_below_vel_ratio must be set together."
    )
  terrain = env.scene.terrain
  assert terrain is not None
  assert terrain.cfg.terrain_generator is not None, (
    "terrain_levels_reward_gated requires generator terrain."
  )

  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]

  if len(env_ids) > 0:
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    reward_manager = env.reward_manager
    dt_scale = env.step_dt if reward_manager._scale_by_dt else 1.0
    steps = env.episode_length_buf[env_ids].clamp(min=1).float()

    def tracking_fraction(name: str) -> torch.Tensor:
      weight = reward_manager.get_term_cfg(name).weight
      assert weight > 0.0, f"tracking term '{name}' must have positive weight."
      return reward_manager._episode_sums[name][env_ids] / (steps * weight * dt_scale)

    frac_lin = tracking_fraction(lin_reward_name)
    frac_ang = tracking_fraction(ang_reward_name)

    # Twist slice [:3] only (v14 pose channels must not read as "moving").
    cmd_norm = (
      torch.norm(command[env_ids, :2], dim=1) + command[env_ids, 2].abs()
    )
    moving = cmd_norm >= min_cmd_norm
    long_enough = steps >= min_episode_frac * env.max_episode_length

    frac_up = (frac_lin >= gamma_lin) & (frac_ang >= gamma_ang)
    frac_down = frac_lin < gamma_lin * demote_frac

    if promote_min_vel_ratio is not None:
      assert demote_below_vel_ratio is not None
      command_term = env.command_manager.get_term(command_name)
      assert isinstance(command_term, UniformVelocityCommand)
      assert command_term.grid_ratio_enabled, (
        "terrain ratio gating reads the achieved-velocity accumulators; "
        "set grid_curriculum.unlock_min_vel_ratio on the command."
      )
      lin_ratio, ang_ratio, lin_cmd, ang_cmd = _episode_vel_ratios(
        command_term, env_ids, steps
      )
      evaluable = lin_cmd | ang_cmd
      ratio_up = (~lin_cmd | (lin_ratio >= promote_min_vel_ratio)) & (
        ~ang_cmd | (ang_ratio >= promote_min_vel_ratio)
      )
      ratio_down = (lin_cmd & (lin_ratio < demote_below_vel_ratio)) | (
        ang_cmd & (ang_ratio < demote_below_vel_ratio)
      )
      up_signal = torch.where(evaluable, ratio_up, frac_up)
      down_signal = torch.where(evaluable, ratio_down, frac_down)
    else:
      up_signal, down_signal = frac_up, frac_down

    move_up = moving & long_enough & up_signal
    move_down = moving & down_signal & ~move_up

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

  When the command cfg sets ``grid_curriculum.unlock_min_vel_ratio`` (opt-in,
  v21b), a passing episode's cell unlock is ADDITIONALLY gated on the cell's
  mean achieved/commanded velocity ratio — see the inline block below and
  the ``GridCurriculumCfg`` field docstring. With the knob at ``None`` this
  function is bit-identical to the pre-ratio code.
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

    if command_term.grid_ratio_enabled and bool(passed.any()):
      # Achieved-velocity AND-gate (opt-in, GridCurriculumCfg.
      # unlock_min_vel_ratio; the v21b idle-inversion fix): the frac gates
      # score REWARD fractions, which idle episodes can pass on cells nobody
      # tracks — additionally require the cell's episodes to actually MOVE
      # in the commanded direction. Per episode (episode-mean achieved
      # velocity = the term's achieved sums / elapsed steps, same
      # denominator as the frac):
      #   lin ratio = dot(mean_v_xy, cmd_xy) / |cmd_xy|^2   (|cmd_xy|>=0.05)
      #   ang ratio = mean_wz * cmd_wz / cmd_wz^2           (|cmd_wz|>=0.1)
      # both clipped to [0, 1] (reverse motion scores 0). Ratios are
      # averaged PER CELL over all valid episodes attributed to the cell in
      # this reset batch; a cell must clear the threshold on every
      # commanded axis (episodes below an axis's command threshold don't
      # contribute to that axis; a cell with no commanded episodes on an
      # axis is exempt there, so stand cells stay frac-gated only).
      gc = command_term.cfg.grid_curriculum
      assert gc is not None and gc.unlock_min_vel_ratio is not None
      threshold = gc.unlock_min_vel_ratio
      lin_ratio, ang_ratio, lin_cmd_mask, ang_cmd_mask = _episode_vel_ratios(
        command_term, env_ids, steps
      )
      lin_commanded = valid & lin_cmd_mask
      ang_commanded = valid & ang_cmd_mask

      n_cells = n_vx * n_wz

      def cell_axis_ok(axis_mask: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        """Per-cell: mean ratio over the axis's episodes >= threshold, or
        no such episodes (axis not commanded on this cell -> exempt)."""
        idx = cells[axis_mask]
        r = ratio[axis_mask]
        total = torch.zeros(n_cells, device=ratio.device, dtype=ratio.dtype)
        count = torch.zeros(n_cells, device=ratio.device, dtype=ratio.dtype)
        total.index_add_(0, idx, r)
        count.index_add_(0, idx, torch.ones_like(r))
        return (count == 0) | (total / count.clamp(min=1.0) >= threshold)

      ratio_ok = cell_axis_ok(lin_commanded, lin_ratio) & cell_axis_ok(
        ang_commanded, ang_ratio
      )
      passed = passed & ratio_ok[cells.clamp(min=0)]

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
