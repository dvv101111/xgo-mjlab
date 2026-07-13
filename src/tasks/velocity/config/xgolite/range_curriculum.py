"""XGOLite-V18Range: grid-adaptive command curriculum preset (2026-07-12).

Why: XGOLite-Precision collapsed to walking-in-place (zero net velocity at
every linear command) — from-scratch training with a tight tracking
tolerance has no reward gradient. Margolis et al. (IJRR 2024, "Walk These
Ways") show the no-curriculum ablation "converges to jittering in place,
tracking error equal to the command"; their validated fix is a
GRID-ADAPTIVE command curriculum (RewardThresholdCurriculum): start
sampling from a small trackable command region and add probability mass to
4-connected neighbor cells whenever an episode's linear AND angular
tracking both clear a threshold. Rescaled to our envelope, the SLOW band
(0.05-0.15 m/s, the stiction regime) becomes its own set of 0.05 m/s cells
that must individually earn competence instead of being a tail of a
box-uniform draw.

Recipe: ``xgolite_v18draft_env_cfg`` (v17 + torque-speed clamp + per-servo
delay) + grid curriculum over an extended envelope + 8% stand-still
injection. Rewards and PPO identical to v17.

To enable grid mode on any other preset::

    twist = cfg.commands["twist"]                # fork cfg class required
    twist.axis_focus_probs = None                # lottery must be OFF
    twist.slow_vx_prob = 0.0
    twist.fast_vx_prob = 0.0
    twist.turn_at_speed_prob = 0.0
    twist.grid_curriculum = (
      local_mdp.UniformVelocityCommandCfg.GridCurriculumCfg()
    )  # grid tiles ranges.lin_vel_x x ranges.ang_vel_z as set on the cfg
    cfg.curriculum["command_grid"] = CurriculumTermCfg(
      func=local_mdp.command_grid_adaptive,
      params={"command_name": "twist"},
      # If the preset renames/replaces the tracking terms, also pass
      # lin_reward_name/ang_reward_name and thresholds gamma_lin/gamma_ang.
    )
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from src.tasks.velocity import mdp as local_mdp

from .sim_fidelity import xgolite_v18draft_env_cfg


def xgolite_v18range_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = xgolite_v18draft_env_cfg(play=play)
  twist = cfg.commands["twist"]
  assert isinstance(twist, local_mdp.UniformVelocityCommandCfg)

  # The grid owns (vx, wz) exposure: the hand-tuned exclusive lottery
  # (axis_focus / slow_vx) inherited from v17 must be off (the sampler
  # enforces this). Trade-off, documented: v17's 25% pure-lateral focus
  # episodes disappear — vy keeps its independent +-0.08 uniform draw, so
  # near-pure-lateral exposure now comes only from draws in the near-origin
  # cells; pose_hold/standing episodes still train the stand regime.
  twist.axis_focus_probs = None
  twist.slow_vx_prob = 0.0
  twist.fast_vx_prob = 0.0
  twist.turn_at_speed_prob = 0.0

  # Grid envelope (spec): vx extended past v17's honest (-0.40, 0.45) —
  # the top/backward cells carry ZERO initial weight, so the wide range no
  # longer dilutes gradient (the v15 lesson); mass only arrives once the
  # neighboring cells demonstrate competence. wz unchanged. 21 x 8 cells
  # at the 0.05 / 0.25 defaults.
  twist.ranges.lin_vel_x = (-0.45, 0.60)
  twist.ranges.ang_vel_z = (-1.0, 1.0)

  # Defaults: 0.05 m/s x 0.25 rad/s cells, seed = v17's known-trackable
  # band vx [-0.15, 0.25] x wz [-0.5, 0.5].
  twist.grid_curriculum = local_mdp.UniformVelocityCommandCfg.GridCurriculumCfg()

  # Stand-still injection: reuse the existing per-resample standing draw
  # (is_standing_env zeroes the twist slice every step). These episodes are
  # excluded from cell attribution by the sampler.
  twist.rel_standing_envs = 0.08

  if not play:
    # Play cfgs run with an empty curriculum (see xgolite_rough_env_cfg).
    cfg.curriculum["command_grid"] = CurriculumTermCfg(
      func=local_mdp.command_grid_adaptive,
      params={
        "command_name": "twist",
        "lin_reward_name": "track_linear_velocity",
        "ang_reward_name": "track_angular_velocity",
        # v1 shipped (0.8, 0.7) and NEVER unlocked a cell in 2500 iters:
        # under stochastic training rollouts (action noise sigma ~0.16) the
        # per-episode angular fraction MAXES at ~0.695 even for a competent
        # policy — the gait-cycle yaw oscillation of this small trot eats
        # the sigma_eff 0.2 budget — and even v17/v18base pass 0.0% of
        # episodes. Measured distributions (debug_range_env_detail.py,
        # 2026-07-12): competent stochastic lin mean 0.708 / ang mean 0.551;
        # a non-tracking policy scores frac_lin ~0.4 on seed commands, so
        # (0.70, 0.55) keeps ~clean separation while passing 28-50% of
        # competent episodes (Margolis-pace expansion, fits a 2500-iter run
        # with ~60 synchronized reset waves).
        "gamma_lin": 0.70,
        "gamma_ang": 0.55,
        "weight_step": 0.2,
      },
    )

  return cfg
