from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .aggressive import (
  xgolite_aggressive_ppo_runner_cfg,
  xgolite_agile_env_cfg,
  xgolite_fastclock_env_cfg,
  xgolite_freegait_env_cfg,
  xgolite_sprint_env_cfg,
)
from .env_cfgs import xgolite_flat_env_cfg
from .precision import xgolite_precision_env_cfg
from .range_curriculum import xgolite_v18range_env_cfg
from .rl_cfg import xgolite_ppo_runner_cfg
from .sim_fidelity import xgolite_precision2_env_cfg, xgolite_v18draft_env_cfg

register_mjlab_task(
  task_id="XGOLite-Flat",
  env_cfg=xgolite_flat_env_cfg(),
  play_env_cfg=xgolite_flat_env_cfg(play=True),
  rl_cfg=xgolite_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Aggressive-locomotion experiment family (2026-07-11): see aggressive.py
# for the preset table, sim-saturation findings and deploy-side notes.
register_mjlab_task(
  task_id="XGOLite-Sprint",
  env_cfg=xgolite_sprint_env_cfg(),
  play_env_cfg=xgolite_sprint_env_cfg(play=True),
  rl_cfg=xgolite_aggressive_ppo_runner_cfg("xgolite_sprint"),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="XGOLite-FastClock",
  env_cfg=xgolite_fastclock_env_cfg(),
  play_env_cfg=xgolite_fastclock_env_cfg(play=True),
  rl_cfg=xgolite_aggressive_ppo_runner_cfg("xgolite_fastclock"),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="XGOLite-FreeGait",
  env_cfg=xgolite_freegait_env_cfg(),
  play_env_cfg=xgolite_freegait_env_cfg(play=True),
  # Fresh gait discovery converges slower than a resume polish: 3000 iters.
  # Symmetry aug/mirror loss OFF: asymmetric gaits are allowed to emerge.
  rl_cfg=xgolite_aggressive_ppo_runner_cfg(
    "xgolite_freegait", max_iterations=3000, symmetry=False
  ),
  runner_cls=VelocityOnPolicyRunner,
)
# Full-range accuracy preset (2026-07-11 speed-accuracy analysis section 3):
# aligned 0.05 reward gates, relative tracking sigma, stiction/damping DR,
# slow-band + transition sampling. Symmetry ON, fresh run preferred.
register_mjlab_task(
  task_id="XGOLite-Precision",
  env_cfg=xgolite_precision_env_cfg(),
  play_env_cfg=xgolite_precision_env_cfg(play=True),
  rl_cfg=xgolite_aggressive_ppo_runner_cfg("xgolite_precision"),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="XGOLite-Agile",
  env_cfg=xgolite_agile_env_cfg(),
  play_env_cfg=xgolite_agile_env_cfg(play=True),
  rl_cfg=xgolite_aggressive_ppo_runner_cfg("xgolite_agile"),
  runner_cls=VelocityOnPolicyRunner,
)
# Sim-fidelity draft (2026-07-12): v17 + one-sided torque-speed clamp +
# per-servo response-delay DR. See sim_fidelity.py for the fix rationale
# and how to enable either fix on any other preset.
register_mjlab_task(
  task_id="XGOLite-V18Draft",
  env_cfg=xgolite_v18draft_env_cfg(),
  play_env_cfg=xgolite_v18draft_env_cfg(play=True),
  rl_cfg=xgolite_aggressive_ppo_runner_cfg("xgolite_v18draft"),
  runner_cls=VelocityOnPolicyRunner,
)
# V18Draft + grid-adaptive command curriculum (Margolis RSS 2022) over an
# extended vx envelope + 8% stand-still injection; rewards/PPO = v17. The
# from-scratch answer to the Precision walking-in-place collapse — see
# range_curriculum.py.
register_mjlab_task(
  task_id="XGOLite-V18Range",
  env_cfg=xgolite_v18range_env_cfg(),
  play_env_cfg=xgolite_v18range_env_cfg(play=True),
  rl_cfg=xgolite_aggressive_ppo_runner_cfg("xgolite_v18range"),
  runner_cls=VelocityOnPolicyRunner,
)
# Precision + both sim-fidelity fixes, nothing else; to be FINE-TUNED from
# a v17 checkpoint (never from scratch — see precision collapse notes).
register_mjlab_task(
  task_id="XGOLite-Precision2",
  env_cfg=xgolite_precision2_env_cfg(),
  play_env_cfg=xgolite_precision2_env_cfg(play=True),
  rl_cfg=xgolite_aggressive_ppo_runner_cfg("xgolite_precision2"),
  runner_cls=VelocityOnPolicyRunner,
)
