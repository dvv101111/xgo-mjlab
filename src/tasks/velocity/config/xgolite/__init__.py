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
from .v19 import xgolite_v19_env_cfg, xgolite_v19_ppo_runner_cfg
from .v20 import xgolite_v20_env_cfg, xgolite_v20_ppo_runner_cfg
from .v21a import xgolite_v21a_env_cfg, xgolite_v21a_ppo_runner_cfg
from .v21b import xgolite_v21b_env_cfg, xgolite_v21b_ppo_runner_cfg

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
# V18Range on the MEASURED servo plant (Stage-1 system-ID fit, 2026-07-14):
# fitted joint dynamics/PD gains, piecewise torque-speed knee, measured
# delay DR, wide deadband + friction DR, shrunk strength DR. Rewards,
# curriculum and obs identical to V18Range; PPO identical except init
# noise std sized to the stiff plant (v1 suicide-collapse postmortem) —
# see v19.py.
register_mjlab_task(
  task_id="XGOLite-V19",
  env_cfg=xgolite_v19_env_cfg(),
  play_env_cfg=xgolite_v19_env_cfg(play=True),
  rl_cfg=xgolite_v19_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
# V18Range on the fit-v4 CORRECTED plant (soft damped spring — the v3
# stiff relay was a v2.3 data artifact): keeps v19's measured-plant
# wiring + deploy-aligned control-rate obs/penalty, reverts the two
# relay-sized knobs (init_std 1.0, gates 0.70/0.55) — see v20.py.
register_mjlab_task(
  task_id="XGOLite-V20",
  env_cfg=xgolite_v20_env_cfg(),
  play_env_cfg=xgolite_v20_env_cfg(play=True),
  rl_cfg=xgolite_v20_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
# V20 + the v21 gait-discovery stack (2026-07-15 lit review, shortlist B):
# ORC phase-contact reward on a speed-scheduled per-env phase clock with a
# lateral-dominant walk member, morphological-symmetry REWARD instead of
# the trot-only mirror loss/augmentation (symmetry=False), feet_air_time
# guard, vy widened to +-0.2 with grid-compatible lateral focus episodes.
# See v21a.py for parameter provenance.
register_mjlab_task(
  task_id="XGOLite-V21A",
  env_cfg=xgolite_v21a_env_cfg(),
  play_env_cfg=xgolite_v21a_env_cfg(play=True),
  rl_cfg=xgolite_v21a_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
# V21A + the v21 terrain stack (2026-07-15 lit review, shortlist A2-A5):
# scaled procedural terrain grid (10 difficulty rows, HIM-leaning stair
# weighting, steps <= 0.5 leg lengths), critic-only height scan (actor
# stays the blind 49-dim deploy contract), terrain-relative base-height +
# foot-clearance rewards, per-env contact-compliance DR, friction low tail
# (0.05) + transient per-foot slip events, reward-gated terrain-level
# curriculum with the low-speed demotion guard. 3000-iter default (fresh
# terrain run). See v21b.py for parameter provenance.
register_mjlab_task(
  task_id="XGOLite-V21B",
  env_cfg=xgolite_v21b_env_cfg(),
  play_env_cfg=xgolite_v21b_env_cfg(play=True),
  rl_cfg=xgolite_v21b_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
