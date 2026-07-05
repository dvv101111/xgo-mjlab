from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import xgolite_flat_env_cfg
from .rl_cfg import xgolite_ppo_runner_cfg

register_mjlab_task(
  task_id="XGOLite-Flat",
  env_cfg=xgolite_flat_env_cfg(),
  play_env_cfg=xgolite_flat_env_cfg(play=True),
  rl_cfg=xgolite_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
