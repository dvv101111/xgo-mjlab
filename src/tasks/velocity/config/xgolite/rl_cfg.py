"""RL configuration for XGO-Lite2 velocity task."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class XgoLitePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO cfg + rsl_rl symmetry passthrough.

  mjlab's cfg dataclass does not expose rsl_rl's ``symmetry_cfg`` kwarg;
  ``asdict`` of this subclass lands it in ``cfg["algorithm"]`` where
  ``construct_algorithm`` forwards it to PPO.
  """

  symmetry_cfg: dict | None = None


def xgolite_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=XgoLitePpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.991,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      # v16: L/R mirror augmentation + mirror loss (arXiv 2403.04359). The
      # v15 review found no symmetry constraint anywhere while the hardware
      # shows sign-asymmetric strafe crosstalk and a one-sided drift; the
      # learned component of that asymmetry is removed at the source.
      symmetry_cfg={
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.5,
        "data_augmentation_func": (
          "src.tasks.velocity.rl.symmetry:mirror_obs_actions"
        ),
      },
    ),
    experiment_name="xgolite_velocity",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    # Measured convergence (curves v11b-v15 + bucket evals): resumes plateau
    # within 1000-1500 iters, fresh runs by ~2000; only the falls rate keeps
    # polishing later. Bucket-eval the result and extend +1000 if a metric
    # is off, instead of defaulting to long runs.
    max_iterations=1500,
  )
