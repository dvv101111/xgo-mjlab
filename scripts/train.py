"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  env_device: str | None = None
  """Optional simulator device, independent of the learner device.

  This is useful on ROCm systems: MuJoCo-Warp can run the environment on CPU
  while RSL-RL runs the actor, critic, rollout storage, and PPO updates on the
  AMD GPU. If unset, the environment uses the learner device as before.
  """
  warm_start_actor: str | None = None
  """Checkpoint whose ACTOR initializes this run (critic/optimizer fresh).

  Loads ONLY ``actor_state_dict`` (MLP weights, action std and the actor
  obs-normalizer state) via rsl-rl's native partial load
  (``runner.load(..., load_cfg={"actor": True}, strict=True)``) after runner
  construction and before ``learn()``. Every actor tensor must match the
  fresh model's keys and shapes exactly or the run aborts. Use case: start
  a task whose CRITIC obs contract differs (e.g. V21B adds a critic-only
  height_scan) from a flat checkpoint whose actor contract is identical —
  a plain ``--agent.resume`` strict load cannot do that. Incompatible with
  ``--agent.resume``."""

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _tensor_checksum(t) -> float:
  """Order-stable scalar fingerprint of a tensor (float64 sum on CPU)."""
  return float(t.detach().cpu().double().sum().item())


def warm_start_actor_from_checkpoint(
  runner: MjlabOnPolicyRunner, ckpt_path: Path, device: str
) -> None:
  """Load ONLY the actor (weights + obs-normalizer state) from a checkpoint.

  Strictly shape-checked: every key of the fresh actor's state dict must be
  present in the checkpoint's ``actor_state_dict`` with an identical shape
  (and vice versa) or this raises before touching the model. The critic,
  optimizer, RND and iteration counter are NOT loaded — the actual load goes
  through rsl-rl 5.x's native partial load
  (``OnPolicyRunner.load(..., load_cfg={"actor": True}, strict=True)``,
  the same path the eval scripts use), which only calls
  ``self.actor.load_state_dict``.
  """
  import torch

  if not ckpt_path.exists():
    raise FileNotFoundError(f"--warm-start-actor checkpoint not found: {ckpt_path}")
  loaded = torch.load(str(ckpt_path), weights_only=False, map_location=device)
  if "actor_state_dict" not in loaded:
    raise KeyError(
      f"--warm-start-actor checkpoint has no 'actor_state_dict' "
      f"(keys: {sorted(loaded.keys())}): {ckpt_path}"
    )
  ckpt_actor = loaded["actor_state_dict"]
  model_actor = runner.alg.actor.state_dict()

  missing = sorted(set(model_actor) - set(ckpt_actor))
  unexpected = sorted(set(ckpt_actor) - set(model_actor))
  mismatched = [
    f"{k}: checkpoint {tuple(ckpt_actor[k].shape)} vs model "
    f"{tuple(model_actor[k].shape)}"
    for k in sorted(set(model_actor) & set(ckpt_actor))
    if tuple(ckpt_actor[k].shape) != tuple(model_actor[k].shape)
  ]
  if missing or unexpected or mismatched:
    raise ValueError(
      "--warm-start-actor: actor state dict is not shape-compatible with "
      f"the fresh model (checkpoint: {ckpt_path}).\n"
      f"  missing from checkpoint: {missing}\n"
      f"  unexpected in checkpoint: {unexpected}\n"
      f"  shape mismatches: {mismatched}"
    )

  actor_before = _tensor_checksum(runner.alg.actor.state_dict()["mlp.0.weight"])
  critic_before = _tensor_checksum(runner.alg.critic.state_dict()["mlp.0.weight"])

  # Native rsl-rl partial load: actor only; critic/optimizer/rnd untouched,
  # iteration not restored (training starts at 0).
  runner.load(str(ckpt_path), load_cfg={"actor": True}, strict=True, map_location=device)

  actor_after = _tensor_checksum(runner.alg.actor.state_dict()["mlp.0.weight"])
  critic_after = _tensor_checksum(runner.alg.critic.state_dict()["mlp.0.weight"])
  ckpt_sum = _tensor_checksum(ckpt_actor["mlp.0.weight"])

  print(
    f"[INFO] Warm-start actor from: {ckpt_path} "
    f"(checkpoint iter {loaded.get('iter', '?')})"
  )
  for k in sorted(ckpt_actor):
    print(f"[INFO]   loaded actor tensor {k}: {tuple(ckpt_actor[k].shape)}")
  print(
    f"[INFO]   actor mlp.0.weight checksum: fresh {actor_before:.6f} -> "
    f"loaded {actor_after:.6f} (checkpoint {ckpt_sum:.6f})"
  )
  print(
    f"[INFO]   critic mlp.0.weight checksum: {critic_before:.6f} -> "
    f"{critic_after:.6f} (must be unchanged; critic/optimizer stay fresh)"
  )
  if actor_after == actor_before:
    print(
      "[WARN] Warm-start actor checksum did not change (checkpoint identical "
      "to the fresh init?)"
    )
  if critic_after != critic_before:
    raise RuntimeError(
      "--warm-start-actor: critic state changed during the actor-only load; "
      "refusing to continue."
    )


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  env_device = cfg.env_device or device
  print(
    f"[INFO] Training with: learner_device={device}, "
    f"env_device={env_device}, seed={seed}, rank={rank}"
  )

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env,
    device=env_device,
    render_mode="rgb_array" if cfg.video else None,
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  # Dump BEFORE runner construction: the runner mutates agent_cfg in place
  # (resolve_symmetry_config injects the live env under
  # algorithm.symmetry_cfg._env), which is not YAML-serializable.
  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  if cfg.warm_start_actor is not None:
    if resume_path is not None:
      raise ValueError(
        "--warm-start-actor is incompatible with --agent.resume (resume "
        "loads the full training state, including the actor)."
      )
    warm_start_actor_from_checkpoint(
      runner, Path(cfg.warm_start_actor).expanduser().resolve(), device
    )

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
