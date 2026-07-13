import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    # v16 deploy contract: the actor may consume stacked history. Frames are
    # flattened TERM-MAJOR (each term's frames contiguous) and OLDEST-FIRST
    # within a term (mjlab CircularBuffer.buffer order); on reset the first
    # frame backfills every slot.
    history = self.env.unwrapped.cfg.observations["actor"].history_length or 1
    metadata["actor_obs_history_length"] = history
    if history > 1:
      metadata["actor_obs_layout"] = "term_major_oldest_first"
    # Aggressive presets (2026-07-11) may run the gait clock faster than the
    # deploy-side PHASE_PERIOD constant (0.4 s). Export the trained period so
    # the deploy loop can build the phase obs from metadata instead of the
    # constant; v17-era deployers ignore unknown keys, so this is additive.
    actor_terms = self.env.unwrapped.cfg.observations["actor"].terms
    if "phase" in actor_terms:
      metadata["phase_period"] = float(actor_terms["phase"].params["period"])
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
