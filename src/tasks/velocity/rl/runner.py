import json
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
      phase_params = actor_terms["phase"].params
      if "freq_knots" in phase_params:
        # v21a scheduled clock (mdp.observations.phase_scheduled): no fixed
        # period exists; export the full schedule + gait-member rule as one
        # JSON blob so the deploy loop can replay
        #   speed_equiv = ||v_xy_cmd|| + wz_equiv * |wz_cmd|
        #   f = piecewise_linear(speed_equiv, freq_knots)   [clamped]
        #   phase = (phase + f * dt) mod 1 per 50 Hz tick, reset to 0,
        #   frozen + obs zeroed while twist norm < stand_norm.
        # Offsets/duty are reward-side only (the obs carries the BASE phase),
        # exported for gait-aware deploy tooling and documentation.
        schedule: dict = {
          "type": "speed_scheduled",
          "update": "phase = (phase + f * dt) % 1 per control tick; "
          "sin/cos obs zeroed and phase frozen while "
          "norm(cmd[:3]) < stand_norm; phase = 0 on reset",
          "freq_knots": [
            [float(s), float(f)] for s, f in phase_params["freq_knots"]
          ],
          "wz_equiv": float(phase_params["wz_equiv"]),
          "stand_norm": float(phase_params["stand_norm"]),
          "control_dt": float(self.env.unwrapped.step_dt),
        }
        reward_cfgs = self.env.unwrapped.cfg.rewards
        if "gait_contact" in reward_cfgs:
          rp = reward_cfgs["gait_contact"].params
          schedule["gait_members"] = {
            "rule": "walk if abs(vy_cmd) > abs(vx_cmd) and "
            "abs(vy_cmd) > lat_threshold else trot",
            "lat_threshold": float(rp["lat_threshold"]),
            "foot_order": ["fl", "fr", "bl", "br"],
            "trot": {
              "offsets": [float(o) for o in rp["trot_offsets"]],
              "duty": float(rp["trot_duty"]),
            },
            "walk": {
              "offsets": [float(o) for o in rp["walk_offsets"]],
              "duty": float(rp["walk_duty"]),
            },
          }
        metadata["phase_schedule"] = json.dumps(schedule)
      else:
        metadata["phase_period"] = float(phase_params["period"])
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
