"""Left-right mirror augmentation for the XGO-Lite velocity task (v16).

Used through rsl_rl's ``symmetry_cfg`` (data augmentation + mirror loss,
arXiv 2403.04359). The mirror is a reflection across the body xz-plane:

- legs swap fl<->fr and bl<->br with UNCHANGED joint angles: the generated
  xgolite.xml uses mirrored hip axes (fl "1 0 0" vs fr "-1 0 0", + = outward
  on both sides) and a shared thigh/calf axis "0 -1 0" (+ = foot forward),
  so conjugating each joint rotation by the reflection lands exactly on the
  opposite joint at the same angle;
- angular quantities about x/z flip sign (pseudovectors), y-components of
  true vectors flip sign;
- the trot phase clock shifts by half a period (foot_gait offsets
  [0, .5, .5, 0] swap to [.5, 0, 0, .5]), i.e. (sin, cos) -> (-sin, -cos).

Per-term maps are derived from the observation manager at first call, so the
same function serves any group (actor with history, critic without) as long
as every term present has an entry in ``_TERM_SPECS``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv

# Model joint/actuator order is (fl, fr, bl, br) x (hip, thigh, calf).
_JOINT_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
_JOINT_SIGN = [1.0] * 12
# Foot-indexed terms are ordered (fl, fr, bl, br).
_FOOT_PERM = [1, 0, 3, 2]

# term name -> (index permutation, sign flip) for ONE history frame.
_TERM_SPECS: dict[str, tuple[list[int], list[float]]] = {
  "base_ang_vel": ([0, 1, 2], [-1.0, 1.0, -1.0]),
  "projected_gravity": ([0, 1, 2], [1.0, -1.0, 1.0]),
  # command: [vx, vy, wz, body_pitch, base_height]
  "command": ([0, 1, 2, 3, 4], [1.0, -1.0, -1.0, 1.0, 1.0]),
  "phase": ([0, 1], [-1.0, -1.0]),
  "joint_pos": (_JOINT_PERM, _JOINT_SIGN),
  "joint_vel": (_JOINT_PERM, _JOINT_SIGN),
  "actions": (_JOINT_PERM, _JOINT_SIGN),
  "base_lin_vel": ([0, 1, 2], [1.0, -1.0, 1.0]),
  "foot_height": (_FOOT_PERM, [1.0] * 4),
  "foot_air_time": (_FOOT_PERM, [1.0] * 4),
  "foot_contact": (_FOOT_PERM, [1.0] * 4),
  # (fl, fr, bl, br) x (fx, fy, fz): swap feet, flip fy.
  "foot_contact_forces": (_JOINT_PERM, [1.0, -1.0, 1.0] * 4),
}

# (group, flat_dim, device) -> (perm index tensor, sign tensor)
_CACHE: dict[tuple[str, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _group_perm_sign(
  env: VecEnv, group: str, flat_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
  key = (group, flat_dim, str(device))
  if key in _CACHE:
    return _CACHE[key]

  om = env.unwrapped.observation_manager
  names = om._group_obs_term_names[group]
  cfgs = om._group_obs_term_cfgs[group]
  flat_dims = [math.prod(d) for d in om.group_obs_term_dim[group]]

  perm: list[int] = []
  sign: list[float] = []
  offset = 0
  for name, cfg, dim in zip(names, cfgs, flat_dims, strict=True):
    if name not in _TERM_SPECS:
      raise KeyError(f"No mirror spec for observation term '{name}'.")
    if cfg.history_length > 0 and not cfg.flatten_history_dim:
      raise ValueError(f"Term '{name}': only flattened history is supported.")
    term_perm, term_sign = _TERM_SPECS[name]
    history = max(1, cfg.history_length)
    base = dim // history
    # command may run without the pose channels (dim 3): truncate the spec.
    if len(term_perm) != base:
      term_perm, term_sign = term_perm[:base], term_sign[:base]
    assert len(term_perm) == base, f"Term '{name}': spec/base-dim mismatch."
    # History frames are stored frame-contiguous per term (oldest->newest);
    # the mirror applies frame-wise.
    for h in range(history):
      frame = offset + h * base
      perm.extend(frame + i for i in term_perm)
      sign.extend(term_sign)
    offset += dim
  assert offset == flat_dim, f"Group '{group}': dim mismatch {offset}/{flat_dim}."

  result = (
    torch.tensor(perm, dtype=torch.long, device=device),
    torch.tensor(sign, dtype=torch.float32, device=device),
  )
  _CACHE[key] = result
  return result


def mirror_obs_actions(
  env: VecEnv,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """rsl_rl data_augmentation_func: returns [original; mirrored] batches."""
  obs_aug = None
  if obs is not None:
    mirrored = {}
    for group, tensor in obs.items():
      perm, sign = _group_perm_sign(env, group, tensor.shape[-1], tensor.device)
      mirrored[group] = tensor.index_select(-1, perm) * sign
    obs_aug = torch.cat(
      [obs, TensorDict(mirrored, batch_size=obs.batch_size)], dim=0
    )
  actions_aug = None
  if actions is not None:
    perm = torch.tensor(_JOINT_PERM, dtype=torch.long, device=actions.device)
    actions_aug = torch.cat([actions, actions.index_select(-1, perm)], dim=0)
  return obs_aug, actions_aug
