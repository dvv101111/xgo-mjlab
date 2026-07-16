"""Validation for the XGOLite-V21A preset (v21 gait-discovery stage).

Instantiates a small XGOLite-V21A env and asserts — by reading the BUILT
env back, not just the cfg — that the v21a gait stack took effect:

1.  Obs contract frozen: actor obs is still 49 dims/frame x history 5, and
    the phase term (actor AND critic) is the scheduled per-env accumulator
    (``mdp.observations.phase_scheduled``), 2-dim sin/cos.
2.  vy range widened to (-0.2, 0.2); grid still owns (vx, wz) only.
3.  Grid-compatible lateral focus: ~15% of resamples become pure-lateral
    (|vy| in (0.08, 0.20), vx = wz = 0) and are excluded from grid cell
    attribution (cell -1).
4.  Phase accumulator advances at the SCHEDULED frequency for pinned
    commands: stand (frozen at 0, obs zeroed), slow fwd 1.6 Hz (clamp
    below the first knot), mid fwd interpolated, fast fwd 2.5 Hz (clamp
    above the last knot), lateral-dominant (scheduled from |vy|); the
    emitted sin/cos matches the buffer.
5.  Gait member rule: lateral-dominant commands select the walk offsets
    (0.75, 0.25, 0.0, 0.5)/duty 0.75, everything else trot
    (0, 0.5, 0.5, 0)/duty 0.5.
6.  ORC contact reward: stance-in-phase force scores HIGHER than
    anti-phase force on a scripted contact pattern (both gait members),
    and the live reward term is finite and correctly shaped.
7.  Reward table: feet_gait replaced by gait_contact (weight 0.7),
    gait_symmetry present (weight 0.3), feet_air_time guard present
    (weight 0.1); feet_clearance/feet_slip/soft_landing at v20 levels.
8.  Symmetry wiring: V21A PPO cfg has symmetry_cfg None (mirror loss +
    augmentation OFF); V20 keeps them ON.
9.  20 random-action steps produce finite rewards/joint states (smoke).
10. V20 UNTOUCHED: still builds with the global episode clock (runtime
    phase obs equals the episode-clock formula), binary feet_gait with
    trot offsets/weight 0.7, vy +-0.08, lateral_focus off, and the ONNX
    metadata path still finds params["period"].

Usage:
  cd luwu_mjlab && PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python \
      scripts/check_v21a_preset.py
"""

import math
import types

import torch

import mjlab.tasks  # noqa: F401  (registry)
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.config.xgolite.v21a import (
  V21A_FREQ_KNOTS,
  V21A_LAT_THRESHOLD,
  V21A_LATERAL_FOCUS_BAND,
  V21A_LATERAL_FOCUS_PROB,
  V21A_LIN_VEL_Y,
  V21A_TROT_DUTY,
  V21A_TROT_OFFSETS,
  V21A_WALK_DUTY,
  V21A_WALK_OFFSETS,
  V21A_WZ_EQUIV,
)
from src.tasks.velocity.mdp.observations import phase as global_phase_fn
from src.tasks.velocity.mdp.observations import phase_scheduled
from src.tasks.velocity.mdp.rewards import (
  feet_gait,
  gait_contact_orc,
  gait_member_offsets,
  orc_contact_reward,
)

NUM_ENVS = 32
SMOKE_STEPS = 20
SEED = 0

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  status = "PASS" if ok else "FAIL"
  print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
  if not ok:
    _failures.append(name)


def scheduled_freq(vx: float, vy: float, wz: float) -> float:
  """Reference implementation of the deploy-side frequency schedule."""
  speed = math.hypot(vx, vy) + V21A_WZ_EQUIV * abs(wz)
  knots = sorted(V21A_FREQ_KNOTS)
  if speed <= knots[0][0]:
    return knots[0][1]
  if speed >= knots[-1][0]:
    return knots[-1][1]
  for (s0, f0), (s1, f1) in zip(knots, knots[1:]):
    if speed <= s1:
      return f0 + (f1 - f0) * (speed - s0) / (s1 - s0)
  raise AssertionError


def term_slice(env: ManagerBasedRlEnv, group: str, name: str) -> tuple[int, int]:
  """(offset, width) of a term in the flattened group obs."""
  names = env.observation_manager.active_terms[group]
  dims = env.observation_manager.group_obs_term_dim[group]
  off = 0
  for n, d in zip(names, dims):
    if n == name:
      return off, int(d[0])
    off += int(d[0])
  raise KeyError(name)


def pin_command(env: ManagerBasedRlEnv, vx: float, vy: float, wz: float) -> None:
  """Pin the twist command (diag_v19_gate_feasibility pattern).

  Replaces ``_resample_command`` wholesale so the grid sampler (which
  ignores cfg.ranges and must stay enabled for the curriculum term) and
  every focus lottery are bypassed; cell attribution is disabled (-1).
  """
  term = env.command_manager.get_term("twist")
  term.cfg.rel_standing_envs = 0.0
  term.is_standing_env[:] = False

  def _pin(self, env_ids):
    self.vel_command_b[env_ids, 0] = vx
    self.vel_command_b[env_ids, 1] = vy
    self.vel_command_b[env_ids, 2] = wz
    if self.pose_enabled:
      self.vel_command_b[env_ids, 3] = 0.0
      self.vel_command_b[env_ids, 4] = 0.116
    self.is_standing_env[env_ids] = False
    if self.grid_enabled:
      self.grid_cell_index[env_ids] = -1

  term._resample_command = types.MethodType(_pin, term)


def main() -> None:
  configure_torch_backends()
  torch.manual_seed(SEED)

  cfg = load_env_cfg("XGOLite-V21A")
  cfg.scene.num_envs = NUM_ENVS
  cfg.observations["actor"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE, render_mode=None)
  env.reset()

  # ------------------------------------------- 1. obs contract + phase ----
  actor_dim = int(env.observation_manager.group_obs_dim["actor"][0])
  history = cfg.observations["actor"].history_length or 1
  check(
    "actor obs frame stays 49-dim (x history 5 = 245)",
    actor_dim == 245 and history == 5 and actor_dim // history == 49,
    f"flat dim {actor_dim}, history {history}",
  )
  actor_phase = env.observation_manager.get_term_cfg("actor", "phase").func
  critic_phase = env.observation_manager.get_term_cfg("critic", "phase").func
  check(
    "actor AND critic phase terms are the scheduled accumulator",
    isinstance(actor_phase, phase_scheduled)
    and isinstance(critic_phase, phase_scheduled)
    and actor_phase is not critic_phase,
    f"actor {type(actor_phase).__name__}, critic {type(critic_phase).__name__}",
  )

  # --------------------------------------------------- 2. vy range --------
  twist_cfg = env.command_manager.get_term("twist").cfg
  check(
    "vy range widened to (-0.2, 0.2)",
    tuple(twist_cfg.ranges.lin_vel_y) == V21A_LIN_VEL_Y,
    f"lin_vel_y {twist_cfg.ranges.lin_vel_y}",
  )
  check(
    "lateral focus configured (prob 0.15, band (0.08, 0.20))",
    twist_cfg.lateral_focus_prob == V21A_LATERAL_FOCUS_PROB
    and tuple(twist_cfg.lateral_focus_band) == V21A_LATERAL_FOCUS_BAND,
    f"prob {twist_cfg.lateral_focus_prob}, band {twist_cfg.lateral_focus_band}",
  )

  # ------------------------------------- 3. lateral-focus resampling ------
  term = env.command_manager.get_term("twist")
  saved = (term.cfg.pose_mode_probs, term.cfg.rel_standing_envs)
  term.cfg.pose_mode_probs = (1.0, 0.0, 0.0)  # Isolate the lateral lottery.
  term.cfg.rel_standing_envs = 0.0
  all_ids = torch.arange(NUM_ENVS, device=DEVICE)
  n_draws = n_lat = n_lat_cell_ok = 0
  vy_abs_max = 0.0
  for _ in range(300):
    term._resample_command(all_ids)
    cmd = term.vel_command_b
    lat = (cmd[:, 0] == 0.0) & (cmd[:, 2] == 0.0) & (
      cmd[:, 1].abs() >= V21A_LATERAL_FOCUS_BAND[0]
    ) & (cmd[:, 1].abs() <= V21A_LATERAL_FOCUS_BAND[1])
    n_draws += NUM_ENVS
    n_lat += int(lat.sum())
    n_lat_cell_ok += int((term.grid_cell_index[lat] == -1).sum())
    vy_abs_max = max(vy_abs_max, float(cmd[:, 1].abs().max()))
  frac = n_lat / n_draws
  check(
    "pure-lateral focus episodes fire at ~15% of resamples",
    0.11 < frac < 0.19,
    f"fraction {frac:.3f} over {n_draws} draws",
  )
  check(
    "lateral-focus envs are excluded from grid cell attribution",
    n_lat > 0 and n_lat_cell_ok == n_lat,
    f"{n_lat_cell_ok}/{n_lat} carried cell -1",
  )
  check(
    "vy draws exercise the widened range (beyond the old 0.08 cap)",
    0.08 < vy_abs_max <= 0.2 + 1e-6,
    f"max |vy| drawn {vy_abs_max:.3f}",
  )
  term.cfg.pose_mode_probs, term.cfg.rel_standing_envs = saved

  # ----------------------------- 4. scheduled phase accumulator rates -----
  phase_off, phase_width = term_slice(env, "actor", "phase")
  cases = {
    "stand-freeze": (0.0, 0.0, 0.0),
    "slow fwd (clamp low)": (0.08, 0.0, 0.0),
    "mid fwd (interp)": (0.275, 0.0, 0.0),
    "fast fwd (clamp high)": (0.60, 0.0, 0.0),
    "lateral-dominant": (0.0, 0.15, 0.0),
  }
  zero_action = torch.zeros(
    NUM_ENVS, env.action_manager.total_action_dim, device=DEVICE
  )
  for name, (vx, vy, wz) in cases.items():
    pin_command(env, vx, vy, wz)
    env.reset()
    expected_f = scheduled_freq(vx, vy, wz)
    standing = math.sqrt(vx * vx + vy * vy + wz * wz) < 0.05
    ok = True
    detail = ""
    for _ in range(10):
      p_before = actor_phase.phase.clone()
      count_before = env.episode_length_buf.clone()
      obs, *_ = env.step(zero_action)
      alive = env.episode_length_buf > count_before  # Not reset mid-step.
      if standing:
        frozen = bool((actor_phase.phase[alive] == p_before[alive]).all())
        zeroed = bool(
          (obs["actor"][alive, phase_off : phase_off + phase_width] == 0).all()
        )
        freq0 = bool((actor_phase.freq[alive] == 0).all())
        ok = ok and frozen and zeroed and freq0
        detail = f"frozen={frozen} obs_zeroed={zeroed} freq0={freq0}"
      else:
        delta = (actor_phase.phase[alive] - p_before[alive]) % 1.0
        rate_err = (delta / env.step_dt - expected_f).abs().max()
        freq_err = (actor_phase.freq[alive] - expected_f).abs().max()
        # Newest history frame of the 2-dim term = the last 2 dims.
        newest = obs["actor"][alive, phase_off + phase_width - 2 : phase_off + phase_width]
        ang = actor_phase.phase[alive] * 2.0 * math.pi
        sc = torch.stack([torch.sin(ang), torch.cos(ang)], dim=1)
        obs_err = (newest - sc).abs().max()
        ok = ok and rate_err < 1e-3 and freq_err < 1e-4 and obs_err < 1e-4
        detail = (
          f"f {expected_f:.4f} Hz, rate err {rate_err:.2e}, "
          f"freq err {freq_err:.2e}, obs err {obs_err:.2e}"
        )
    check(f"phase schedule: {name}", ok, detail)

  # ------------------------------------------- 5. gait member rule --------
  member_cases = [
    ((0.20, 0.05, 0.0), False, "fwd-dominant -> trot"),
    ((0.03, 0.15, 0.0), True, "lateral-dominant -> walk"),
    ((0.10, 0.10, 0.0), False, "tie |vy|==|vx| -> trot"),
    ((0.00, 0.04, 0.0), False, "|vy| below threshold -> trot"),
  ]
  ok = True
  for (vx, vy, wz), want_walk, _label in member_cases:
    cmd = torch.tensor([[vx, vy, wz, 0.0, 0.116]], device=DEVICE)
    offsets, duty, walk_mask = gait_member_offsets(
      cmd,
      V21A_TROT_OFFSETS,
      V21A_WALK_OFFSETS,
      V21A_TROT_DUTY,
      V21A_WALK_DUTY,
      V21A_LAT_THRESHOLD,
    )
    want_off = V21A_WALK_OFFSETS if want_walk else V21A_TROT_OFFSETS
    want_duty = V21A_WALK_DUTY if want_walk else V21A_TROT_DUTY
    ok = (
      ok
      and bool(walk_mask[0]) == want_walk
      and offsets[0].tolist() == list(want_off)
      and float(duty[0]) == want_duty
    )
  check(
    "offsets/duty switch trot<->walk on lateral dominance",
    ok,
    "; ".join(label for *_x, label in member_cases),
  )

  # --------------------------------- 6. ORC reward responsiveness ---------
  # Trot member at mid-stance of the fl/br pair: force on the in-phase
  # (stance) diagonal must beat force on the anti-phase (swing) diagonal.
  base = torch.tensor([0.25], device=DEVICE)
  trot_off = torch.tensor([V21A_TROT_OFFSETS], device=DEVICE)
  trot_duty = torch.tensor([V21A_TROT_DUTY], device=DEVICE)
  in_phase = torch.tensor([[3.0, 0.0, 0.0, 3.0]], device=DEVICE)
  anti_phase = torch.tensor([[0.0, 3.0, 3.0, 0.0]], device=DEVICE)
  r_in = float(orc_contact_reward(base, trot_off, trot_duty, in_phase, 2.8))
  r_anti = float(orc_contact_reward(base, trot_off, trot_duty, anti_phase, 2.8))
  # Walk member, mid-stance of bl (offset 0) vs mid-swing of br: base phase
  # 0.375 puts bl at warped mid-stance and br (leg phase 0.875) in swing.
  walk_off = torch.tensor([V21A_WALK_OFFSETS], device=DEVICE)
  walk_duty = torch.tensor([V21A_WALK_DUTY], device=DEVICE)
  base_w = torch.tensor([0.375], device=DEVICE)
  f_bl = torch.tensor([[0.0, 0.0, 3.0, 0.0]], device=DEVICE)
  f_br = torch.tensor([[0.0, 0.0, 0.0, 3.0]], device=DEVICE)
  rw_in = float(orc_contact_reward(base_w, walk_off, walk_duty, f_bl, 2.8))
  rw_anti = float(orc_contact_reward(base_w, walk_off, walk_duty, f_br, 2.8))
  check(
    "ORC reward: stance-in-phase force > anti-phase force (both members)",
    r_in > 0.0 > r_anti and rw_in > 0.0 > rw_anti,
    f"trot {r_in:.3f} vs {r_anti:.3f}; walk {rw_in:.3f} vs {rw_anti:.3f}",
  )
  live = gait_contact_orc(env, **cfg.rewards["gait_contact"].params)
  check(
    "live gait_contact reward finite, shape [num_envs], within [-1, 1]",
    live.shape == (NUM_ENVS,)
    and bool(torch.isfinite(live).all())
    and bool((live.abs() <= 1.0).all()),
    f"range [{live.min():.3f}, {live.max():.3f}]",
  )

  # ------------------------------------------------ 7. reward table -------
  r = cfg.rewards
  check(
    "feet_gait replaced by gait_contact 0.7 + gait_symmetry 0.3 + "
    "feet_air_time 0.1",
    "foot_gait" not in r
    and r["gait_contact"].weight == 0.7
    and r["gait_symmetry"].weight == 0.3
    and r["feet_air_time"].weight == 0.1
    and r["feet_air_time"].params["threshold"] == 0.25,
    f"keys {sorted(k for k in r if k.startswith(('gait', 'feet', 'foot')))}",
  )
  check(
    "anti-degeneracy guards stay at v20 levels",
    r["foot_clearance"].weight == -3
    and r["foot_slip"].weight == -0.15
    and r["soft_landing"].weight == -1e-3
    and r["joint_acc_l2"].weight == -2.5e-7,
    f"clearance {r['foot_clearance'].weight}, slip {r['foot_slip'].weight}, "
    f"landing {r['soft_landing'].weight}, joint_acc {r['joint_acc_l2'].weight}",
  )

  # --------------------------------------------- 8. symmetry wiring -------
  rl21 = load_rl_cfg("XGOLite-V21A")
  rl20 = load_rl_cfg("XGOLite-V20")
  sym20 = rl20.algorithm.symmetry_cfg
  check(
    "V21A mirror loss/augmentation OFF; V20 keeps them ON",
    rl21.algorithm.symmetry_cfg is None
    and sym20 is not None
    and sym20["use_mirror_loss"]
    and sym20["use_data_augmentation"],
    f"v21a symmetry_cfg {rl21.algorithm.symmetry_cfg}, "
    f"v20 mirror_loss {sym20 and sym20['use_mirror_loss']}",
  )
  check(
    "V21A experiment dir is xgolite_v21a",
    rl21.experiment_name == "xgolite_v21a",
    rl21.experiment_name,
  )

  # --------------------------------------- 9. random-action smoke ---------
  pin_command(env, 0.2, 0.1, 0.0)
  env.reset()
  action = torch.zeros_like(zero_action)
  finite = True
  reward_sum = 0.0
  for _ in range(SMOKE_STEPS):
    action.uniform_(-1.0, 1.0)
    _, reward, terminated, truncated, _ = env.step(action)
    finite = (
      finite
      and bool(torch.isfinite(reward).all())
      and bool(torch.isfinite(env.scene["robot"].data.joint_pos).all())
      and bool(torch.isfinite(env.scene["robot"].data.joint_vel).all())
      and bool(torch.isfinite(actor_phase.phase).all())
    )
    reward_sum += float(reward.mean())
  check(
    f"{SMOKE_STEPS} random-action steps finite (no NaN)",
    finite,
    f"mean step reward {reward_sum / SMOKE_STEPS:.4f}",
  )
  env.close()
  del env

  # ------------------------------------------------ 10. v20 untouched -----
  cfg20 = load_env_cfg("XGOLite-V20")
  p20 = cfg20.observations["actor"].terms["phase"]
  check(
    "V20 actor phase term stays the global episode clock, period 0.4",
    p20.func is global_phase_fn
    and p20.params.get("period") == 0.4
    and "freq_knots" not in p20.params,
    f"func {getattr(p20.func, '__name__', type(p20.func).__name__)}, "
    f"params {sorted(p20.params)}",
  )
  fg20 = cfg20.rewards["foot_gait"]
  check(
    "V20 keeps binary feet_gait: trot offsets, weight 0.7, threshold 0.56",
    fg20.func is feet_gait
    and fg20.params["offset"] == [0.0, 0.5, 0.5, 0.0]
    and fg20.weight == 0.7
    and fg20.params["threshold"] == 0.56
    and fg20.params["period"] == 0.4
    and "gait_contact" not in cfg20.rewards
    and "gait_symmetry" not in cfg20.rewards
    and "feet_air_time" not in cfg20.rewards,
    f"offset {fg20.params['offset']}, weight {fg20.weight}",
  )
  t20 = cfg20.commands["twist"]
  check(
    "V20 keeps vy +-0.08 and lateral focus OFF",
    tuple(t20.ranges.lin_vel_y) == (-0.08, 0.08)
    and t20.lateral_focus_prob == 0.0,
    f"lin_vel_y {t20.ranges.lin_vel_y}, "
    f"lateral_focus_prob {t20.lateral_focus_prob}",
  )
  g20 = cfg20.curriculum["command_grid"].params
  check(
    "V20 grid gates stay 0.70/0.55",
    g20.get("gamma_lin") == 0.70 and g20.get("gamma_ang") == 0.55,
    f"gamma_lin {g20.get('gamma_lin')}, gamma_ang {g20.get('gamma_ang')}",
  )
  cfg20.scene.num_envs = 8
  cfg20.observations["actor"].enable_corruption = False
  env20 = ManagerBasedRlEnv(cfg=cfg20, device=DEVICE, render_mode=None)
  env20.reset()
  pin_command(env20, 0.2, 0.0, 0.0)
  env20.reset()
  a20 = torch.zeros(8, env20.action_manager.total_action_dim, device=DEVICE)
  ok = True
  for _ in range(5):
    obs20, *_ = env20.step(a20)
    gp = (env20.episode_length_buf * env20.step_dt) % 0.4 / 0.4
    expected = torch.stack(
      [torch.sin(gp * 2 * torch.pi), torch.cos(gp * 2 * torch.pi)], dim=1
    )
    off20, width20 = term_slice(env20, "actor", "phase")
    newest = obs20["actor"][:, off20 + width20 - 2 : off20 + width20]
    ok = ok and bool((newest - expected).abs().max() < 1e-4)
  finite20 = True
  for _ in range(5):
    a20.uniform_(-1.0, 1.0)
    _, r20, *_ = env20.step(a20)
    finite20 = finite20 and bool(torch.isfinite(r20).all())
  check(
    "V20 runtime phase obs still follows the global episode clock",
    ok and finite20,
    "5 pinned steps match sin/cos(episode_clock / 0.4); random steps finite",
  )
  env20.close()

  print()
  if _failures:
    print(f"OVERALL: FAIL ({len(_failures)} failed: {', '.join(_failures)})")
    raise SystemExit(1)
  print("OVERALL: PASS")


if __name__ == "__main__":
  main()
