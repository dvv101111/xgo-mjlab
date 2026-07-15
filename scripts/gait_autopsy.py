"""Per-leg gait autopsy for a trained checkpoint (sim-vs-hardware fork).

Purpose (2026-07-11 night): Agile v1 and Sprint v1 fail on hardware (Agile:
front legs nearly static, rear-drive only; Sprint: front/rear contact-timing
desync -> slip). This script measures the gait IN SIM to resolve, per preset,
whether the sim gait already shows the pathology (reward exploit) or looks
healthy (sim-real actuator gap).

Per command bucket it logs, per foot (fl, fr, bl, br):
  - contact duty factor and stride frequency (contact onsets/s);
  - normal-force share front pair vs rear pair (mean Fz over the window);
  - stance-mean propulsive force GRF_x in the base-yaw frame
    (negative on the front pair = "fronts act as brakes");
  - foot vertical excursion (site z p95-p05) and joint excursion per leg
    (thigh/calf angle p95-p05) front vs rear -> swing amplitude;
  - relative contact phase via FFT at the stride frequency
    (trot template fl=0, fr=0.5, bl=0.5, br=0 -> fl-br 0.0, fl-fr 0.5);
  - slip |v_xy| during stance (mean + p95);
plus actuator-envelope occupancy for the meta-question: |qvel| p50/p95/p99
and |tau| p95 / saturation fraction per joint group, action rate.

Full v17 plant DR stays active (friction, CoM, kp/kv/strength, delay,
sensor bias). push_robot is disabled: push recovery transients corrupt the
phase cross-correlation; robustness is not what is being measured here.

Usage:
  PYTHONPATH=. .venv/bin/python -u scripts/gait_autopsy.py <preset> <outdir>
  preset in {v17, fastclock, sprint, agile}
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab.tasks  # noqa: F401  (registry)
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

NUM_ENVS = 32
SETTLE_STEPS = 100   # 2 s
MEASURE_STEPS = 300  # 6 s = ~15 strides at 2.5 Hz
NOMINAL_POSE = (0.0, 0.116)

FOOT_NAMES = ("fl", "fr", "bl", "br")
FRONT, REAR = (0, 1), (2, 3)
# Trot template (foot_gait offset [0.0, 0.5, 0.5, 0.0]).
TROT_PHASE = {"fl_br": 0.0, "fr_bl": 0.0, "fl_fr": 0.5, "fl_bl": 0.5}

PRESETS = {
  "v17": (
    "XGOLite-Flat",
    "logs/rsl_rl/xgolite_velocity/2026-07-11_14-16-46/model_1499.pt",
    [("fwd_010", 0.10, 0.0), ("fwd_015", 0.15, 0.0), ("fwd_0186", 0.186, 0.0),
     ("turn_015_08", 0.15, 0.8)],
  ),
  "fastclock": (
    "XGOLite-FastClock",
    "logs/rsl_rl/xgolite_fastclock/2026-07-11_22-05-31_fastclock_v1/model_1499.pt",
    [("fwd_030", 0.30, 0.0), ("fwd_045", 0.45, 0.0)],
  ),
  "freegait": (
    "XGOLite-FreeGait",
    "logs/rsl_rl/xgolite_freegait/2026-07-11_22-53-42_freegait_v1/model_2999.pt",
    [("fwd_010", 0.10, 0.0), ("fwd_030", 0.30, 0.0), ("fwd_050", 0.50, 0.0)],
  ),
  "precision": (
    "XGOLite-Precision",
    "logs/rsl_rl/xgolite_precision/2026-07-11_23-39-09_precision_v1/model_1499.pt",
    [("fwd_006", 0.06, 0.0), ("fwd_010", 0.10, 0.0), ("back_006", -0.06, 0.0),
     ("fwd_015", 0.15, 0.0)],
  ),
  # Precision rewards fine-tuned FROM v17 under the clamped env — collapsed
  # to in-place stepping like the from-scratch run (tight relative sigma is
  # reward-dead at this robot's gait-noise floor even from a walking init).
  "precision2_ft": (
    "XGOLite-Precision2",
    "logs/rsl_rl/xgolite_precision2/2026-07-12_02-40-38_precision2_ft_v1/model_2998.pt",
    [("fwd_006", 0.06, 0.0), ("fwd_010", 0.10, 0.0), ("fwd_015", 0.15, 0.0),
     ("back_006", -0.06, 0.0)],
  ),
  # v17 weights evaluated under the clamped/per-servo-delay env: the honest-
  # physics baseline that v18base must beat like-for-like.
  "v17_clamped": (
    "XGOLite-V18Draft",
    "logs/rsl_rl/xgolite_velocity/2026-07-11_14-16-46/model_1499.pt",
    [("fwd_006", 0.06, 0.0), ("fwd_010", 0.10, 0.0), ("fwd_015", 0.15, 0.0),
     ("fwd_030", 0.30, 0.0), ("back_010", -0.10, 0.0),
     ("turn_015_08", 0.15, 0.8)],
  ),
  # v18range v2 weights autopsied under the V18Draft task id ON PURPOSE:
  # identical physics (clamp + per-servo delay), but the plain box sampler —
  # this script pins commands via cfg.ranges, which grid-sampler tasks
  # ignore (see eval_policy_buckets.py fix, 2026-07-12).
  "v18range_v2": (
    "XGOLite-V18Draft",
    "logs/rsl_rl/xgolite_v18range/2026-07-12_04-13-29_v18range_v2/model_2499.pt",
    [("fwd_006", 0.06, 0.0), ("fwd_010", 0.10, 0.0), ("fwd_015", 0.15, 0.0),
     ("fwd_030", 0.30, 0.0), ("back_010", -0.10, 0.0),
     ("turn_015_08", 0.15, 0.8)],
  ),
  "v18base": (
    "XGOLite-V18Draft",
    "logs/rsl_rl/xgolite_v18draft/2026-07-12_01-21-55_v18base_v1/model_1499.pt",
    [("fwd_006", 0.06, 0.0), ("fwd_010", 0.10, 0.0), ("fwd_015", 0.15, 0.0),
     ("fwd_030", 0.30, 0.0), ("back_010", -0.10, 0.0),
     ("turn_015_08", 0.15, 0.8)],
  ),
  # v20 autopsied on its own task id (fit-v4 measured plant): the grid
  # sampler is disabled in main() (same fix as eval_policy_buckets.py) so
  # the cfg.ranges pinning below is effective. Buckets = the v18range_v2
  # set for like-for-like comparison + fwd_040 (the new frontier where the
  # bucket eval showed falls: 7/256 envs).
  "v20": (
    "XGOLite-V20",
    "logs/rsl_rl/xgolite_v20/2026-07-15_18-35-05_v20_v1/model_2499.pt",
    [("fwd_006", 0.06, 0.0), ("fwd_010", 0.10, 0.0), ("fwd_015", 0.15, 0.0),
     ("fwd_030", 0.30, 0.0), ("fwd_040", 0.40, 0.0), ("back_010", -0.10, 0.0),
     ("turn_015_08", 0.15, 0.8)],
  ),
  "sprint": (
    "XGOLite-Sprint",
    "logs/rsl_rl/xgolite_sprint/2026-07-11_21-41-50_sprint_v1/model_1499.pt",
    [("fwd_030", 0.30, 0.0), ("fwd_045", 0.45, 0.0), ("fwd_060", 0.60, 0.0)],
  ),
  "agile": (
    "XGOLite-Agile",
    "logs/rsl_rl/xgolite_agile/2026-07-11_22-29-33_agile_v1/model_1499.pt",
    [("fwd_020", 0.20, 0.0), ("fwd_030", 0.30, 0.0), ("fwd_040", 0.40, 0.0),
     ("turn_030_08", 0.30, 0.8), ("turn_020_12", 0.20, 1.2)],
  ),
  # Low band = what the hardware sessions actually commanded (deploy ran
  # v17-era CMD_LIMITS, vx <= 0.186).
  "sprint_low": (
    "XGOLite-Sprint",
    "logs/rsl_rl/xgolite_sprint/2026-07-11_21-41-50_sprint_v1/model_1499.pt",
    [("fwd_010", 0.10, 0.0), ("fwd_0186", 0.186, 0.0)],
  ),
  "agile_low": (
    "XGOLite-Agile",
    "logs/rsl_rl/xgolite_agile/2026-07-11_22-29-33_agile_v1/model_1499.pt",
    [("fwd_010", 0.10, 0.0), ("fwd_0186", 0.186, 0.0),
     ("turn_015_08", 0.15, 0.8)],
  ),
}


def pctl(x: np.ndarray, q: float) -> float:
  return float(np.percentile(x, q)) if x.size else float("nan")


def circ_stats(phases: np.ndarray) -> tuple[float, float]:
  """Circular mean (cycles, in [-0.5, 0.5)) and resultant length R."""
  if phases.size == 0:
    return float("nan"), float("nan")
  z = np.exp(2j * np.pi * phases)
  m = z.mean()
  mean = np.angle(m) / (2 * np.pi)
  return float(mean), float(abs(m))


def main() -> None:
  preset = sys.argv[1]
  outdir = Path(sys.argv[2])
  outdir.mkdir(parents=True, exist_ok=True)
  task, ckpt, buckets = PRESETS[preset]

  configure_torch_backends()
  device = "cuda:0"

  env_cfg = load_env_cfg(task, play=False)
  env_cfg.scene.num_envs = NUM_ENVS
  env_cfg.events.pop("push_robot", None)  # see module docstring
  # Grid tasks: the adaptive curriculum would ignore cfg.ranges pinning
  # (eval_policy_buckets.py fix, 2026-07-12); drop it and disable the
  # sampler's grid mode + standing envs after build (below).
  env_cfg.curriculum.pop("command_grid", None)
  agent_cfg = load_rl_cfg(task)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  uenv = wrapped.unwrapped
  dt = uenv.step_dt
  robot = uenv.scene["robot"]
  contact = uenv.scene["feet_ground_contact"]

  joint_names = list(robot.joint_names)
  print(f"joint order: {joint_names}")
  site_ids, site_names = robot.find_sites(list(FOOT_NAMES), preserve_order=True)
  print(f"foot sites: {site_names} -> ids {site_ids}")
  assert list(site_names) == list(FOOT_NAMES)

  def jgroup(leg_prefixes, part):
    return [i for i, n in enumerate(joint_names)
            if any(n.startswith(f"{p}_{part}") for p in leg_prefixes)]

  groups = {
    "front_thigh": jgroup(("fl", "fr"), "thigh"),
    "rear_thigh": jgroup(("bl", "br"), "thigh"),
    "front_calf": jgroup(("fl", "fr"), "calf"),
    "rear_calf": jgroup(("bl", "br"), "calf"),
    "front_hip": jgroup(("fl", "fr"), "hip"),
    "rear_hip": jgroup(("bl", "br"), "hip"),
  }
  for k, v in groups.items():
    assert len(v) == 2, (k, v, joint_names)

  # Pin the command sampler to a constant twist at nominal pose.
  term = uenv.command_manager.get_term("twist")
  if getattr(term, "grid_enabled", False):
    term.grid_enabled = False
    term.cfg.rel_standing_envs = 0.0
    term.is_standing_env[:] = False
  term.cfg.axis_focus_probs = None
  term.cfg.pose_mode_probs = (1.0, 0.0, 0.0)
  term.cfg.nominal_pose = NOMINAL_POSE
  term.cfg.ranges.body_pitch = (NOMINAL_POSE[0], NOMINAL_POSE[0])
  term.cfg.ranges.base_height = (NOMINAL_POSE[1], NOMINAL_POSE[1])
  for attr in ("slow_vx_prob", "fast_vx_prob", "turn_at_speed_prob",
               "init_velocity_prob"):
    if hasattr(term.cfg, attr):
      setattr(term.cfg, attr, 0.0)

  results = {}
  for bucket_name, vx, wz in buckets:
    term.cfg.ranges.lin_vel_x = (vx, vx)
    term.cfg.ranges.lin_vel_y = (0.0, 0.0)
    term.cfg.ranges.ang_vel_z = (wz, wz)

    obs, _ = wrapped.reset()
    falls = 0
    done_mask = torch.zeros(NUM_ENVS, dtype=torch.bool, device=device)
    rec = {k: [] for k in
           ("found", "force", "foot_z", "foot_vxy", "qpos", "qvel", "tau",
            "vxb", "wzb", "yaw", "gx", "act")}
    prev_act = None

    with torch.no_grad():
      for step in range(SETTLE_STEPS + MEASURE_STEPS):
        actions = policy(obs)
        obs, _, dones, _ = wrapped.step(actions)
        tm = uenv.termination_manager
        falls += int(tm.get_term("fell_over").sum().item())
        falls += int(tm.get_term("illegal_contact").sum().item())
        if step < SETTLE_STEPS:
          prev_act = actions.clone()
          continue
        done_mask |= dones.to(device).bool().view(-1)

        quat = robot.data.root_link_quat_w  # (B, 4) wxyz
        w, x, y, z = quat.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        rec["found"].append((contact.data.found > 0).clone())
        rec["force"].append(contact.data.force.reshape(NUM_ENVS, 4, 3).clone())
        rec["foot_z"].append(robot.data.site_pos_w[:, site_ids, 2].clone())
        rec["foot_vxy"].append(
          robot.data.site_lin_vel_w[:, site_ids, :2].clone())
        rec["qpos"].append(robot.data.joint_pos.clone())
        rec["qvel"].append(robot.data.joint_vel.clone())
        rec["tau"].append(robot.data.actuator_force.clone())
        rec["vxb"].append(robot.data.root_link_lin_vel_b[:, 0].clone())
        rec["wzb"].append(robot.data.root_link_ang_vel_b[:, 2].clone())
        rec["yaw"].append(yaw.clone())
        rec["gx"].append(robot.data.projected_gravity_b[:, 0].clone())
        rec["act"].append((actions - prev_act).abs().mean(dim=1).clone())
        prev_act = actions.clone()

    arr = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
    clean = ~done_mask.cpu().numpy()  # envs with no reset inside the window
    n_clean = int(clean.sum())

    T = MEASURE_STEPS
    found = arr["found"][:, clean]              # (T, C, 4)
    force = arr["force"][:, clean]              # (T, C, 4, 3)
    foot_z = arr["foot_z"][:, clean]
    foot_vxy = arr["foot_vxy"][:, clean]
    qpos = arr["qpos"][:, clean]
    qvel = arr["qvel"][:, clean]
    tau = arr["tau"][:, clean]
    yaw = arr["yaw"][:, clean]

    stance = found.astype(bool)
    stance_f = stance.astype(np.float64)
    duty = stance_f.mean(axis=0).mean(axis=0)  # (4,)

    # Sensor convention (verified at stand: sum(fz)*duty ~ -weight): the
    # reported force is what the FOOT applies to the TERRAIN. Negate to get
    # the ground-reaction force ON the robot; rotate into the base-yaw frame
    # so +x = propulsion, +z = support. Mask by stance: the buffer holds
    # stale garbage for feet not in contact.
    grf = -force * stance_f[..., None]  # (T, C, 4, 3)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    grf_x = cos_y[..., None] * grf[..., 0] + sin_y[..., None] * grf[..., 1]
    grf_t = np.linalg.norm(grf[..., :2], axis=-1)  # tangential magnitude
    fz = grf[..., 2]

    # Support share: window-mean normal force per foot (impulse share).
    fz_mean = fz.mean(axis=(0, 1))  # (4,)
    front_share = float(fz_mean[list(FRONT)].sum() / max(fz_mean.sum(), 1e-9))

    def stance_mean(sig):  # (T, C, 4) -> per-foot mean over stance samples
      s = (sig * stance_f).sum(axis=(0, 1))
      n = stance_f.sum(axis=(0, 1))
      return s / np.maximum(n, 1)

    fz_stance = stance_mean(fz)
    grfx_stance = stance_mean(grf_x)

    # Friction utilization |Ft|/Fn per stance sample: p95 near the per-env
    # mu means the foot rides the Coulomb limit (slipping).
    fric_util = grf_t / np.maximum(fz, 0.05)
    fric_vals = {f: fric_util[:, :, i][stance[:, :, i] & (fz[:, :, i] > 0.2)]
                 for i, f in enumerate(FOOT_NAMES)}
    fric_p50 = np.array([pctl(fric_vals[f], 50) for f in FOOT_NAMES])
    fric_p95 = np.array([pctl(fric_vals[f], 95) for f in FOOT_NAMES])

    # Stride frequency: contact onsets per second, per foot (clean envs).
    onsets = (stance[1:] & ~stance[:-1]).sum(axis=0)  # (C, 4)
    stride_hz = onsets.mean(axis=0) / (T * dt)

    # Slip during stance.
    slip = np.linalg.norm(foot_vxy, axis=-1)  # (T, C, 4)
    slip_stance_mean = stance_mean(slip)
    slip_vals = {f: slip[:, :, i][stance[:, :, i]] for i, f in
                 enumerate(FOOT_NAMES)}
    slip_p95 = np.array([pctl(slip_vals[f], 95) for f in FOOT_NAMES])

    # Foot vertical excursion (swing height proxy).
    foot_z_exc = (np.percentile(foot_z, 95, axis=0)
                  - np.percentile(foot_z, 5, axis=0)).mean(axis=0)  # (4,)

    # Joint excursion per group (p95-p05 per env per joint, then mean).
    exc = (np.percentile(qpos, 95, axis=0) - np.percentile(qpos, 5, axis=0))
    joint_exc = {k: float(exc[:, ids].mean()) for k, ids in groups.items()}

    # Relative contact phase at the stride frequency (FFT bin).
    c = stance_f - stance_f.mean(axis=0, keepdims=True)  # (T, C, 4)
    spec = np.fft.rfft(c, axis=0)  # (F, C, 4)
    freqs = np.fft.rfftfreq(T, d=dt)
    band = (freqs >= 1.0) & (freqs <= 8.0)
    mag = np.abs(spec).mean(axis=2)  # (F, C)
    f0_idx = np.array([np.flatnonzero(band)[np.argmax(mag[band, e])]
                       for e in range(n_clean)])
    f0 = freqs[f0_idx]  # (C,)
    X = spec[f0_idx, np.arange(n_clean)]  # (C, 4) complex at each env's f0

    def rel_phase(a, b):
      ph = np.angle(X[:, a] * np.conj(X[:, b])) / (2 * np.pi)
      return circ_stats(ph)

    pairs = {"fl_br": (0, 3), "fr_bl": (1, 2), "fl_fr": (0, 1),
             "fl_bl": (0, 2)}
    phases = {k: rel_phase(a, b) for k, (a, b) in pairs.items()}

    # Actuator-envelope occupancy.
    qv = np.abs(qvel)
    tq = np.abs(tau)
    occupancy = {}
    for k, ids in groups.items():
      qvg = qv[:, :, ids].ravel()
      tqg = tq[:, :, ids].ravel()
      occupancy[k] = {
        "qvel_p50": pctl(qvg, 50), "qvel_p95": pctl(qvg, 95),
        "qvel_p99": pctl(qvg, 99),
        "tau_p95": pctl(tqg, 95),
        "tau_sat_frac": float((tqg > 0.209).mean()),  # 95% of 0.22 nominal
      }

    res = {
      "cmd": [vx, 0.0, wz],
      "n_clean_envs": n_clean,
      "falls": falls,
      "vx_ach": float(arr["vxb"][:, clean].mean()),
      "wz_ach": float(arr["wzb"][:, clean].mean()),
      "grav_x_mean": float(arr["gx"][:, clean].mean()),
      "action_rate_mean": float(arr["act"][:, clean].mean()),
      "duty": {f: float(duty[i]) for i, f in enumerate(FOOT_NAMES)},
      "fz_mean": {f: float(fz_mean[i]) for i, f in enumerate(FOOT_NAMES)},
      "front_force_share": front_share,
      "fz_stance_mean": {f: float(fz_stance[i])
                         for i, f in enumerate(FOOT_NAMES)},
      "grfx_stance_mean": {f: float(grfx_stance[i])
                           for i, f in enumerate(FOOT_NAMES)},
      "fric_util_p50": {f: float(fric_p50[i])
                        for i, f in enumerate(FOOT_NAMES)},
      "fric_util_p95": {f: float(fric_p95[i])
                        for i, f in enumerate(FOOT_NAMES)},
      "stride_hz": {f: float(stride_hz[i]) for i, f in enumerate(FOOT_NAMES)},
      "f0_hz_mean": float(f0.mean()),
      "slip_stance_mean": {f: float(slip_stance_mean[i])
                           for i, f in enumerate(FOOT_NAMES)},
      "slip_p95": {f: float(slip_p95[i]) for i, f in enumerate(FOOT_NAMES)},
      "foot_z_excursion": {f: float(foot_z_exc[i])
                           for i, f in enumerate(FOOT_NAMES)},
      "joint_excursion": joint_exc,
      "phase": {k: {"mean_cycles": v[0], "R": v[1]}
                for k, v in phases.items()},
      "phase_template": TROT_PHASE,
      "occupancy": occupancy,
    }
    results[bucket_name] = res

    fs = res["front_force_share"]
    print(f"\n== {preset}/{bucket_name} cmd=({vx:+.2f}, wz {wz:+.2f}) "
          f"clean {n_clean}/{NUM_ENVS} falls {falls}")
    print(f"  vx_ach {res['vx_ach']:+.3f}  wz_ach {res['wz_ach']:+.3f}  "
          f"front Fz share {fs:.3f}  f0 {res['f0_hz_mean']:.2f} Hz")
    print(f"  duty {['%.2f' % duty[i] for i in range(4)]}  "
          f"stride_hz {['%.2f' % stride_hz[i] for i in range(4)]}")
    print(f"  GRFx stance {['%+.3f' % grfx_stance[i] for i in range(4)]}  "
          f"slip {['%.3f' % slip_stance_mean[i] for i in range(4)]}")
    print(f"  fric p95 {['%.2f' % fric_p95[i] for i in range(4)]}  "
          f"foot_z exc {['%.3f' % foot_z_exc[i] for i in range(4)]}")
    print(f"  phase fl_br {phases['fl_br'][0]:+.3f} (R {phases['fl_br'][1]:.2f})"
          f"  fl_fr {phases['fl_fr'][0]:+.3f} (R {phases['fl_fr'][1]:.2f})"
          f"  fr_bl {phases['fr_bl'][0]:+.3f} (R {phases['fr_bl'][1]:.2f})",
          flush=True)

    np.savez_compressed(outdir / f"{preset}_{bucket_name}.npz", **arr,
                        clean=clean)

  with open(outdir / f"{preset}_summary.json", "w") as f:
    json.dump(results, f, indent=2)
  print(f"\nwrote {outdir}/{preset}_summary.json")


if __name__ == "__main__":
  main()
