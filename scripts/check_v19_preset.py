"""Validation for the XGOLite-V19 preset (measured servo plant, 2026-07-14).

Instantiates a small XGOLite-V19 env and asserts — by reading the BUILT
model/env back, not the cfg — that every measured-plant change took effect:

1. MJCF joint dynamics: dof damping/armature/frictionloss of the 12 leg
   joints equal the Stage-1 fit values (spec_fn override, XML untouched).
2. PD gains: default actuator gainprm/biasprm equal the measured
   kp 39.713 / kd 0.0068 (strength DR then scales per env from these).
3. Piecewise clamp: event params carry tau_max 0.22 / qd_knee 3.647 /
   qd_max 12.094; after the first step the captured per-servo tau_max sits
   inside the SHRUNK strength envelope and the speed axis (qd_knee, qd_max)
   scales with the SAME strength factor (DC motor: tau AND omega
   proportional to voltage, so the battery-sag proxy moves the whole
   envelope — matching the v18 clamp's omega_nl scaling).
4. Delay DR: per-servo actuator with lag range 10..32 steps (21-65 ms).
5. Deadband: per-(env, servo) draws inside (0.0, 0.025) rad.
6. Friction DR: foot-pad sliding friction inside (0.25, 2.0), shared
   across the 4 pads of an env, varying across envs; pads have collision
   priority 1 so the low half is not masked by the plane's friction 1.0.
7. Control-rate joint_vel obs (v19.py point 9): the ACTOR's joint_vel term
   is the 50 Hz position finite difference (deploy-aligned, dither-free);
   verified by type AND by value against a manual finite difference across
   one control step. The CRITIC keeps the instantaneous privileged term.
8. Stability: ~50 random-action steps produce finite rewards/joint states.
9. V18 UNTOUCHED: XGOLite-V18Range still builds; its clamp params map to
   the old single-line envelope (tau_max 0.22, qd_knee 0, qd_max 4.5), its
   compiled model keeps the hand-set XML plant (kp 5.0, damping 0.05), its
   DR ranges are the v17/v18 originals, and its actor joint_vel obs stays
   the upstream instantaneous term.

Usage:
  cd luwu_mjlab && PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python \
      scripts/check_v19_preset.py
"""

import torch

import mjlab.tasks  # noqa: F401  (registry)
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.config.xgolite.measured_actuators import (
  DELAY_MAX_LAG_STEPS,
  DELAY_MIN_LAG_STEPS,
  MEASURED_JOINTS,
  MEASURED_QD_KNEE,
  MEASURED_QD_MAX,
  TAU_MAX,
)
from src.tasks.velocity.config.xgolite.v19 import (
  V19_DEADBAND_RANGE,
  V19_FRICTION_RANGE,
  V19_STRENGTH_ENV_RANGE,
  V19_STRENGTH_SERVO_RANGE,
)
from src.tasks.velocity.mdp.actuators import PerServoDelayedActuator
from src.tasks.velocity.mdp.events import TorqueSpeedClamp
from src.tasks.velocity.mdp.observations import joint_vel_control_rate_rel

NUM_ENVS = 32
NUM_STEPS = 50
SEED = 0

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  status = "PASS" if ok else "FAIL"
  print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
  if not ok:
    _failures.append(name)


def named_id(mj_model, obj: str, name: str) -> int:
  """Resolve a possibly entity-prefixed mujoco name to its id."""
  getter = {"joint": mj_model.joint, "actuator": mj_model.actuator,
            "geom": mj_model.geom}[obj]
  for cand in (name, f"robot/{name}"):
    try:
      return getter(cand).id
    except KeyError:
      continue
  raise KeyError(f"{obj} {name!r} not found in compiled model")


def main() -> None:
  configure_torch_backends()
  torch.manual_seed(SEED)

  cfg = load_env_cfg("XGOLite-V19")
  cfg.scene.num_envs = NUM_ENVS
  # Corruption off so check 7 can compare raw obs values exactly; the
  # training cfg keeps it on (the term itself is what is under test).
  cfg.observations["actor"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE, render_mode=None)
  env.reset()
  mj_model = env.sim.mj_model

  # ------------------------------------------ 1. measured joint dynamics --
  damp_err = arm_err = fric_err = 0.0
  for joint, p in MEASURED_JOINTS.items():
    jid = named_id(mj_model, "joint", joint + "_joint")
    dof = int(mj_model.jnt_dofadr[jid])
    damp_err = max(damp_err, abs(float(mj_model.dof_damping[dof]) - p["damping"]))
    arm_err = max(arm_err, abs(float(mj_model.dof_armature[dof]) - p["armature"]))
    fric_err = max(
      fric_err, abs(float(mj_model.dof_frictionloss[dof]) - p["frictionloss"])
    )
  check(
    "compiled dof damping/armature/frictionloss == measured fit",
    max(damp_err, arm_err, fric_err) < 1e-9,
    f"max errs damping {damp_err:.2e} armature {arm_err:.2e} "
    f"frictionloss {fric_err:.2e}",
  )

  # ------------------------------------------------- 2. measured PD gains --
  default_gain = env.sim.get_default_field("actuator_gainprm")
  default_bias = env.sim.get_default_field("actuator_biasprm")
  kp_err = kd_err = 0.0
  for joint, p in MEASURED_JOINTS.items():
    aid = named_id(mj_model, "actuator", joint + "_joint")
    kp_err = max(
      kp_err,
      abs(float(default_gain[aid, 0]) - p["kp"]),
      abs(float(default_bias[aid, 1]) + p["kp"]),
    )
    kd_err = max(kd_err, abs(float(default_bias[aid, 2]) + p["kd"]))
  check(
    "default actuator gains == measured kp 39.713 / kd 0.0068",
    max(kp_err, kd_err) < 1e-4,
    f"max kp err {kp_err:.2e}, kd err {kd_err:.2e}",
  )

  # ------------------------------------------------- 3. piecewise clamp ----
  clamp_params = env.event_manager.get_term_cfg("torque_speed_clamp").params
  check(
    "clamp params are the measured piecewise curve",
    clamp_params["tau_max"] == TAU_MAX
    and clamp_params["qd_knee"] == MEASURED_QD_KNEE
    and clamp_params["qd_max"] == MEASURED_QD_MAX,
    f"tau_max={clamp_params['tau_max']}, qd_knee={clamp_params['qd_knee']}, "
    f"qd_max={clamp_params['qd_max']}",
  )
  term = env.event_manager.get_term_cfg("torque_speed_clamp").func
  assert isinstance(term, TorqueSpeedClamp)
  action = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=DEVICE
  )
  env.step(action)  # triggers the clamp's lazy init (strength capture)
  assert term._tau_max is not None
  strength = term._tau_max / TAU_MAX
  s_lo = V19_STRENGTH_ENV_RANGE[0] * V19_STRENGTH_SERVO_RANGE[0]
  s_hi = V19_STRENGTH_ENV_RANGE[1] * V19_STRENGTH_SERVO_RANGE[1]
  check(
    f"captured tau_max strength inside shrunk envelope {s_lo:.4f}..{s_hi:.4f}",
    bool(strength.min() >= s_lo - 1e-4)
    and bool(strength.max() <= s_hi + 1e-4)
    and bool((strength.max() - strength.min()) > 0.01),
    f"strength range [{strength.min():.4f}, {strength.max():.4f}]",
  )
  check(
    "clamp speed axis scales with the strength factor (tau AND omega ~ V)",
    isinstance(term._qd_knee, torch.Tensor)
    and isinstance(term._qd_max, torch.Tensor)
    and bool((term._qd_knee - strength * MEASURED_QD_KNEE).abs().max() < 1e-5)
    and bool((term._qd_max - strength * MEASURED_QD_MAX).abs().max() < 1e-4),
    f"qd_max/strength range [{(term._qd_max / strength).min():.3f}, "
    f"{(term._qd_max / strength).max():.3f}] (nominal {MEASURED_QD_MAX})",
  )

  # ------------------------------------------------------- 4. delay DR ----
  actuator = env.scene["robot"].actuators[0]
  check(
    "per-servo delayed actuator with measured 10..32 step lag range",
    isinstance(actuator, PerServoDelayedActuator)
    and actuator.cfg.delay_min_lag == DELAY_MIN_LAG_STEPS
    and actuator.cfg.delay_max_lag == DELAY_MAX_LAG_STEPS
    and all(
      buf.min_lag == DELAY_MIN_LAG_STEPS and buf.max_lag == DELAY_MAX_LAG_STEPS
      for buf in actuator._delay_buffers.values()
    ),
    f"type={type(actuator).__name__}, cfg lag range "
    f"[{actuator.cfg.delay_min_lag}, {actuator.cfg.delay_max_lag}]",
  )

  # ------------------------------------------------------- 5. deadband ----
  db = actuator._deadband
  lo, hi = V19_DEADBAND_RANGE
  within_env = (
    db.max(dim=1).values - db.min(dim=1).values if db is not None else None
  )
  check(
    "deadband drawn per (env, servo) inside (0.0, 0.025) rad",
    db is not None
    and db.shape == (NUM_ENVS, 12)
    and bool((db >= lo).all() and (db <= hi).all())
    and bool((within_env > 1e-4).all()),
    f"range [{db.min():.4f}, {db.max():.4f}], "
    f"min within-env spread {within_env.min():.5f}"
    if db is not None
    else "deadband tensor not allocated",
  )

  # ---------------------------------------------------- 6. friction DR ----
  pad_ids = [
    named_id(mj_model, "geom", f"{f}_foot_pad") for f in ("fl", "fr", "bl", "br")
  ]
  prio_ok = all(int(mj_model.geom_priority[g]) == 1 for g in pad_ids)
  check("foot pads have collision priority 1 (friction DR authoritative)",
        prio_ok)
  fr = env.sim.model.geom_friction[:, pad_ids, 0]  # (envs, 4)
  f_lo, f_hi = V19_FRICTION_RANGE
  shared = (fr.max(dim=1).values - fr.min(dim=1).values).max().item()
  spread = (fr[:, 0].max() - fr[:, 0].min()).item()
  check(
    "foot sliding friction inside (0.25, 2.0), shared per env, varied across",
    bool((fr >= f_lo - 1e-6).all() and (fr <= f_hi + 1e-6).all())
    and shared < 1e-6
    and spread > 0.8,
    f"range [{fr.min():.3f}, {fr.max():.3f}], within-env spread {shared:.2e}, "
    f"across-env spread {spread:.3f}",
  )

  # -------------------------------------- 7. control-rate joint_vel obs ----
  actor_jv = env.observation_manager.get_term_cfg("actor", "joint_vel")
  critic_jv = env.observation_manager.get_term_cfg("critic", "joint_vel")
  check(
    "actor joint_vel is the control-rate term; critic keeps the upstream one",
    isinstance(actor_jv.func, joint_vel_control_rate_rel)
    and not isinstance(critic_jv.func, joint_vel_control_rate_rel),
    f"actor {type(actor_jv.func).__name__}, critic "
    f"{getattr(critic_jv.func, '__name__', type(critic_jv.func).__name__)}",
  )
  robot = env.scene["robot"]
  q0 = robot.data.joint_pos.clone()
  obs, *_ = env.step(torch.zeros_like(action))
  q1 = robot.data.joint_pos
  expected = (q1 - q0) / env.step_dt
  # Actor obs layout is term-major flattened history, oldest first: the
  # newest joint_vel frame is the last 12 dims of the term's block.
  names = env.observation_manager.active_terms["actor"]
  dims = env.observation_manager.group_obs_term_dim["actor"]
  off = 0
  jv_width = 0
  for n, d in zip(names, dims):
    if n == "joint_vel":
      jv_width = int(d[0])
      break
    off += int(d[0])
  latest = obs["actor"][:, off + jv_width - 12 : off + jv_width]
  obs_err = (latest - expected).abs().max()
  inst_gap = (robot.data.joint_vel - expected).abs().max()
  check(
    "actor joint_vel obs == control-step position finite difference",
    jv_width == 60 and bool(obs_err < 1e-4),
    f"max |obs - fd| {obs_err:.2e}; max |instantaneous - fd| {inst_gap:.3f} "
    "(nonzero gap = relay dither the obs no longer carries)",
  )
  grid_params = cfg.curriculum["command_grid"].params
  check(
    "grid unlock gates recalibrated to the measured plant (v19.py point 10)",
    grid_params.get("gamma_lin") == 0.15
    and grid_params.get("gamma_ang") == 0.17,
    f"gamma_lin={grid_params.get('gamma_lin')}, "
    f"gamma_ang={grid_params.get('gamma_ang')}",
  )

  # ------------------------------------------------------ 8. stability ----
  finite = True
  reward_sum = 0.0
  for _ in range(NUM_STEPS):
    action.uniform_(-1.0, 1.0)
    _, reward, terminated, truncated, _ = env.step(action)
    finite = (
      finite
      and bool(torch.isfinite(reward).all())
      and bool(torch.isfinite(env.scene["robot"].data.joint_pos).all())
      and bool(torch.isfinite(env.scene["robot"].data.joint_vel).all())
    )
    reward_sum += float(reward.mean())
  check(
    f"{NUM_STEPS} random-action steps finite (no NaN)",
    finite,
    f"mean step reward {reward_sum / NUM_STEPS:.4f}",
  )
  env.close()
  del env

  # -------------------------------------------------- 9. v18 untouched ----
  v18_cfg = load_env_cfg("XGOLite-V18Range")
  p18 = v18_cfg.events["torque_speed_clamp"].params
  check(
    "V18Range clamp params == old single-line envelope (knee 0, qd_max 4.5)",
    p18["tau_max"] == 0.22 and p18["qd_knee"] == 0.0 and p18["qd_max"] == 4.5,
    f"tau_max={p18['tau_max']}, qd_knee={p18['qd_knee']}, qd_max={p18['qd_max']}",
  )
  s18 = v18_cfg.events["actuator_strength"].params
  check(
    "V18Range keeps v17 strength DR and friction DR ranges",
    s18["strength_scale_range"] == (0.75, 1.25)
    and s18["strength_servo_range"] == (0.90, 1.10)
    and v18_cfg.events["foot_friction"].params["ranges"] == (0.3, 1.6),
    f"strength {s18['strength_scale_range']} x {s18['strength_servo_range']}, "
    f"friction {v18_cfg.events['foot_friction'].params['ranges']}",
  )
  act18 = v18_cfg.scene.entities["robot"].articulation.actuators[0]
  check(
    "V18Range keeps 30..50 step delay and no deadband",
    act18.delay_min_lag == 30
    and act18.delay_max_lag == 50
    and act18.deadband_range == (0.0, 0.0),
    f"lag [{act18.delay_min_lag}, {act18.delay_max_lag}], "
    f"deadband_range {act18.deadband_range}",
  )
  jv18 = v18_cfg.observations["actor"].terms["joint_vel"].func
  check(
    "V18Range actor joint_vel stays the upstream instantaneous term",
    jv18 is not joint_vel_control_rate_rel,
    f"func {getattr(jv18, '__name__', type(jv18).__name__)}",
  )
  v18_cfg.scene.num_envs = 8
  v18_env = ManagerBasedRlEnv(cfg=v18_cfg, device=DEVICE, render_mode=None)
  v18_env.reset()
  mj18 = v18_env.sim.mj_model
  jid = named_id(mj18, "joint", "fl_hip_joint")
  aid = named_id(mj18, "actuator", "fl_hip_joint")
  dof = int(mj18.jnt_dofadr[jid])
  g18 = v18_env.sim.get_default_field("actuator_gainprm")
  check(
    "V18Range compiled model keeps the hand-set XML plant",
    abs(float(mj18.dof_damping[dof]) - 0.05) < 1e-9
    and abs(float(mj18.dof_armature[dof]) - 0.002) < 1e-9
    and abs(float(g18[aid, 0]) - 5.0) < 1e-6,
    f"damping {float(mj18.dof_damping[dof]):.4f}, "
    f"armature {float(mj18.dof_armature[dof]):.4f}, "
    f"kp {float(g18[aid, 0]):.3f}",
  )
  a18 = torch.zeros(
    v18_env.num_envs, v18_env.action_manager.total_action_dim, device=DEVICE
  )
  term18 = v18_env.event_manager.get_term_cfg("torque_speed_clamp").func
  v18_env.step(a18)
  strength18 = term18._tau_max / 0.22
  check(
    "V18Range env steps and captures knee-0 clamp (old effective envelope)",
    bool((term18._qd_knee == 0.0).all())
    and bool((term18._qd_max - strength18 * 4.5).abs().max() < 1e-5)
    and bool(torch.isfinite(v18_env.scene["robot"].data.joint_pos).all()),
    f"qd_knee all 0, qd_max/strength range "
    f"[{(term18._qd_max / strength18).min():.3f}, "
    f"{(term18._qd_max / strength18).max():.3f}] (nominal 4.5)",
  )
  v18_env.close()

  print()
  if _failures:
    print(f"OVERALL: FAIL ({len(_failures)} failed: {', '.join(_failures)})")
    raise SystemExit(1)
  print("OVERALL: PASS")


if __name__ == "__main__":
  main()
