"""Validation for the per-servo deadband (lost motion / backlash) support
(2026-07-13 servo-ID finding, staged for the v19 preset).

Builds small XGOLite-V18Draft envs with the deadband opted in and checks:

1. OPT-IN DEFAULT: V18Draft's per-servo actuator cfg carries
   deadband_range (0.0, 0.0) and allocates no deadband tensor — the
   feature is structurally off unless a preset enables it.
2. FAIL FAST: enable_servo_deadband on a cfg without the per-servo
   delayed actuator (XGOLite-Flat) raises TypeError.
3. FORMULA + COMPOSITION: with the delay buffer pinned to a fixed lag L
   and the deadband forced to a known constant, manual actuator.compute
   calls return ctrl = q + (err - clip(err, -db, +db)) computed against
   the POST-delay target (the target from L physics steps ago), exactly.
4. ATTENUATION: a small sinusoid commanded on one joint delivers an
   amplitude reduced by ~db versus a deadband-off baseline env
   (delivered ~ commanded - db; the hardware sine sweeps showed the
   same frequency-flat small-amplitude attenuation).
5. DRAW / RESET DR: with the default 0.015-0.035 rad range, deadbands
   are per (env, servo), stay in range, vary across servos, and are
   redrawn for reset envs only.
6. OFF = BIT-IDENTICAL: a V18Draft env with enable_servo_deadband(cfg,
   (0.0, 0.0)) replays the exact same joint-position trajectory as a
   plain V18Draft env under identical seeding and actions (no extra RNG
   consumption, no code-path change).

Usage:
  cd luwu_mjlab && PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python \
      scripts/check_deadband.py
"""

import math

import torch

import mjlab.tasks  # noqa: F401  (registry)
import src.tasks  # noqa: F401

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.config.xgolite.sim_fidelity import (
  DEADBAND_RANGE,
  enable_servo_deadband,
)
from src.tasks.velocity.mdp.actuators import PerServoDelayedActuator

NUM_ENVS = 32
SINE_JOINT = "fl_thigh_joint"
ACTION_SCALE = 0.25       # env_cfgs.py joint_pos_action.scale
SINE_AMP = 0.20           # rad, commanded joint amplitude
SINE_FREQ = 0.5           # Hz; 100 control steps per period at 50 Hz
SETTLE_STEPS = 100
MEASURE_STEPS = 400       # 4 full periods
DB_FORCED = 0.05          # rad, forced deadband for formula + sine checks
PIN_LAG = 40              # physics steps, inside the 30..50 cfg range
FORMULA_LAG = 3
BITIDENT_STEPS = 100
SEED = 0

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  status = "PASS" if ok else "FAIL"
  print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
  if not ok:
    _failures.append(name)


def make_env(deadband_range: tuple[float, float] | None, num_envs: int):
  """Fresh V18Draft env, deterministic seed, push events removed."""
  torch.manual_seed(SEED)
  cfg = load_env_cfg("XGOLite-V18Draft")
  cfg.scene.num_envs = num_envs
  cfg.events.pop("push_robot", None)
  if deadband_range is not None:
    enable_servo_deadband(cfg, deadband_range)
  env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE, render_mode=None)
  env.reset()
  return env


def pin_lags(actuator: PerServoDelayedActuator, lag: int) -> None:
  """Make the delay deterministic: every draw and current lag becomes `lag`."""
  for buf in actuator._delay_buffers.values():
    buf.min_lag = lag
    buf.max_lag = lag
    buf.set_lags(
      torch.full(
        (buf.batch_size, buf.num_targets), lag, dtype=torch.long, device=buf.device
      )
    )


def sine_rollout(env, joint_id: int, action_index: int) -> torch.Tensor:
  """Drive a sine on one joint; return per-env delivered amplitude (no-reset envs only)."""
  action = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=DEVICE
  )
  robot = env.scene["robot"]
  ever_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=DEVICE)
  history = []
  for step in range(SETTLE_STEPS + MEASURE_STEPS):
    t = step * env.step_dt
    action[:, action_index] = (SINE_AMP / ACTION_SCALE) * math.sin(
      2.0 * math.pi * SINE_FREQ * t
    )
    env.step(action)
    ever_reset |= env.reset_buf.bool()
    if step >= SETTLE_STEPS:
      history.append(robot.data.joint_pos[:, joint_id].clone())
  q = torch.stack(history)  # (steps, envs)
  amp = 0.5 * (q.max(dim=0).values - q.min(dim=0).values)
  if ever_reset.any():
    print(f"  note: excluding {int(ever_reset.sum())} env(s) that reset mid-sine")
  return amp[~ever_reset]


def main() -> None:
  configure_torch_backends()
  torch.manual_seed(SEED)

  # ------------------------------------------------ env ON (forced db) ----
  env = make_env((DB_FORCED, DB_FORCED), NUM_ENVS)
  robot = env.scene["robot"]
  actuator = robot.actuators[0]
  check(
    "deadband env uses PerServoDelayedActuator",
    isinstance(actuator, PerServoDelayedActuator),
    f"type={type(actuator).__name__}",
  )
  db = actuator._deadband
  check(
    "forced deadband tensor is per (env, servo) at the forced value",
    db is not None
    and db.shape == (NUM_ENVS, 12)
    and bool(torch.all(db == DB_FORCED)),
    f"shape {tuple(db.shape) if db is not None else None}",
  )

  act_term = env.action_manager.get_term("joint_pos")
  target_names = list(act_term._target_names)
  assert SINE_JOINT in target_names, (
    f"{SINE_JOINT!r} not in action targets {target_names}"
  )
  action_index = target_names.index(SINE_JOINT)
  joint_id = int(act_term._target_ids[action_index].item())

  # ------------------------------------- check 3: formula + composition ---
  # Pin the delay to a known constant lag, then drive compute() by hand so
  # the post-delay target is exactly the target from FORMULA_LAG calls ago.
  pin_lags(actuator, FORMULA_LAG)
  num_servos = db.shape[1]
  pos = torch.full((NUM_ENVS, num_servos), 0.10, device=DEVICE)
  vel = torch.zeros_like(pos)
  zeros = torch.zeros_like(pos)
  # Per-servo spread so both in-band (|err| < db) and out-of-band errors
  # are exercised: err in roughly -0.09 .. +0.09 rad vs db = 0.05.
  spread = torch.linspace(0.5, 1.5, num_servos, device=DEVICE).unsqueeze(0)
  n_calls = 12
  targets = [pos + (k - 5) * 0.012 * spread for k in range(n_calls)]
  outs = []
  for k in range(n_calls):
    cmd = ActuatorCmd(
      position_target=targets[k],
      velocity_target=zeros,
      effort_target=zeros,
      pos=pos,
      vel=vel,
    )
    outs.append(actuator.compute(cmd))
  err_max = 0.0
  for k in range(FORMULA_LAG, n_calls):
    err = targets[k - FORMULA_LAG] - pos
    expected = pos + (err - err.clamp(min=-db, max=db))
    err_max = max(err_max, (outs[k] - expected).abs().max().item())
  check(
    "ctrl == q + (err - clip(err, -db, +db)) on the POST-delay target",
    err_max < 1e-6,
    f"max |ctrl - expected| = {err_max:.2e} over lags of {FORMULA_LAG} steps",
  )

  # ----------------------------------------- check 4: sine attenuation ----
  env.reset()
  pin_lags(actuator, PIN_LAG)
  amp_on = sine_rollout(env, joint_id, action_index)
  env.close()
  del env

  env_off = make_env(None, NUM_ENVS)
  actuator_off = env_off.scene["robot"].actuators[0]
  check(
    "V18Draft default cfg keeps deadband off",
    actuator_off.cfg.deadband_range == (0.0, 0.0)
    and actuator_off._deadband is None,
    f"deadband_range={actuator_off.cfg.deadband_range}",
  )
  pin_lags(actuator_off, PIN_LAG)
  amp_off = sine_rollout(env_off, joint_id, action_index)
  env_off.close()
  del env_off

  mean_on = amp_on.mean().item()
  mean_off = amp_off.mean().item()
  diff = mean_off - mean_on
  print(
    f"  commanded amp {SINE_AMP:.3f} rad | delivered off {mean_off:.4f} "
    f"| delivered on {mean_on:.4f} | diff {diff:.4f} (db = {DB_FORCED})"
  )
  check(
    "deadband attenuates the delivered sine amplitude by ~db",
    0.5 * DB_FORCED <= diff <= 1.5 * DB_FORCED,
    f"amp_off - amp_on = {diff:.4f}, expected ~{DB_FORCED}",
  )
  check(
    "delivered ~ commanded - db with deadband on",
    mean_on < SINE_AMP - 0.4 * DB_FORCED,
    f"delivered {mean_on:.4f} vs commanded {SINE_AMP:.3f}",
  )

  # ------------------------------------------- check 5: draw / reset DR ---
  env_dr = make_env(DEADBAND_RANGE, 16)
  act_dr = env_dr.scene["robot"].actuators[0]
  db_dr = act_dr._deadband
  assert db_dr is not None
  lo, hi = DEADBAND_RANGE
  check(
    "default-range draws stay inside DEADBAND_RANGE",
    bool((db_dr >= lo).all() and (db_dr <= hi).all()),
    f"range [{db_dr.min():.4f}, {db_dr.max():.4f}] vs cfg [{lo}, {hi}]",
  )
  within_env = db_dr.max(dim=1).values - db_dr.min(dim=1).values
  check(
    "draws differ across servos within an env",
    bool((within_env > 1e-4).all()),
    f"min within-env spread {within_env.min():.5f} rad",
  )
  before = db_dr.clone()
  reset_ids = torch.arange(8, device=DEVICE, dtype=torch.long)
  env_dr.scene["robot"].reset(reset_ids)
  after = act_dr._deadband
  check(
    "reset redraws deadbands for reset envs only",
    bool(not torch.equal(before[:8], after[:8]))
    and bool(torch.equal(before[8:], after[8:])),
  )
  env_dr.close()
  del env_dr

  # -------------------------------------- check 6: off = bit-identical ----
  # Three runs: plain twice (determinism baseline for the GPU sim itself)
  # and deadband-off once. The off run must consume the exact same RNG
  # stream, and its trajectory must match the plain runs at least as well
  # as they match each other.
  gen = torch.Generator().manual_seed(123)
  actions_seq = (
    torch.rand((BITIDENT_STEPS, 8, 12), generator=gen) * 0.4 - 0.2
  ).to(DEVICE)

  def rollout(db_range):
    env_b = make_env(db_range, 8)
    robot_b = env_b.scene["robot"]
    traj = []
    for step in range(BITIDENT_STEPS):
      env_b.step(actions_seq[step])
      traj.append(robot_b.data.joint_pos.clone())
    rng_sig = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(DEVICE).clone())
    env_b.close()
    return torch.stack(traj), rng_sig

  traj_plain1, rng_plain1 = rollout(None)
  traj_plain2, _ = rollout(None)
  traj_off, rng_off = rollout((0.0, 0.0))

  check(
    "deadband-off run consumes the exact same RNG stream as plain",
    bool(torch.equal(rng_plain1[0], rng_off[0]))
    and bool(torch.equal(rng_plain1[1], rng_off[1])),
  )
  baseline_diff = (traj_plain1 - traj_plain2).abs().max().item()
  off_diff = (traj_plain1 - traj_off).abs().max().item()
  if baseline_diff == 0.0:
    check(
      "deadband_range (0.0, 0.0) is bit-identical to no deadband",
      off_diff == 0.0,
      f"max |dq| = {off_diff:.2e}",
    )
  else:
    # GPU sim itself is not bitwise-reproducible on this machine; require
    # the off run to sit within the sim's own repeat noise.
    check(
      "deadband_range (0.0, 0.0) matches plain within sim repeat noise",
      off_diff <= max(10.0 * baseline_diff, 1e-6),
      f"off diff {off_diff:.2e} vs plain-vs-plain repeat diff {baseline_diff:.2e}",
    )

  # ------------------------------------------------ check 2: fail fast ----
  flat_cfg = load_env_cfg("XGOLite-Flat")
  try:
    enable_servo_deadband(flat_cfg)
    raised = False
  except TypeError:
    raised = True
  check(
    "enable_servo_deadband without per-servo delay raises TypeError",
    raised,
  )

  print()
  if _failures:
    print(f"OVERALL: FAIL ({len(_failures)} failed: {', '.join(_failures)})")
    raise SystemExit(1)
  print("OVERALL: PASS")


if __name__ == "__main__":
  main()
