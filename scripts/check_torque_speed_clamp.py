"""Validation for the sim-fidelity torque-speed clamp (piecewise knee form,
2026-07-14) + per-servo delay DR.

Builds a small XGOLite-V18Draft env (both fixes on), drives square-wave
extreme actions to sweep joint speeds through the servo envelope, and checks:

0. KNEE FORM: piecewise_tau_line is flat at tau_max below qd_knee, tapers
   linearly to zero at qd_max, and with qd_knee 0 reproduces the pre-v19
   single line tau_max * clip(1 - |qd|/qd_max, 0, 1) EXACTLY — so the v18
   presets (tau_max 0.22, qd_knee 0, qd_max 4.5) keep their old envelope.
1. CLAMP FORMULA: the forcerange written by the step event matches
   tau_line(qd) = piecewise_tau_line(|qd|, tau_max_i, qd_knee, qd_max)
   one-sidedly (driving side droops, braking side stays at tau_max_i).
2. PHYSICS ENVELOPE: the applied actuator force never leaves the bounds
   that were active during the step's physics substeps, i.e. driving force
   follows the torque-speed curve while braking force can reach tau_max.
3. ONE-SIDEDNESS: braking samples exceed the driving line at high |qd|
   (the clamp must NOT limit braking torque).
4. STRENGTH COMPOSITION: per-servo tau_max differs across servos within
   an env and across envs, consistent with the v17 strength DR ranges
   (per-env 0.75-1.25 x per-servo 0.90-1.10 => 0.675..1.375 x 0.22);
   the speed axis (qd_knee, qd_max) scales with the SAME strength factor
   (DC motor: stall torque AND no-load speed are both proportional to
   supply voltage, so the battery-sag proxy moves the whole envelope).
5. PER-SERVO DELAY: the robot actuator is the per-servo variant and, after
   enough refresh ticks, lags differ across servos within one env.
6. OPT-IN DEFAULT: XGOLite-Flat (v17) has no clamp event, keeps the plain
   per-env DelayedActuatorCfg, and its forcerange stays constant during
   stepping (bit-identical config path).

Usage:
  cd luwu_mjlab && PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python \
      scripts/check_torque_speed_clamp.py
"""

import torch

import mjlab.tasks  # noqa: F401  (registry)
import src.tasks  # noqa: F401

from mjlab.actuator.delayed_actuator import DelayedActuator, DelayedActuatorCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.mdp.actuators import (
  PerServoDelayedActuator,
  PerServoDelayedActuatorCfg,
)
from src.tasks.velocity.mdp.events import TorqueSpeedClamp, piecewise_tau_line

NUM_ENVS = 64
NUM_STEPS = 400          # 8 s at 50 Hz = 16 delay refresh ticks (0.5 s period)
SQUARE_HALF_PERIOD = 10  # control steps; 0.4 s full period forces reversals
ACTION_AMP = 3.0         # x0.25 scale = +/-0.75 rad target swings
# v18 envelope (sim_fidelity.V18_*): knee at 0 = the pre-v19 single line.
TAU_MAX = 0.22
QD_KNEE = 0.0
QD_MAX = 4.5

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  status = "PASS" if ok else "FAIL"
  print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
  if not ok:
    _failures.append(name)


def main() -> None:
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  torch.manual_seed(0)

  # ------------------------------------------- check 0: knee-form analytic --
  qd = torch.linspace(0.0, 15.0, 3001)
  # v18 equivalence: knee at 0 == the old single line. The two expressions
  # are algebraically identical; float32 rounding leaves ~1e-8 N*m residue
  # (7e-8 relative), far below sim force resolution.
  knee0 = piecewise_tau_line(qd, TAU_MAX, QD_KNEE, QD_MAX)
  old_line = TAU_MAX * (1.0 - qd / QD_MAX).clamp(0.0, 1.0)
  eq_err = (knee0 - old_line).abs().max().item()
  check(
    "knee form with qd_knee 0 == old single line (v18 envelope preserved)",
    eq_err < 1e-7,
    f"max |knee0 - old| = {eq_err:.2e} over qd in [0, 15]",
  )
  # Measured v19 shape: flat to the knee, linear taper, zero past qd_max.
  m_tau, m_knee, m_max = 0.22, 3.6468, 12.0939
  line = piecewise_tau_line(qd, m_tau, m_knee, m_max)
  below = qd <= m_knee
  beyond = qd >= m_max
  mid = (qd > m_knee) & (qd < m_max)
  taper = m_tau * (m_max - qd) / (m_max - m_knee)
  check(
    "knee form: flat tau_max below knee, linear taper, zero past qd_max",
    bool(
      ((line[below] - m_tau).abs().max() < 1e-8)
      and (line[beyond] == 0.0).all()
      and ((line[mid] - taper[mid]).abs().max() < 1e-7)
    ),
    f"tau({m_knee:.2f})={piecewise_tau_line(torch.tensor(m_knee), m_tau, m_knee, m_max):.4f}, "
    f"tau({0.5 * (m_knee + m_max):.2f})="
    f"{piecewise_tau_line(torch.tensor(0.5 * (m_knee + m_max)), m_tau, m_knee, m_max):.4f}, "
    f"tau({m_max:.2f})={piecewise_tau_line(torch.tensor(m_max), m_tau, m_knee, m_max):.4f}",
  )

  # --------------------------------------------------------------- env A --
  cfg = load_env_cfg("XGOLite-V18Draft")
  cfg.scene.num_envs = NUM_ENVS
  cfg.events.pop("push_robot", None)  # keep the sweep clean
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  env.reset()

  term = env.event_manager.get_term_cfg("torque_speed_clamp").func
  assert isinstance(term, TorqueSpeedClamp)
  robot = env.scene["robot"]
  actuator = robot.actuators[0]

  check(
    "per-servo actuator active in V18Draft",
    isinstance(actuator, PerServoDelayedActuator),
    f"type={type(actuator).__name__}",
  )

  # One step to trigger the clamp's lazy init (strength capture).
  action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)
  env.step(action)

  assert term._tau_max is not None
  assert term._qd_knee is not None and term._qd_max is not None
  assert term._joint_ids is not None and term._ctrl_ids is not None
  tau_max_i = term._tau_max  # (envs, 12)
  joint_ids = term._joint_ids
  ctrl_ids = term._ctrl_ids
  all_envs = torch.arange(env.num_envs, device=device, dtype=torch.long)

  # ------------------------------------------------- check 4: strength DR --
  strength = tau_max_i / TAU_MAX
  within_env_spread = (
    tau_max_i.max(dim=1).values - tau_max_i.min(dim=1).values
  ) / tau_max_i.mean(dim=1)
  check(
    "per-servo tau_max differs across servos (strength DR composed)",
    bool((within_env_spread > 0.01).all()),
    f"within-env max/min spread: min {within_env_spread.min():.3f}, "
    f"median {within_env_spread.median():.3f}",
  )
  env_means = tau_max_i.mean(dim=1)
  check(
    "per-env tau_max differs across envs",
    bool(((env_means.max() - env_means.min()) / env_means.mean()) > 0.02),
    f"env-mean range [{env_means.min():.4f}, {env_means.max():.4f}] N*m",
  )
  check(
    "strength factors inside DR envelope 0.675..1.375",
    bool((strength.min() >= 0.675 - 1e-4) and (strength.max() <= 1.375 + 1e-4)),
    f"strength range [{strength.min():.3f}, {strength.max():.3f}]",
  )
  check(
    "speed axis scales with the strength factor (DC motor: tau AND omega "
    "proportional to voltage)",
    isinstance(term._qd_knee, torch.Tensor)
    and isinstance(term._qd_max, torch.Tensor)
    and bool((term._qd_knee - strength * QD_KNEE).abs().max() < 1e-6)
    and bool((term._qd_max - strength * QD_MAX).abs().max() < 1e-6),
    f"qd_max/strength range "
    f"[{(term._qd_max / strength).min():.3f}, "
    f"{(term._qd_max / strength).max():.3f}] (nominal {QD_MAX})",
  )

  # ------------------------------------------- sweep with square-wave cmds --
  formula_err_max = 0.0
  force_violation_max = -1e9
  n_brake_above_line = 0
  brake_force_max = 0.0
  qd_abs_max = 0.0
  # Per-|qd| bin driving/braking envelopes for reporting.
  bins = torch.linspace(0.0, 9.0, 19, device=device)
  drive_env = torch.zeros(len(bins) - 1, device=device)
  brake_env = torch.zeros(len(bins) - 1, device=device)
  line_env = torch.full((len(bins) - 1,), float("nan"), device=device)

  prev_lower = env.sim.model.actuator_forcerange[all_envs[:, None], ctrl_ids, 0].clone()
  prev_upper = env.sim.model.actuator_forcerange[all_envs[:, None], ctrl_ids, 1].clone()
  prev_qd = robot.data.joint_vel[:, joint_ids].clone()

  for step in range(NUM_STEPS):
    sign = 1.0 if (step // SQUARE_HALF_PERIOD) % 2 == 0 else -1.0
    action.fill_(sign * ACTION_AMP)
    env.step(action)

    # Applied force from the last physics substep of this control step; the
    # forcerange active during ALL of this step's substeps is prev_*.
    force = env.sim.data.actuator_force[:, ctrl_ids]
    viol = torch.maximum(force - prev_upper, prev_lower - force).max().item()
    force_violation_max = max(force_violation_max, viol)

    # Braking-above-the-line evidence: force opposing the velocity that set
    # the bounds, with magnitude above the driving line at that speed.
    prev_line = piecewise_tau_line(
      prev_qd.abs(), tau_max_i, term._qd_knee, term._qd_max
    )
    braking = (torch.sign(force) * torch.sign(prev_qd) < 0) & (prev_qd.abs() > 1.0)
    above_line = braking & (force.abs() > prev_line + 0.10 * tau_max_i)
    n_brake_above_line += int(above_line.sum().item())
    if braking.any():
      brake_force_max = max(brake_force_max, force.abs()[braking].max().item())

    # Envelope per |qd_prev| bin (driving vs braking force magnitude).
    flat_qd = prev_qd.abs().flatten()
    flat_force = force.flatten()
    flat_drive = (torch.sign(flat_force) * torch.sign(prev_qd.flatten())) > 0
    bin_idx = torch.bucketize(flat_qd, bins[1:-1])
    for b in range(len(bins) - 1):
      sel = bin_idx == b
      if sel.any():
        d = flat_force.abs()[sel & flat_drive]
        k = flat_force.abs()[sel & ~flat_drive]
        if d.numel():
          drive_env[b] = torch.maximum(drive_env[b], d.max())
        if k.numel():
          brake_env[b] = torch.maximum(brake_env[b], k.max())

    # Formula check on the freshly written bounds vs current qd.
    qd = robot.data.joint_vel[:, joint_ids]
    qd_abs_max = max(qd_abs_max, qd.abs().max().item())
    tau_line = piecewise_tau_line(
      qd.abs(), tau_max_i, term._qd_knee, term._qd_max
    )
    exp_upper = torch.where(qd >= 0, tau_line, tau_max_i)
    exp_lower = torch.where(qd >= 0, -tau_max_i, -tau_line)
    lower = env.sim.model.actuator_forcerange[all_envs[:, None], ctrl_ids, 0]
    upper = env.sim.model.actuator_forcerange[all_envs[:, None], ctrl_ids, 1]
    err = torch.maximum((upper - exp_upper).abs(), (lower - exp_lower).abs())
    formula_err_max = max(formula_err_max, err.max().item())

    prev_lower = lower.clone()
    prev_upper = upper.clone()
    prev_qd = qd.clone()

  mid = 0.5 * (bins[:-1] + bins[1:])
  line_env = piecewise_tau_line(mid, TAU_MAX, QD_KNEE, QD_MAX)
  print("\n|qd| bin mid [rad/s] | nominal line | max driving | max braking")
  for b in range(len(bins) - 1):
    print(
      f"  {mid[b]:5.2f}              |  {line_env[b]:.4f}      |"
      f"  {drive_env[b]:.4f}     |  {brake_env[b]:.4f}"
    )

  check(
    "written forcerange follows one-sided tau_line formula",
    formula_err_max < 1e-5,
    f"max |written - expected| = {formula_err_max:.2e}",
  )
  check(
    "applied force stays inside the active (previous-step) bounds",
    force_violation_max < 1e-4,
    f"max bound violation = {force_violation_max:.2e} N*m",
  )
  check(
    "joint speeds actually swept the droop region",
    qd_abs_max > QD_MAX,
    f"max |qd| = {qd_abs_max:.2f} rad/s (qd_max = {QD_MAX})",
  )
  check(
    "braking force exceeds the driving line (clamp is one-sided)",
    n_brake_above_line > 0,
    f"{n_brake_above_line} braking samples above line, "
    f"max braking |force| = {brake_force_max:.3f} N*m",
  )

  # ------------------------------------------- check 5: per-servo delays --
  buf = actuator._delay_buffers["position"]
  lags = buf.current_lags
  check(
    "delay lags are per (env, servo)",
    lags.shape == (NUM_ENVS, len(joint_ids)),
    f"shape {tuple(lags.shape)}",
  )
  in_range = ((lags == 0) | ((lags >= 30) & (lags <= 50))).all()
  # Envs that reset (fell) just before the end of the sweep have all lags
  # back at the post-reset 0 — no refresh tick yet, nothing to compare;
  # excluding them removes the falls-timing flakiness of this check.
  refreshed = (lags > 0).any(dim=1)
  distinct = torch.tensor(
    [len(torch.unique(lags[e])) for e in range(NUM_ENVS)], dtype=torch.float
  )
  frac_multi = (distinct[refreshed.cpu()] >= 2).float().mean().item()
  nonzero_frac = (lags > 0).float().mean().item()
  check(
    "per-servo lag draws differ across servos within an env",
    frac_multi > 0.9,
    f"{frac_multi * 100:.0f}% of refreshed envs ({int(refreshed.sum())}/"
    f"{NUM_ENVS}) have >=2 distinct lags; "
    f"{nonzero_frac * 100:.0f}% of servo lags refreshed (in 30..50)",
  )
  check("lags stay in the configured 30..50 range (or pre-refresh 0)", bool(in_range))

  env.close()
  del env

  # --------------------------------------------------------------- env B --
  # Default path: v17 must be untouched by the opt-in fixes.
  flat_cfg = load_env_cfg("XGOLite-Flat")
  check(
    "XGOLite-Flat has no torque_speed_clamp event",
    "torque_speed_clamp" not in flat_cfg.events,
  )
  flat_act_cfg = flat_cfg.scene.entities["robot"].articulation.actuators[0]
  check(
    "XGOLite-Flat keeps plain per-env DelayedActuatorCfg",
    type(flat_act_cfg) is DelayedActuatorCfg
    and not isinstance(flat_act_cfg, PerServoDelayedActuatorCfg),
    f"type={type(flat_act_cfg).__name__}",
  )

  flat_cfg.scene.num_envs = 8
  flat_cfg.events.pop("push_robot", None)
  flat_env = ManagerBasedRlEnv(cfg=flat_cfg, device=device, render_mode=None)
  flat_env.reset()
  flat_act = flat_env.scene["robot"].actuators[0]
  check(
    "XGOLite-Flat actuator instance is the plain per-env DelayedActuator",
    isinstance(flat_act, DelayedActuator)
    and not isinstance(flat_act, PerServoDelayedActuator),
    f"type={type(flat_act).__name__}",
  )
  fr0 = flat_env.sim.model.actuator_forcerange.clone()
  flat_action = torch.zeros(
    flat_env.num_envs, flat_env.action_manager.total_action_dim, device=device
  )
  for step in range(20):
    sign = 1.0 if (step // SQUARE_HALF_PERIOD) % 2 == 0 else -1.0
    flat_action.fill_(sign * ACTION_AMP)
    flat_env.step(flat_action)
  fr1 = flat_env.sim.model.actuator_forcerange.clone()
  check(
    "XGOLite-Flat forcerange is static during stepping (no clamp)",
    bool(torch.equal(fr0, fr1)),
  )
  flat_env.close()

  print()
  if _failures:
    print(f"OVERALL: FAIL ({len(_failures)} failed: {', '.join(_failures)})")
    raise SystemExit(1)
  print("OVERALL: PASS")


if __name__ == "__main__":
  main()
