"""Fixed-timestep MuJoCo replay of recorded servo-ID excitations.

Primary regime (all 2026-07-13 sessions): LOADED — the robot standing on
the floor. The replay therefore uses the full robot with a FREE base and a
ground plane; every non-swept joint is PD-held at the recorded base pose
while the swept joint follows the recorded targets ZOH, exactly like the
capture. Before the scored window the sim settles for ``settle_s`` at the
base pose from the stand keyframe height, mirroring the capture tool's
settle phase, so contacts and posture reach their equilibrium under the
candidate parameters.

A fixed-base variant (no floor, gravity vector overridable from the IMU
reading) is kept for a possible future unloaded (folded / on-back)
regime, but is not the default.

Timeline: everything runs on the firmware clock (ms). The command stream
is ZOH: at sim time t the target is the last command with
``cmd_time_ms <= t - delay_ms`` (before the first command: the base
pose). Sim positions are compared against real telemetry at the real
(deduped) per-servo sample timestamps by nearest-step lookup (sim dt 2 ms
vs 1 ms timestamp resolution).

Actuation per physics step: PD + asymmetric torque-speed clamp
(:mod:`src.servo_id.actuator_model`) applied via ``qfrc_applied``; the
XML ``<position>`` actuators are removed so no double actuation occurs.
Coulomb/viscous/rotor terms use the native MJCF dof fields, written per
candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from src.servo_id import LEG_JOINTS
from src.servo_id.actuator_model import ServoParams
from src.servo_id.loader import TestCapture

XGOLITE_XML = (
  Path(__file__).resolve().parents[1]
  / "assets" / "robots" / "xgolite" / "xmls" / "xgolite.xml"
)

SIM_DT = 0.002  # s, matches the training stack's physics timestep
SETTLE_S = 1.0  # pre-roll at the base pose before the scored window
LOSS_SKIP_MS = 300.0  # scored window starts this far into the replay
FFT_WEIGHT = 0.01
FFT_FMAX_HZ = 10.0
FFT_GRID_HZ = 100.0


@dataclass
class ReplayModel:
  """Compiled replay model + canonical-order joint addressing."""

  model: mujoco.MjModel
  qadr: np.ndarray  # (12,) qpos address per canonical leg joint
  dadr: np.ndarray  # (12,) dof address per canonical leg joint
  fixed_base: bool


@dataclass
class SimTraj:
  """One rollout: sim joint positions on a uniform step grid (fw clock)."""

  t0_ms: float  # firmware time of traj[0]
  dt_ms: float
  q: np.ndarray  # (n_steps, 12) float32, canonical leg order

  def at(self, times_ms: np.ndarray, joint_col: int) -> np.ndarray:
    """Nearest-step sim positions at the given firmware times."""
    idx = np.rint((np.asarray(times_ms, dtype=float) - self.t0_ms)
                  / self.dt_ms).astype(int)
    idx = np.clip(idx, 0, self.q.shape[0] - 1)
    return self.q[idx, joint_col].astype(float)


def build_replay_model(fixed_base: bool = False) -> ReplayModel:
  """Compile the replay model from the repo MJCF.

  free base (default): keeps the floating joint and adds a ground plane.
  fixed base: deletes the floating joint (base welded at the XML height)
  and disables all contacts; used only for unloaded captures.
  """
  spec = mujoco.MjSpec.from_file(str(XGOLITE_XML))
  for act in list(spec.actuators):
    spec.delete(act)
  if fixed_base:
    for joint in list(spec.joints):
      if joint.name == "floating_base":
        spec.delete(joint)
  else:
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
  model = spec.compile()
  if fixed_base:
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
  qadr, dadr = [], []
  for name in LEG_JOINTS:
    joint = model.joint(f"{name}_joint")
    qadr.append(int(joint.qposadr[0]))
    dadr.append(int(joint.dofadr[0]))
  return ReplayModel(
    model=model,
    qadr=np.asarray(qadr),
    dadr=np.asarray(dadr),
    fixed_base=fixed_base,
  )


def set_joint_dynamics(
  rm: ReplayModel, params: list[ServoParams]
) -> tuple[np.ndarray, ...]:
  """Write MJCF dof fields; return stacked torque-law arrays (12,)."""
  assert len(params) == 12
  for i, p in enumerate(params):
    dof = rm.dadr[i]
    rm.model.dof_damping[dof] = p.damping
    rm.model.dof_armature[dof] = p.armature
    rm.model.dof_frictionloss[dof] = p.frictionloss
  kp = np.array([p.kp for p in params])
  kd = np.array([p.kd for p in params])
  deadband = np.array([p.deadband for p in params])
  tau_max = np.array([p.tau_max for p in params])
  qd_knee = np.array([p.qd_knee for p in params])
  qd_max = np.array([p.qd_max for p in params])
  return kp, kd, deadband, tau_max, qd_knee, qd_max


def capture_window(capture: TestCapture) -> tuple[float, float]:
  """Replay window (t_lo, t_hi) on the firmware clock.

  From the earliest real sample (or first command, whichever is earlier)
  to the latest real sample, across the 12 leg joints (arm columns are
  not in the model). Shared by the CPU and warp backends so both replay
  bit-identical windows.
  """
  t_lo = min(
    float(s.t_ms[0]) for n, s in capture.samples.items()
    if n in LEG_JOINTS and s.t_ms.size
  )
  t_lo = min(t_lo, float(capture.cmd_time_ms[0]))
  t_hi = max(
    float(s.t_ms[-1]) for n, s in capture.samples.items()
    if n in LEG_JOINTS and s.t_ms.size
  )
  return t_lo, t_hi


def zoh_cmd_indices(
  cmd_t: np.ndarray,
  t_lo: float,
  n_replay: int,
  dt_ms: float,
  delay_ms: float,
) -> np.ndarray:
  """Per-step ZOH command index for the replay window (vectorized).

  Index -1 means "before the first delayed command" (hold the base pose).
  Shared by the CPU and warp backends.
  """
  t_steps = t_lo + dt_ms * np.arange(n_replay)
  return np.searchsorted(cmd_t, t_steps - delay_ms, side="right") - 1


def replay_capture(
  rm: ReplayModel,
  capture: TestCapture,
  params: list[ServoParams],
  delay_ms: float,
  sim_dt: float = SIM_DT,
  settle_s: float = SETTLE_S,
  data: mujoco.MjData | None = None,
  gravity_from_imu: bool = False,
) -> SimTraj:
  """Roll out one capture under the given per-joint parameters.

  ``params`` is the 12-entry canonical-order list; ``delay_ms`` is the
  single global command delay (firmware receipt -> actuation).
  """
  from src.servo_id.actuator_model import pd_clamped_torque

  model = rm.model
  if gravity_from_imu and rm.fixed_base and capture.gravity_body is not None:
    model.opt.gravity[:] = capture.gravity_body
  kp, kd, deadband, tau_max, qd_knee, qd_max = set_joint_dynamics(rm, params)

  base12 = capture.base_pose[:12]
  cmd12 = np.ascontiguousarray(capture.cmd_pose[:, :12])
  cmd_t = capture.cmd_time_ms

  t_lo, t_hi = capture_window(capture)
  dt_ms = sim_dt * 1000.0
  n_settle = int(round(settle_s / sim_dt))
  n_replay = int(np.ceil((t_hi - t_lo) / dt_ms)) + 1

  if data is None:
    data = mujoco.MjData(model)
  mujoco.mj_resetData(model, data)
  if not rm.fixed_base:
    # Stand keyframe height + level base; the settle pre-roll finds the
    # true contact equilibrium under the candidate parameters, mirroring
    # the capture tool's settle at the base pose.
    data.qpos[:] = 0.0
    data.qpos[2] = 0.1159
    data.qpos[3] = 1.0
  data.qpos[rm.qadr] = base12
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)

  cmd_idx = zoh_cmd_indices(cmd_t, t_lo, n_replay, dt_ms, delay_ms)

  traj = np.empty((n_replay, 12), dtype=np.float32)
  qadr, dadr = rm.qadr, rm.dadr
  for i in range(-n_settle, n_replay):
    q = data.qpos[qadr]
    qd = data.qvel[dadr]
    if i < 0 or cmd_idx[i] < 0:
      q_des = base12
    else:
      q_des = cmd12[cmd_idx[i]]
    data.qfrc_applied[dadr] = pd_clamped_torque(
      q, qd, q_des, kp, kd, deadband, tau_max, qd_knee, qd_max
    )
    if i >= 0:
      traj[i] = q
    mujoco.mj_step(model, data)
  return SimTraj(t0_ms=t_lo, dt_ms=dt_ms, q=traj)


def _fft_magnitude(x: np.ndarray, grid_hz: float, fmax_hz: float):
  """Length-normalized rfft magnitude below fmax (units: rad)."""
  mag = np.abs(np.fft.rfft(x)) / max(len(x), 1)
  freqs = np.fft.rfftfreq(len(x), d=1.0 / grid_hz)
  return mag[freqs < fmax_hz]


def capture_loss(
  traj: SimTraj,
  capture: TestCapture,
  joints: tuple[str, ...],
  skip_ms: float = LOSS_SKIP_MS,
  fft_weight: float = FFT_WEIGHT,
) -> float:
  """Time-averaged position MSE + weighted FFT-magnitude RMSE (< 10 Hz).

  Scored at the real (deduped) telemetry sample times of the given
  joints; the first ``skip_ms`` of the window is excluded (episode-init
  transient, spec pitfall). The FFT term compares length-normalized
  magnitude spectra on a uniform 100 Hz grid interpolated from the same
  samples (ToddlerBot loss, weight 0.01).
  """
  t_end = traj.t0_ms + traj.dt_ms * (traj.q.shape[0] - 1)
  t_start = traj.t0_ms + skip_ms
  losses = []
  for name in joints:
    col = LEG_JOINTS.index(name)
    s = capture.samples[name]
    mask = (s.t_ms >= t_start) & (s.t_ms <= t_end)
    if mask.sum() < 16:
      continue
    t_eval = s.t_ms[mask].astype(float)
    q_real = s.q[mask]
    q_sim = traj.at(t_eval, col)
    mse = float(np.mean((q_real - q_sim) ** 2))

    grid = np.arange(t_start, t_end, 1000.0 / FFT_GRID_HZ)
    real_g = np.interp(grid, t_eval, q_real)
    sim_g = traj.at(grid, col)
    mag_r = _fft_magnitude(real_g, FFT_GRID_HZ, FFT_FMAX_HZ)
    mag_s = _fft_magnitude(sim_g, FFT_GRID_HZ, FFT_FMAX_HZ)
    fft_rmse = float(np.sqrt(np.mean((mag_r - mag_s) ** 2)))
    losses.append(mse + fft_weight * fft_rmse)
  if not losses:
    return float("nan")
  return float(np.mean(losses))


def segment_metrics(
  traj: SimTraj,
  capture: TestCapture,
  skip_ms: float = LOSS_SKIP_MS,
) -> list[dict]:
  """Per-excitation-segment sim-vs-real diagnostics.

  RMSE for every segment; for sine segments additionally the amplitude
  ratio (response half peak-to-peak / commanded amplitude) of real and
  sim — the small-amplitude (0.08 rad) rows are the stiction/deadband
  validation target (measured real attenuation 0.77-0.91).
  """
  out = []
  t_start = traj.t0_ms + skip_ms
  t_end = traj.t0_ms + traj.dt_ms * (traj.q.shape[0] - 1)
  for seg in capture.segments:
    if seg.joint not in LEG_JOINTS:
      continue
    col = LEG_JOINTS.index(seg.joint)
    s = capture.samples[seg.joint]
    mask = (
      (s.t_ms >= max(seg.t0_ms, t_start)) & (s.t_ms <= min(seg.t1_ms, t_end))
    )
    if mask.sum() < 8:
      continue
    t_eval = s.t_ms[mask].astype(float)
    q_real = s.q[mask]
    q_sim = traj.at(t_eval, col)
    rec = {
      "file": capture.path.name,
      "joint": seg.joint,
      "kind": capture.kind,
      "desc": seg.desc,
      "n_samples": int(mask.sum()),
      "rmse": float(np.sqrt(np.mean((q_real - q_sim) ** 2))),
    }
    amp = seg.desc.get("amplitude")
    if amp and capture.kind == "sine_grid":
      def half_p2p(x: np.ndarray) -> float:
        return float(np.percentile(x, 97.5) - np.percentile(x, 2.5)) / 2.0

      rec["amp_cmd"] = float(amp)
      rec["amp_ratio_real"] = half_p2p(q_real) / float(amp)
      rec["amp_ratio_sim"] = half_p2p(q_sim) / float(amp)
    out.append(rec)
  return out
