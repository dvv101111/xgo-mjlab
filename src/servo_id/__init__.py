"""Stage-1 servo-model system identification for the XGO-Lite2.

Fits the ToddlerBot-style actuator model (PD + piecewise torque-speed
clamp + Coulomb friction via native MJCF joint fields) to the capture
sessions produced by ``tools/servo_id_capture.py`` (parent repo), using
PACE-style CMA-ES over normalized parameters.

Modules:
  loader          capture npz / manifest loading, per-servo dedupe,
                  host->firmware clock alignment, segment slicing
  actuator_model  parameter spec, bounds, PD + torque-speed clamp torque law
  cma             minimal dependency-free CMA-ES (ask/tell)
  replay          fixed-base MuJoCo replay of recorded targets (ZOH) and
                  the position-MSE + FFT loss
  replay_warp     GPU-batched mujoco_warp replay backend (one world per
                  candidate x capture; same window/ZOH/loss code)
  fitting         CMA-ES orchestration (fork-pool CPU population eval, or
                  one batched warp rollout per generation)
  synthetic       synthetic capture generation for self-tests
"""

CANONICAL_JOINTS = (
  "fl_hip", "fl_thigh", "fl_calf",
  "fr_hip", "fr_thigh", "fr_calf",
  "bl_hip", "bl_thigh", "bl_calf",
  "br_hip", "br_thigh", "br_calf",
  "arm_shoulder", "arm_elbow", "claw",
)
LEG_JOINTS = CANONICAL_JOINTS[:12]
