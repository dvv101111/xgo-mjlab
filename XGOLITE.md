# XGO-Lite2 fork notes

This fork (github: dvv101111/xgo-mjlab, branch `xgolite`) adds XGO-Lite2
locomotion training on top of upstream
[LuwuDynamics/luwu_mjlab](https://github.com/LuwuDynamics/luwu_mjlab)
(remote `upstream`). The upstream README describes their XGOMini setup
(Ubuntu 22.04 / conda / python 3.11); this fork runs differently:

## Environment

uv-managed venv, not conda (WSL2 host, python 3.14 locally; 3.13 on the
training box):

    uv venv --python 3.13 .venv
    uv pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
    uv pip install -e . rsl-rl-lib==5.0.1 onnx==1.22.0 onnxscript==0.7.1

## Robot model

`src/assets/robots/xgolite/xgolite.xml` is GENERATED — do not hand-edit:

    .venv/bin/python src/assets/robots/xgolite/build_model.py

The generator derives kinematics, meshes, joint ranges, and foot pads
from the hardware-verified lite2 URDF and `config/servo_calibration.json`
in the parent repo (this repo must live at `Quadruped-robot/luwu_mjlab`
for the relative paths to resolve). It validates FK and mesh placement
against the URDF to sub-millimeter accuracy. Re-run it after any URDF or
servo-calibration change.

Actuators: kp 5.0 / kv 0.12 (fit from hardware step responses), torque
limit 0.22 N·m, position targets delayed 60-100 ms per env
(`DelayedActuatorCfg`) to match the measured command->motion latency of
the real servo bus.

## Train / evaluate / deploy

    # train (XGOLite-Flat task; ~0.8 s/iter at 4096 envs on an RTX 5090)
    .venv/bin/python scripts/train.py XGOLite-Flat --env.scene.num-envs 4096 --agent.max-iterations 1500

    # headless eval: fall rate + velocity tracking under full DR
    .venv/bin/python scripts/eval_policy.py logs/rsl_rl/xgolite_velocity/<run>/model_<n>.pt

    # inspect the model standing in a floor scene
    MUJOCO_GL=egl .venv/bin/python -m mujoco.viewer --mjcf=src/assets/robots/xgolite/xmls/scene.xml

Training exports `policy.onnx` (actor + baked obs normalizer + metadata:
joint order, default pose, action scale, kp/kv). The parent repo's
`tools/run_policy.py` deploys it to the robot at 50 Hz over the
open-firmware link.

## Contract summary

47-dim observation: gyro(3) + projected gravity(3) + command(3) +
gait phase(2) + joint pos rel(12) + joint vel(12) + previous action(12);
12-dim action, target = default_pose + 0.25 * action; control 50 Hz,
gait period 0.4 s; stand pose thigh -0.90 / calf +0.28 (hip height
116 mm).
