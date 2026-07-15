#!/usr/bin/env bash
# v20 campaign (2026-07-15): the v18range recipe on the fit-v4 CORRECTED
# measured plant (soft damped spring kp 5.59 — the v3 stiff relay was a
# v2.3 telemetry artifact) + CAD-derived inertials (577 g total, light
# legs). Keeps v19's deploy-aligned control-rate obs/penalty; reverts the
# relay-sized knobs (init_std 1.0, unlock gates 0.70/0.55). See v20.py.
# Designed to run DETACHED on the GPU box:
#
#   ssh dvv@192.168.1.49 'setsid nohup ~/dev/bots/luwu_mjlab/scripts/run_v20.sh \
#       >/dev/null 2>&1 </dev/null & disown'
#
# Training is followed by the sim bucket eval and the gate-feasibility /
# wobble-floor diagnostic (the plant-correction headline: v3-plant stand
# yaw wobble was 0.13 rad/s vs the old empirical plant's ~0.03).
# V20_DONE marks completion. Progress in chain_v20_status.log.
set -u
cd "$(dirname "$0")/.."

LOG=chain_v20_status.log
echo "[chain] $(date +%FT%T) v20 chain pid $$ starting" >> "$LOG"

run() {
  local task=$1 iters=$2 name=$3
  shift 3
  echo "[chain] $(date +%FT%T) START $task ($iters iters, 4096 envs) -> train_${name}.log" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u scripts/train.py "$task" \
    --env.scene.num-envs 4096 --agent.max-iterations "$iters" \
    --agent.run-name "$name" "$@" > "train_${name}.log" 2>&1
  local rc=$?
  echo "[chain] $(date +%FT%T) END $task rc=$rc" >> "$LOG"
}

latest_ckpt() {
  local exp=$1 rundir
  rundir=$(ls -d logs/rsl_rl/"$exp"/*/ 2>/dev/null | sort | tail -1)
  ls "${rundir}"model_*.pt 2>/dev/null | sort -V | tail -1
}

evalb() {
  local exp=$1 task=$2 name=$3
  local ckpt
  ckpt=$(latest_ckpt "$exp")
  if [ -z "$ckpt" ]; then
    echo "[chain] $(date +%FT%T) EVAL-SKIP $name (no checkpoint found)" >> "$LOG"
    return
  fi
  echo "[chain] $(date +%FT%T) EVAL $name ckpt=$ckpt" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u scripts/eval_policy_buckets.py \
    "$ckpt" "$task" > "eval_${name}.log" 2>&1
  echo "[chain] $(date +%FT%T) EVAL-END $name rc=$?" >> "$LOG"
}

run XGOLite-V20 2500 v20_v1
evalb xgolite_v20 XGOLite-V20 v20_v1

# Wobble floor + gate feasibility on the fresh checkpoint (control-rate
# obs contract matches the v20 training world). asis variant only: the
# headline numbers are the stand yaw wobble and the natural-roll gate
# distribution vs the restored (0.70, 0.55) gates.
CKPT=$(latest_ckpt xgolite_v20)
if [ -n "$CKPT" ]; then
  echo "[chain] $(date +%FT%T) DIAG v20_v1 ckpt=$CKPT" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u \
    scripts/diag_v19_gate_feasibility.py --task XGOLite-V20 --ckpt "$CKPT" \
    --obs control --variants asis > diag_v20_v1.log 2>&1
  echo "[chain] $(date +%FT%T) DIAG-END v20_v1 rc=$?" >> "$LOG"
fi

date +%FT%T > V20_DONE
echo "[chain] $(date +%FT%T) CHAIN COMPLETE (V20_DONE written)" >> "$LOG"
