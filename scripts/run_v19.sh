#!/usr/bin/env bash
# v19 campaign (2026-07-14): the proven v18range recipe on the MEASURED
# servo plant (Stage-1 system-ID fit) + wide friction/deadband DR.
# Designed to run DETACHED on the GPU box:
#
#   ssh dvv@192.168.1.49 'setsid nohup ~/dev/bots/luwu_mjlab/scripts/run_v19.sh \
#       >/dev/null 2>&1 </dev/null & disown'
#
# Training is followed by its sim bucket eval so verification data is ready
# when the chain ends; V19_DONE marks completion. Progress in
# chain_v19_status.log. A failed run is recorded and the chain moves on.
set -u
cd "$(dirname "$0")/.."

LOG=chain_v19_status.log
echo "[chain] $(date +%FT%T) v19 chain pid $$ starting" >> "$LOG"

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

evalb() {
  local exp=$1 task=$2 name=$3
  local rundir ckpt
  # Run dirs are <timestamp>_<run_name>; lexical sort == chronological.
  rundir=$(ls -d logs/rsl_rl/"$exp"/*/ 2>/dev/null | sort | tail -1)
  ckpt=$(ls "${rundir}"model_*.pt 2>/dev/null | sort -V | tail -1)
  if [ -z "$ckpt" ]; then
    echo "[chain] $(date +%FT%T) EVAL-SKIP $name (no checkpoint found)" >> "$LOG"
    return
  fi
  echo "[chain] $(date +%FT%T) EVAL $name ckpt=$ckpt" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u scripts/eval_policy_buckets.py \
    "$ckpt" "$task" > "eval_${name}.log" 2>&1
  echo "[chain] $(date +%FT%T) EVAL-END $name rc=$?" >> "$LOG"
}

# Measured plant + grid-adaptive command curriculum. History:
# v1: from scratch, v18 PPO -> suicide-attractor collapse.
# v2: init_std 0.4 -> collapsed too (noise was not the root cause).
# v3: joint_acc_l2 measured at control rate (root cause: the fitted
# relay's physics-rate dither made qacc-based joint_acc_l2 a ~-0.8/step
# tax on being alive; see v19.py point 8). Trained clean (reward +63,
# eplen ~985, zero falls in eval) but plateaued below the curriculum
# gate (seed_tracking ~0.13 vs 0.70/0.55) — grid never expanded.
# v3_ext: +2500 iterations resumed from v3 -> still plateaued
# (seed_tracking 0.13-0.16 for all 5000 iters; budget was not the issue).
# v4: actor joint_vel obs at CONTROL rate (v19.py point 9). Stopped
# early at iter ~1600: diag_v19_gate_feasibility.py proved the v18
# gates (0.70/0.55) unreachable on this plant (0/180 stochastic
# episodes pass, for v3_ext AND v4 checkpoints), so v4 could never
# unlock regardless of the obs fix. Bucket eval of model_1600 kept for
# the obs-fix comparison (eval_v19_v4.log).
# v5: v4 obs fix + unlock gates recalibrated to the measured plant's
# stochastic distributions (0.15/0.17, v19.py point 10). From scratch.
run XGOLite-V19 2500 v19_v5
evalb xgolite_v19 XGOLite-V19 v19_v5

date +%FT%T > V19_DONE
echo "[chain] $(date +%FT%T) CHAIN COMPLETE (V19_DONE written)" >> "$LOG"
