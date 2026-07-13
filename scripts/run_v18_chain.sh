#!/usr/bin/env bash
# v18 full-range campaign chain (2026-07-12): actuator-fidelity fixes +
# grid-adaptive command curriculum + Precision fine-tune from v17.
# Designed to run DETACHED on the GPU box:
#
#   ssh dvv@192.168.1.49 'setsid nohup ~/dev/bots/luwu_mjlab/scripts/run_v18_chain.sh \
#       >/dev/null 2>&1 </dev/null & disown'
#
# One training at a time; each run is followed by its sim bucket eval so
# verification data is ready when the chain ends. Progress in
# chain_v18_status.log. A failed run is recorded and the chain moves on.
set -u
cd "$(dirname "$0")/.."

LOG=chain_v18_status.log
echo "[chain] $(date +%FT%T) v18 chain pid $$ starting" >> "$LOG"

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

# 1. v17 recipe retrained under the torque-speed clamp + per-servo delay.
run XGOLite-V18Draft 1500 v18base_v1
evalb xgolite_v18draft XGOLite-V18Draft v18base_v1

# 2. Same, plus the grid-adaptive command curriculum (extra iterations so
#    the grid has time to expand from the seed region to the full envelope).
run XGOLite-V18Range 2500 v18range_v1
evalb xgolite_v18range XGOLite-V18Range v18range_v1

# 3. Precision rewards + fidelity fixes, fine-tuned FROM v17 (the staged
#    copy of xgolite_velocity/2026-07-11_14-16-46 inside this experiment
#    dir) instead of from scratch — the from-scratch run collapsed to
#    walking in place (no tracking gradient at tight relative sigma).
run XGOLite-Precision2 1500 precision2_ft_v1 \
  --agent.resume --agent.load-run 2026-07-11_14-16-46 \
  --agent.load-checkpoint model_1499.pt
evalb xgolite_precision2 XGOLite-Precision2 precision2_ft_v1

echo "[chain] $(date +%FT%T) CHAIN COMPLETE" >> "$LOG"
