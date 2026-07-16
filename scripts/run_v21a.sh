#!/usr/bin/env bash
# v21a campaign (2026-07-15): gait discovery on the v20 plant — ORC
# phase-contact reward on a speed-scheduled per-env phase clock (1.6 Hz
# stand-adjacent -> 2.5 Hz at 0.45 m/s), lateral-dominant walk member
# (offsets 0.75/0.25/0.0/0.5, duty 0.75), morphological-symmetry reward
# instead of the trot-only mirror loss, vy widened to +-0.20 with 15%
# pure-lateral focus episodes. See v21a.py and
# docs/research/v21-terrain-gait-litreview-2026-07-15.md.
# Designed to run DETACHED on the GPU box:
#
#   ssh dvv@192.168.1.49 'setsid nohup ~/dev/bots/luwu_mjlab/scripts/run_v21a.sh \
#       >/dev/null 2>&1 </dev/null & disown'
#
# Training is followed by the sim bucket eval (with the v21a wide-lateral
# buckets, EVAL_LATERAL_BUCKETS=1) and the gate-feasibility diagnostic.
# V21A_DONE marks completion. Progress in chain_v21a_status.log.
set -u
cd "$(dirname "$0")/.."

LOG=chain_v21a_status.log
echo "[chain] $(date +%FT%T) v21a chain pid $$ starting" >> "$LOG"

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
  PYTHONPATH=. MUJOCO_GL=egl EVAL_LATERAL_BUCKETS=1 \
    .venv/bin/python -u scripts/eval_policy_buckets.py \
    "$ckpt" "$task" > "eval_${name}.log" 2>&1
  echo "[chain] $(date +%FT%T) EVAL-END $name rc=$?" >> "$LOG"
}

run XGOLite-V21A 2500 v21a_v1
evalb xgolite_v21a XGOLite-V21A v21a_v1

# Gate feasibility + wobble floor on the fresh checkpoint (control-rate
# obs contract matches the v21a training world). asis variant only.
CKPT=$(latest_ckpt xgolite_v21a)
if [ -n "$CKPT" ]; then
  echo "[chain] $(date +%FT%T) DIAG v21a_v1 ckpt=$CKPT" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u \
    scripts/diag_v19_gate_feasibility.py --task XGOLite-V21A --ckpt "$CKPT" \
    --obs control --variants asis > diag_v21a_v1.log 2>&1
  echo "[chain] $(date +%FT%T) DIAG-END v21a_v1 rc=$?" >> "$LOG"
fi

date +%FT%T > V21A_DONE
echo "[chain] $(date +%FT%T) CHAIN COMPLETE (V21A_DONE written)" >> "$LOG"
