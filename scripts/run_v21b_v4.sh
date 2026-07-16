#!/usr/bin/env bash
# v21b v4 campaign (2026-07-16): v3 + the entropy fix from the v3
# postmortem. v3's two fixes (ratio unlock gate, actor warm-start) WORKED
# — the warm walker opened at eplen 730 / tracking 0.31 and the grid never
# idle-inverted — but entropy_coef=0.01 (the from-scratch coef) inflated
# the loaded action std 0.14->0.41 over 2k iters; terrain hazard penalties
# on that noise tipped the per-step reward negative and early termination
# became reward-optimal (v19-v1 suicide trap): eplen 730->80 by iter 950.
#   v4 change: entropy_coef=0.001 (v21b.py xgolite_v21b_ppo_runner_cfg) —
#   fine-tuning regime; std moves by policy gradient, not entropy bonus.
# Designed to run DETACHED on the GPU box:
#
#   ssh dvv@192.168.1.49 'setsid nohup ~/dev/bots/luwu_mjlab/scripts/run_v21b_v4.sh \
#       >/dev/null 2>&1 </dev/null & disown'
#
# Training (10000 iters) is followed by:
#   1. FLAT-REGRESSION eval: the v21b checkpoint bucket-eval'd on the
#      XGOLite-V21A task env (actor contracts are identical 49-dim; the
#      loader is actor-only) with the wide-lateral buckets.
#   2. Gate-feasibility diagnostic on the flat V21A env.
#   3. Terrain survival eval: pinned commands x pinned difficulty rows
#      (0/3/6/9), survival + tracking per row -> eval_v21b_v4_terrain.log.
# V21B_V4_DONE marks completion. Progress in chain_v21b_v4_status.log.
set -u
cd "$(dirname "$0")/.."

LOG=chain_v21b_v4_status.log
WARM_START_CKPT=logs/rsl_rl/xgolite_v21a/2026-07-15_22-16-35_v21a_v1ext/model_4998.pt
echo "[chain] $(date +%FT%T) v21b_v4 chain pid $$ starting" >> "$LOG"

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
  echo "[chain] $(date +%FT%T) EVAL $name ckpt=$ckpt task=$task" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl EVAL_LATERAL_BUCKETS=1 \
    .venv/bin/python -u scripts/eval_policy_buckets.py \
    "$ckpt" "$task" > "eval_${name}.log" 2>&1
  echo "[chain] $(date +%FT%T) EVAL-END $name rc=$?" >> "$LOG"
}

run XGOLite-V21B 10000 v21b_v4 --warm-start-actor "$WARM_START_CKPT"

# FLAT-REGRESSION: the v21b (terrain-trained, blind 49-dim) checkpoint on
# the flat XGOLite-V21A task env — did terrain cost flat tracking?
evalb xgolite_v21b XGOLite-V21A v21b_v4_flat

CKPT=$(latest_ckpt xgolite_v21b)
if [ -n "$CKPT" ]; then
  # Gate feasibility + wobble floor on the flat env (control-rate obs
  # contract matches the v21b training world). asis variant only.
  echo "[chain] $(date +%FT%T) DIAG v21b_v4 ckpt=$CKPT" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u \
    scripts/diag_v19_gate_feasibility.py --task XGOLite-V21A --ckpt "$CKPT" \
    --obs control --variants asis > diag_v21b_v4.log 2>&1
  echo "[chain] $(date +%FT%T) DIAG-END v21b_v4 rc=$?" >> "$LOG"

  # Terrain survival: pinned commands on pinned difficulty rows.
  echo "[chain] $(date +%FT%T) TERRAIN-EVAL v21b_v4 ckpt=$CKPT" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u \
    scripts/eval_terrain_survival.py "$CKPT" XGOLite-V21B \
    > eval_v21b_v4_terrain.log 2>&1
  echo "[chain] $(date +%FT%T) TERRAIN-EVAL-END v21b_v4 rc=$?" >> "$LOG"
fi

date +%FT%T > V21B_V4_DONE
echo "[chain] $(date +%FT%T) CHAIN COMPLETE (V21B_V4_DONE written)" >> "$LOG"
