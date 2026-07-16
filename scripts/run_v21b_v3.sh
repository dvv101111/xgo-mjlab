#!/usr/bin/env bash
# v21b v3 campaign (2026-07-16): rough-terrain retraining with the two
# v21b-v2 collapse fixes (docs/research/v19-postmortem-v4-plant-2026-07-15.md
# gate-metric section):
#   1. Achieved-velocity ratio AND-gate on the command-grid unlocks
#      (grid_curriculum.unlock_min_vel_ratio=0.5, wired in v21b.py) — idle
#      episodes can no longer unlock cells nobody tracks (the v19-v5
#      idle-inversion that unlocked 100% of the grid by iter 2000 at
#      seed_tracking 0.11).
#   2. Actor warm-start from the flat v21a-ext checkpoint
#      (--warm-start-actor, scripts/train.py): the ACTOR (49-dim contract,
#      identical shapes) starts from a competent flat walker so standing is
#      not the local optimum on lethal terrain; the critic (whose input
#      grew a 187-dim height_scan) and optimizer train fresh.
# Designed to run DETACHED on the GPU box:
#
#   ssh dvv@192.168.1.49 'setsid nohup ~/dev/bots/luwu_mjlab/scripts/run_v21b_v3.sh \
#       >/dev/null 2>&1 </dev/null & disown'
#
# Training (10000 iters) is followed by:
#   1. FLAT-REGRESSION eval: the v21b checkpoint bucket-eval'd on the
#      XGOLite-V21A task env (actor contracts are identical 49-dim; the
#      loader is actor-only) with the wide-lateral buckets.
#   2. Gate-feasibility diagnostic on the flat V21A env.
#   3. Terrain survival eval: pinned commands x pinned difficulty rows
#      (0/3/6/9), survival + tracking per row -> eval_v21b_v3_terrain.log.
# V21B_V3_DONE marks completion. Progress in chain_v21b_v3_status.log.
set -u
cd "$(dirname "$0")/.."

LOG=chain_v21b_v3_status.log
WARM_START_CKPT=logs/rsl_rl/xgolite_v21a/2026-07-15_22-16-35_v21a_v1ext/model_4998.pt
echo "[chain] $(date +%FT%T) v21b_v3 chain pid $$ starting" >> "$LOG"

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

run XGOLite-V21B 10000 v21b_v3 --warm-start-actor "$WARM_START_CKPT"

# FLAT-REGRESSION: the v21b (terrain-trained, blind 49-dim) checkpoint on
# the flat XGOLite-V21A task env — did terrain cost flat tracking?
evalb xgolite_v21b XGOLite-V21A v21b_v3_flat

CKPT=$(latest_ckpt xgolite_v21b)
if [ -n "$CKPT" ]; then
  # Gate feasibility + wobble floor on the flat env (control-rate obs
  # contract matches the v21b training world). asis variant only.
  echo "[chain] $(date +%FT%T) DIAG v21b_v3 ckpt=$CKPT" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u \
    scripts/diag_v19_gate_feasibility.py --task XGOLite-V21A --ckpt "$CKPT" \
    --obs control --variants asis > diag_v21b_v3.log 2>&1
  echo "[chain] $(date +%FT%T) DIAG-END v21b_v3 rc=$?" >> "$LOG"

  # Terrain survival: pinned commands on pinned difficulty rows.
  echo "[chain] $(date +%FT%T) TERRAIN-EVAL v21b_v3 ckpt=$CKPT" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u \
    scripts/eval_terrain_survival.py "$CKPT" XGOLite-V21B \
    > eval_v21b_v3_terrain.log 2>&1
  echo "[chain] $(date +%FT%T) TERRAIN-EVAL-END v21b_v3 rc=$?" >> "$LOG"
fi

date +%FT%T > V21B_V3_DONE
echo "[chain] $(date +%FT%T) CHAIN COMPLETE (V21B_V3_DONE written)" >> "$LOG"
