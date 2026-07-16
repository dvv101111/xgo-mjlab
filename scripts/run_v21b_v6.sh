#!/usr/bin/env bash
# v21b v6 campaign (2026-07-16): the standing-basin fix. v2/v3/v4 all
# converged to (or eroded toward) NOT WALKING because standing is the
# reward-dominant local optimum on hazard terrain: relative-sigma tracking
# pays ~0.42 raw at zero velocity over the seed band, stand_still only
# fires at |cmd|<=0.1, and compliance/slip/ice DR ran at full severity
# from iteration 0 on all rows (the warm flat gait crashed instantly,
# eplen 19 at iter 0 -> gradient learned walking=crashing). v4's warm
# walker abandoned locomotion by iter ~250 and stood for 9750 iters
# (error_vel_xy pinned ~0.21, final vx_ach <=0.003 everywhere).
#   v5 changes (v21b.py points 9/10):
#   - velocity_progress_lin/ang rewards: directional achieved/commanded
#     ratio (the unlock-gate math as a per-step reward), weight 6.0 =>
#     +0.06/step at ratio 0.5 (2x v4's measured standing income), zero
#     when standing. Standing is now strictly non-competitive.
#   - hazard-DR severity scaled per env by terrain level: row 0 = benign
#     physics (timeconst <=0.05 s, friction >=0.4, no slips), full
#     severity only at the top row. Walking stays viable at curriculum
#     entry; hazards arrive with competence.
#   Kept from v3/v4: entropy_coef 0.001, warm-start, ratio unlock gate.
#   v6 change (v5 killed at iter 1108): the progress rewards WORKED (the
#   policy genuinely walks, progress_lin 1.26->2.12) but the TERRAIN
#   curriculum collapsed to row 0 (terrain_levels 0.97->0.02 by iter 500,
#   pinned): terrain promotion gated on per-episode relative-sigma
#   tracking fractions >= 0.70/0.55 — a metric that scores standing at
#   slow commands (~0.49) HIGHER than honest ratio-0.5 walking (~0.26).
#   v4's stander passed those gates in the tail (hollow promotions to
#   0.33); v5's walker never passed. v6 gates terrain promotion/demotion
#   on the honest directional achieved/commanded velocity ratio
#   (promote >= 0.5, demote < 0.25, per commanded axis — the same math
#   as the grid unlock gate and the progress rewards). v21b.py point 11.
# Designed to run DETACHED on the GPU box:
#
#   ssh dvv@192.168.1.49 'setsid nohup ~/dev/bots/luwu_mjlab/scripts/run_v21b_v6.sh \
#       >/dev/null 2>&1 </dev/null & disown'
#
# Training (10000 iters) is followed by:
#   1. FLAT-REGRESSION eval: the v21b checkpoint bucket-eval'd on the
#      XGOLite-V21A task env (actor contracts are identical 49-dim; the
#      loader is actor-only) with the wide-lateral buckets.
#   2. Gate-feasibility diagnostic on the flat V21A env.
#   3. Terrain survival eval: pinned commands x pinned difficulty rows
#      (0/3/6/9), survival + tracking per row -> eval_v21b_v6_terrain.log.
# V21B_V6_DONE marks completion. Progress in chain_v21b_v6_status.log.
set -u
cd "$(dirname "$0")/.."

LOG=chain_v21b_v6_status.log
WARM_START_CKPT=logs/rsl_rl/xgolite_v21a/2026-07-15_22-16-35_v21a_v1ext/model_4998.pt
echo "[chain] $(date +%FT%T) v21b_v6 chain pid $$ starting" >> "$LOG"

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

run XGOLite-V21B 10000 v21b_v6 --warm-start-actor "$WARM_START_CKPT"

# FLAT-REGRESSION: the v21b (terrain-trained, blind 49-dim) checkpoint on
# the flat XGOLite-V21A task env — did terrain cost flat tracking?
evalb xgolite_v21b XGOLite-V21A v21b_v6_flat

CKPT=$(latest_ckpt xgolite_v21b)
if [ -n "$CKPT" ]; then
  # Gate feasibility + wobble floor on the flat env (control-rate obs
  # contract matches the v21b training world). asis variant only.
  echo "[chain] $(date +%FT%T) DIAG v21b_v6 ckpt=$CKPT" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u \
    scripts/diag_v19_gate_feasibility.py --task XGOLite-V21A --ckpt "$CKPT" \
    --obs control --variants asis > diag_v21b_v6.log 2>&1
  echo "[chain] $(date +%FT%T) DIAG-END v21b_v6 rc=$?" >> "$LOG"

  # Terrain survival: pinned commands on pinned difficulty rows.
  echo "[chain] $(date +%FT%T) TERRAIN-EVAL v21b_v6 ckpt=$CKPT" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u \
    scripts/eval_terrain_survival.py "$CKPT" XGOLite-V21B \
    > eval_v21b_v6_terrain.log 2>&1
  echo "[chain] $(date +%FT%T) TERRAIN-EVAL-END v21b_v6 rc=$?" >> "$LOG"
fi

date +%FT%T > V21B_V6_DONE
echo "[chain] $(date +%FT%T) CHAIN COMPLETE (V21B_V6_DONE written)" >> "$LOG"
