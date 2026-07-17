#!/usr/bin/env bash
# v21a new-stack requalification chain (2026-07-17): reproduce the v21a
# gait recipe — 2500-iter fresh run + 2500-iter extension (= the deployed
# v21a_ext hardware policy) — on the upgraded stack (mjlab 1.5.1,
# mujoco/mjx/mujoco-warp 3.10, warp 1.15, rsl-rl 5.4) to verify quality
# parity with the 2026-07-15 campaign. Compare eval_v21a_ns1.log /
# eval_v21a_ns1ext.log against the committed eval_v21a_v1.log /
# eval_v21a_v1ext.log (same buckets, EVAL_LATERAL_BUCKETS=1).
#
# Layout note: the venv is the parent Quadruped-robot repo's single .venv
# (../.venv), not a local one — see scripts/setup-venv.sh in the parent.
#
# Designed to run DETACHED on the GPU box:
#
#   ssh dvv@192.168.1.49 'setsid nohup ~/dev/bots/Quadruped-robot/luwu_mjlab/scripts/run_v21a_ns.sh \
#       >/dev/null 2>&1 </dev/null & disown'
#
# Progress in chain_v21a_ns_status.log; V21A_NS_DONE marks completion.
set -u
cd "$(dirname "$0")/.."

PY=../.venv/bin/python
LOG=chain_v21a_ns_status.log
echo "[chain] $(date +%FT%T) v21a_ns chain pid $$ starting" >> "$LOG"

# Gate: the deterministic preset checker must pass on this box before
# burning GPU-hours.
PYTHONPATH=. MUJOCO_GL=egl $PY scripts/check_v21a_preset.py > check_v21a_ns.log 2>&1
if ! grep -q "OVERALL: PASS" check_v21a_ns.log; then
  echo "[chain] $(date +%FT%T) ABORT: check_v21a_preset FAILED (see check_v21a_ns.log)" >> "$LOG"
  exit 1
fi
echo "[chain] $(date +%FT%T) preset checker PASS" >> "$LOG"

run() {
  local task=$1 iters=$2 name=$3
  shift 3
  echo "[chain] $(date +%FT%T) START $task ($iters iters, 4096 envs) -> train_${name}.log" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl $PY -u scripts/train.py "$task" \
    --env.scene.num-envs 4096 --agent.max-iterations "$iters" \
    --agent.run-name "$name" "$@" > "train_${name}.log" 2>&1
  local rc=$?
  echo "[chain] $(date +%FT%T) END $task rc=$rc" >> "$LOG"
  return $rc
}

latest_ckpt() {
  local exp=$1 pat=$2 rundir
  rundir=$(ls -d logs/rsl_rl/"$exp"/*"$pat"/ 2>/dev/null | sort | tail -1)
  ls "${rundir}"model_*.pt 2>/dev/null | sort -V | tail -1
}

evalb() {
  local task=$1 name=$2 ckpt=$3
  if [ -z "$ckpt" ]; then
    echo "[chain] $(date +%FT%T) EVAL-SKIP $name (no checkpoint found)" >> "$LOG"
    return
  fi
  echo "[chain] $(date +%FT%T) EVAL $name ckpt=$ckpt" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl EVAL_LATERAL_BUCKETS=1 \
    $PY -u scripts/eval_policy_buckets.py "$ckpt" "$task" > "eval_${name}.log" 2>&1
  echo "[chain] $(date +%FT%T) EVAL-END $name rc=$?" >> "$LOG"
}

diag() {
  local name=$1 ckpt=$2
  [ -z "$ckpt" ] && return
  echo "[chain] $(date +%FT%T) DIAG $name ckpt=$ckpt" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl $PY -u \
    scripts/diag_v19_gate_feasibility.py --task XGOLite-V21A --ckpt "$ckpt" \
    --obs control --variants asis > "diag_${name}.log" 2>&1
  echo "[chain] $(date +%FT%T) DIAG-END $name rc=$?" >> "$LOG"
}

# Stage 1: fresh 2500-iter v21a run (mirrors the 2026-07-15 v21a_v1).
run XGOLite-V21A 2500 v21a_ns1 || exit 1
CKPT1=$(latest_ckpt xgolite_v21a _v21a_ns1)
evalb XGOLite-V21A v21a_ns1 "$CKPT1"
diag v21a_ns1 "$CKPT1"

# Stage 2: +2500-iter extension to 5000 total (mirrors v21a_v1ext, the
# checkpoint that became the deployed v21a_ext policy). rsl-rl semantics:
# max-iterations on resume = ADDITIONAL iterations, not total (the first
# 2026-07-17 chain run passed 5000 here and produced a 7500-total policy,
# model_7498 — kept as run v21a_ns1ext, a longer-training data point; the
# apples-to-apples 5000-total rerun is v21a_ns1ext2).
run XGOLite-V21A 2500 v21a_ns1ext2 \
  --agent.resume True --agent.load-run '.*_v21a_ns1$' || exit 1
CKPT2=$(latest_ckpt xgolite_v21a _v21a_ns1ext2)
evalb XGOLite-V21A v21a_ns1ext2 "$CKPT2"
diag v21a_ns1ext2 "$CKPT2"

date +%FT%T > V21A_NS_DONE
echo "[chain] $(date +%FT%T) CHAIN COMPLETE (V21A_NS_DONE written)" >> "$LOG"
