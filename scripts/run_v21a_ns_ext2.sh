#!/usr/bin/env bash
# One-off corrective stage for the 2026-07-17 v21a_ns chain: the original
# stage 2 passed --agent.max-iterations 5000 not knowing rsl-rl treats it
# as ADDITIONAL iterations on resume, producing a 7500-total policy
# (v21a_ns1ext, kept as a longer-training data point). This reruns the
# extension with 2500 additional iterations from the same v21a_ns1
# checkpoint -> 5000 total, apples-to-apples with the 2026-07-15
# v21a_v1ext baseline (the deployed v21a_ext policy).
set -u
cd "$(dirname "$0")/.."

PY=../.venv/bin/python
LOG=chain_v21a_ns_status.log
echo "[chain] $(date +%FT%T) v21a_ns_ext2 corrective stage pid $$ starting" >> "$LOG"

echo "[chain] $(date +%FT%T) START XGOLite-V21A (2500 iters, 4096 envs) -> train_v21a_ns1ext2.log" >> "$LOG"
PYTHONPATH=. MUJOCO_GL=egl $PY -u scripts/train.py XGOLite-V21A \
  --env.scene.num-envs 4096 --agent.max-iterations 2500 \
  --agent.run-name v21a_ns1ext2 \
  --agent.resume True --agent.load-run '.*_v21a_ns1$' \
  > train_v21a_ns1ext2.log 2>&1
rc=$?
echo "[chain] $(date +%FT%T) END XGOLite-V21A rc=$rc" >> "$LOG"
[ $rc -ne 0 ] && exit 1

rundir=$(ls -d logs/rsl_rl/xgolite_v21a/*_v21a_ns1ext2/ 2>/dev/null | sort | tail -1)
CKPT=$(ls "${rundir}"model_*.pt 2>/dev/null | sort -V | tail -1)

echo "[chain] $(date +%FT%T) EVAL v21a_ns1ext2 ckpt=$CKPT" >> "$LOG"
PYTHONPATH=. MUJOCO_GL=egl EVAL_LATERAL_BUCKETS=1 \
  $PY -u scripts/eval_policy_buckets.py "$CKPT" XGOLite-V21A > eval_v21a_ns1ext2.log 2>&1
echo "[chain] $(date +%FT%T) EVAL-END v21a_ns1ext2 rc=$?" >> "$LOG"

echo "[chain] $(date +%FT%T) DIAG v21a_ns1ext2 ckpt=$CKPT" >> "$LOG"
PYTHONPATH=. MUJOCO_GL=egl $PY -u \
  scripts/diag_v19_gate_feasibility.py --task XGOLite-V21A --ckpt "$CKPT" \
  --obs control --variants asis > diag_v21a_ns1ext2.log 2>&1
echo "[chain] $(date +%FT%T) DIAG-END v21a_ns1ext2 rc=$?" >> "$LOG"

date +%FT%T > V21A_NS_EXT2_DONE
echo "[chain] $(date +%FT%T) EXT2 COMPLETE (V21A_NS_EXT2_DONE written)" >> "$LOG"
