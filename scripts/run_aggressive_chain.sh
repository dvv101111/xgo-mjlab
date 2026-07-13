#!/usr/bin/env bash
# Sequential training chain for the aggressive-locomotion presets
# (2026-07-11 campaign). Designed to run DETACHED on the GPU box:
#
#   ssh dvv@192.168.1.49 'setsid nohup ~/dev/bots/luwu_mjlab/scripts/run_aggressive_chain.sh \
#       >/dev/null 2>&1 </dev/null & disown'
#
# One training at a time (never parallel); one log per preset; chain
# progress in chain_status.log. A failed run is recorded and the chain
# moves on to the next preset.
set -u
cd "$(dirname "$0")/.."

LOG=chain_status.log
echo "[chain] $(date +%FT%T) chain pid $$ starting" >> "$LOG"

run() {
  local task=$1 iters=$2 name=$3
  echo "[chain] $(date +%FT%T) START $task ($iters iters, 4096 envs) -> train_${name}.log" >> "$LOG"
  PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -u scripts/train.py "$task" \
    --env.scene.num-envs 4096 --agent.max-iterations "$iters" \
    --agent.run-name "$name" > "train_${name}.log" 2>&1
  local rc=$?
  echo "[chain] $(date +%FT%T) END $task rc=$rc" >> "$LOG"
}

run XGOLite-Sprint    1500 sprint_v1
run XGOLite-FastClock 1500 fastclock_v1
run XGOLite-Agile     1500 agile_v1
run XGOLite-FreeGait  3000 freegait_v1

echo "[chain] $(date +%FT%T) CHAIN COMPLETE" >> "$LOG"
