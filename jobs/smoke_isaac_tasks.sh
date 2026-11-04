#!/bin/bash
# Inside the Isaac Lab 3.0 image (repo at /ws): build + zero-action step each K1 task, one process per task.
#   usage: smoke_isaac_tasks.sh <out_dir> <task> [task ...]   (K1_PHYSICS, K1_ASSET honoured)
# Summary lines: grep SMOKE <out_dir>/*.log
set +e
cd /ws || exit 1
PY=/isaac-sim/python.sh
$PY -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train \
  -e isaac_tasks/k1_velocity -e src/k1_description/assets >/dev/null 2>&1
OUT="${1:?out_dir}"; shift
mkdir -p "$OUT"
for t in "$@"; do
  timeout 900 $PY isaac_tasks/k1_velocity/scripts/smoke_task.py --task "$t" --headless --enable_cameras > "$OUT/$t.log" 2>&1
  grep -a "^SMOKE" "$OUT/$t.log" || echo "SMOKE $t FAIL (no result, see log)"
done
chmod -R a+rw "$OUT" 2>/dev/null
