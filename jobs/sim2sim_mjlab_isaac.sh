#!/bin/bash
# Inside the Isaac Lab 3.0 image: run an mjlab K1 checkpoint in Isaac Lab and record it.
#   usage: sim2sim_mjlab_isaac.sh <ckpt> <out.mp4> [vx vy wz]   (K1_PHYSICS=newton|physx)
# Repo mounted at /ws. grep SIM2SIM / VIDEO in the output.
set +e
cd /ws || exit 1
PY=/isaac-sim/python.sh
$PY -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train \
  -e isaac_tasks/k1_velocity -e src/k1_description/assets >/dev/null 2>&1
CKPT="${1:?ckpt}"; OUT="${2:?out.mp4}"; shift 2
CMD="${*:-0.4 0 0}"
$PY isaac_tasks/k1_velocity/scripts/play_mjlab_policy.py --checkpoint "$CKPT" --cmd $CMD \
  --num_envs 16 --steps 750 --video_out "$OUT" --headless > "${OUT%.mp4}.log" 2>&1
grep -a "SIM2SIM\|VIDEO\|OBS\|Error\|Traceback\|error:" "${OUT%.mp4}.log" | grep -v Warning
chmod a+rw "${OUT%.mp4}.log" 2>/dev/null
chmod a+rw "$OUT" 2>/dev/null
