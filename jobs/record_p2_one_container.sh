#!/bin/bash
set +e
REPO=/workspace/booster_ws
PY=/isaac-sim/python.sh
cd "$REPO" || exit 1
OUT="$REPO/isaac_tasks/k1_velocity/videos"
mkdir -p "$OUT" "$REPO/models"

"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/booster_train_ref/source/booster_train >/tmp/rec_install.log 2>&1
"$PY" -m pip install --no-deps --no-build-isolation -e isaac_tasks/k1_velocity >>/tmp/rec_install.log 2>&1
"$PY" -m pip install --no-deps --no-build-isolation -e src/k1_description/assets >>/tmp/rec_install.log 2>&1

echo "=== P2_REC_ONE name=$P2_REC_NAME task=$P2_REC_TASK"
timeout 1800 "$PY" -u isaac_tasks/k1_velocity/scripts/play_record.py \
  --task "$P2_REC_TASK" --checkpoint "$P2_REC_CKPT" --num_envs 4 --steps 900 --headless \
  --cmd 0.4 0 0 --panel_video \
  --video_out "$OUT/${P2_REC_NAME}.mp4" \
  --trace_out "$OUT/${P2_REC_NAME}_trace.npz" \
  --export "$REPO/models/${P2_REC_NAME}.pt" \
  --label "P2 ${P2_REC_NAME} · AGILE-aligned" 2>&1 | tail -25
rc=${PIPESTATUS[0]}
echo "P2_REC_RC=$rc"
[ "$rc" -eq 0 ] || exit 10

ffmpeg -y -ss 8 -i "$OUT/${P2_REC_NAME}.mp4" -frames:v 1 \
  "$OUT/${P2_REC_NAME}_poster.png" >/tmp/poster.log 2>&1
sz=$(stat -c%s "$OUT/${P2_REC_NAME}.mp4" 2>/dev/null || echo 0)
[ "$sz" -gt 200000 ] || { echo "P2_REC_ARTIFACT_FAIL size=${sz}B"; exit 12; }

# Same gates as the full recorder: finite trace and real displacement.
"$PY" - <<'PY'
import numpy as np, os
from pathlib import Path
name = os.environ["P2_REC_NAME"]
d = np.load(Path("isaac_tasks/k1_velocity/videos") / f"{name}_trace.npz")
assert all(np.isfinite(d[k]).all() for k in d.files), f"non-finite trace: {name}"
p = d["root_pos"]
disp = np.linalg.norm(p[-1, :, :2] - p[0, :, :2], axis=1)
z = p[:, :, 2]
print(f"P2_REC_MOVE {name} env_displacement={np.round(disp,3).tolist()}")
print(f"P2_REC_HEIGHT {name} trunk_z mean={z.mean():.3f} min={z.min():.3f} "
      f"(fell below 0.35 m for {100*(z<0.35).mean():.1f}% of samples)")
assert float(disp.max()) > 0.5, f"no gait displacement: {name} {disp.tolist()}"
print("P2_REC_MOVE_CHECK=OK")
PY
[ "$?" -eq 0 ] || { echo P2_REC_MOVE_CHECK=FAIL; exit 13; }
echo "P2_REC_ONE_DONE"
exit 0
