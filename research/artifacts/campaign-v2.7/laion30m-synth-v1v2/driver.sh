#!/usr/bin/env bash
# C1 at 30 M clean: laion30m-synth V1 + V2 triton at campaign-v2.7 (its 8 cells ran pre-CLAUSE-SKIP at v2.5) on pod d GPU 0, from a worktree at
# the campaign-v2.7 tag (/scratch/wt/v27-tag, venv /venvs/d-run-v27t), one process per sweep until H-ARMFREE; laion30m's oracle blobs are not moved
# (the GPU-1 official leg reads the same gt_d256; they key on the attrs digest). Hub campaign-v2.7/laion30m-synth-synth (V1 + V2).
#   GPU=0 setsid nohup flock -n /scratch/gpu0.lock bash driver.sh > /scratch/v27/laion30m-synth-v1v2.driver.log 2>&1 &
export TAG=campaign-v2.7 LEG=laion30m-synth-v1v2 REPO=/scratch/wt/v27-tag PY=/venvs/d-run-v27t/bin/python GPU=${GPU:-0}
DS=laion30m-synth SUITE=laion30m-synth
R=/scratch/campaign-v27/$DS-$SUITE
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
step check check --dataset $DS --dim 256
for sw in p01 p02 p05 p1; do NARROW="--sweep $sw" stream $DS $SUITE "linr_v1_filter_mask linr_v2|triton"; done
left=$($PY "$HERE/../../campaign-v2/v-gr-deep/pending.py" "$R" "$cv" "$DS" "$SUITE" linr_v1_filter_mask linr_v2 2>/dev/null | tail -1)
echo "$(date -Is) $left"
grep -q ": 0 to run" <<< "$left" || { note pod-d "$LEG: cells still pending ($left); driver stopped"; exit 6; }
upload campaign-v2.7/$DS-$SUITE
msg="$LEG (8 cells) done on pod d GPU $GPU at $cv; Hub campaign-v2.7/$DS-$SUITE: $UP"
note pod-d "$msg"; note control "$msg"
finish
