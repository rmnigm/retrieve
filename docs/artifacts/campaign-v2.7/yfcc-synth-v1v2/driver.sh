#!/usr/bin/env bash
# yfcc10m-synth V1 + V2 triton re-time at campaign-v2.7 on pod d GPU 0 (10 cells: p 0.01 / 0.1 / 0.2 / 0.5 / 1, clause, seed 0, k 100, bs 1 + 16;
# exact_gate 0.9716 at k 100): C1's 10 M crossover without the 10-clause width cost, against pod d's own v2.5 YFCC synth (same box).
#   GPU=0 setsid nohup flock -n /scratch/gpu0.lock bash driver.sh > /scratch/v27/yfcc-synth-v1v2.driver.log 2>&1 &
export TAG=campaign-v2.7 LEG=yfcc-synth-v1v2 REPO=/scratch/wt/v27-axsynth PY=/venvs/d-run-v27-ax/bin/python GPU=${GPU:-0}
DS=yfcc10m-synth SUITE=synth
R=/scratch/campaign-v27/$DS-$SUITE
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
old_oracles $DS
step check check --dataset $DS --dim 192
step oracle oracle --dataset $DS --suite $SUITE --dim 192
stream $DS $SUITE "linr_v1_filter_mask linr_v2|triton"
left=$($PY "$HERE/../../campaign-v2/v-gr-deep/pending.py" "$R" "$cv" "$DS" "$SUITE" linr_v1_filter_mask linr_v2 2>/dev/null | tail -1)
echo "$(date -Is) $left"
grep -q ": 0 to run" <<< "$left" || { note pod-d "$LEG: cells still pending ($left); driver stopped"; exit 6; }
upload campaign-v2.7/$DS-$SUITE
msg="$LEG (10 cells) done on pod d GPU $GPU at $cv; Hub campaign-v2.7/$DS-$SUITE: $UP"
note pod-d "$msg"; note control "$msg"
finish
