#!/usr/bin/env bash
# arxiv-synth V1 + V2 triton re-time at campaign-v2.7 on pod d GPU 0 (14 cells: 7 rates, clause, seed 0, bs 1 + 16; CLAUSE-SKIP removes the 10-clause
# width cost): C1's 3 M crossover against pod d's own v2.5 V-AX-SYNTH (same box). Hub campaign-v2.7/arxiv-synth-synth (V1 + V2 only).
#   GPU=0 setsid nohup flock -n /scratch/gpu0.lock bash driver.sh > /scratch/v27/ax-synth-v1v2.driver.log 2>&1 &
export TAG=campaign-v2.7 LEG=ax-synth-v1v2 REPO=/scratch/wt/v27-axsynth PY=/venvs/d-run-v27-ax/bin/python GPU=${GPU:-0}
DS=arxiv-synth SUITE=synth
R=/scratch/campaign-v27/$DS-$SUITE
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
old_oracles $DS
step check check --dataset $DS --dim 128
step oracle oracle --dataset $DS --suite $SUITE --dim 128
stream $DS $SUITE "linr_v1_filter_mask linr_v2|triton"
left=$($PY "$HERE/../../campaign-v2/v-gr-deep/pending.py" "$R" "$cv" "$DS" "$SUITE" linr_v1_filter_mask linr_v2 2>/dev/null | tail -1)
echo "$(date -Is) $left"
grep -q ": 0 to run" <<< "$left" || { note pod-d "$LEG: cells still pending ($left); driver stopped"; exit 6; }
upload campaign-v2.7/$DS-$SUITE
msg="$LEG (14 cells) done on pod d GPU $GPU at $cv; Hub campaign-v2.7/$DS-$SUITE: $UP"
note pod-d "$msg"; note control "$msg"
finish
