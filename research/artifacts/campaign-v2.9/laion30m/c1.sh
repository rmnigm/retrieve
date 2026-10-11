#!/usr/bin/env bash
# Night queue 1: C1 + F1 at 30 M on one tag: laion30m-synth V1 + V2 triton at the nine C1 rates (p 0.001 ... 1 without 0.3), bs 1 + 16,
# eager + graph, seed 0, k 100, clause, interleaved V1 / V2 in one process per sweep, at campaign-v2.9 (this tree, library e8958bd2;
# staging has moved past it). Pod d GPU 0 -> campaign-v2.9/laion30m-synth-v1v2. Rerun after a crash: --resume skips finished cells.
. "$(dirname "$(readlink -f "$0")")/common.sh"
R=/scratch/campaign-v29/laion30m-synth-v1v2
mkdir -p "$R/logs"
for sw in p0001 p0003 p001 p003 p005 p01 p02 p05 p1; do
  run "c1-$sw" run --dataset laion30m-synth --dim 256 --suite laion30m-synth --sweep $sw \
    --algo linr_v1_filter_mask --algo linr_v2 --interleave --out "$R" --resume
done
cp -a "$L/logs/clocks-gpu$GPU.csv" "$L"/logs/c1-*.log "$R/logs/"
$PY "$W/docs/artifacts/campaign-v2.5/laion30m/cal_table.py" "$R" | tee "$R/table.md"
run upload-c1 upload --results "$R" --path-in-repo campaign-v2.9/laion30m-synth-v1v2 --verify
grep -E "MANIFEST|round trip" "$L/logs/upload-c1.log"
echo "$(date -Is) c1 rc=0"
