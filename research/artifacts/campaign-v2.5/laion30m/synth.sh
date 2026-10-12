#!/usr/bin/env bash
# V-LAION30 synth leg at campaign-v2.5: suite `laion30m-synth` (p {0.1, 0.2, 0.5, 1.0}, clause, k 100, seed 0, bs 1 + 16):
# V1 / V2 interleaved, SilverTorch n_lists 16384 x n_probe {24, 256, 1024}; one tree $R, uploaded as
# campaign-v2.5/laion30m-synth-synth. One process per (sweep, group): at 30 M a V1 + V2 child holds 68 GB after one
# sweep and the next sweep runs out of memory (grid tags4). Rerun after a crash: --resume skips finished cells.
. "$(dirname "$(readlink -f "$0")")/common.sh"
R=/scratch/campaign-v25/laion30m-synth-synth
mkdir -p "$R/logs"
run synth-check check --dataset laion30m-synth --dim 256
run synth-oracle oracle --dataset laion30m-synth --suite laion30m-synth --dim 256
for sw in p01 p02 p05 p1; do
  run "synth-$sw-v1v2" run --dataset laion30m-synth --dim 256 --suite laion30m-synth --sweep $sw \
    --algo linr_v1_filter_mask --algo linr_v2 --interleave --out "$R" --resume
  run "synth-$sw-st" run --dataset laion30m-synth --dim 256 --suite laion30m-synth --sweep $sw \
    --algo silvertorch --out "$R" --resume
done
cp -a "$L/logs/clocks-gpu$GPU.csv" "$R/logs/" && cp -a "$L"/logs/synth-*.log "$R/logs/"
$PY "$HERE/cal_table.py" "$R" | tee "$R/table.md"
run synth-upload upload --results "$R" --path-in-repo campaign-v2.5/laion30m-synth-synth --verify
grep -E "MANIFEST|round trip" "$L/logs/synth-upload.log"
echo "$(date -Is) synth rc=0"
