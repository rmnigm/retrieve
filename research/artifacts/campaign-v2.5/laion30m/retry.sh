#!/usr/bin/env bash
# V-LAION30 grid, the rest after grid.sh stopped (its V1 + V2 child ran out of memory on its second sweep, 68 GB
# reserved after c0_domain, so `bench campaign` exited 1 before laion30m-bs1): V1 / V2 tags4 in a fresh process, the
# laion30m-bs1 suite, then the tree uploaded to campaign-v2.5/laion30m-filter.
. "$(dirname "$(readlink -f "$0")")/common.sh"
R=/scratch/campaign-v25/laion30m-filter
C=$L/grid-config   # the grid's config (143bde9) so config_sha stays 395ff0fa; staging since added laion30m-synth
mkdir -p "$C"
for f in suites laion30m; do git -C "$W" show 143bde9:evaluation/config/$f.yaml > "$C/$f.yaml"; done
run retry-tags4 run --config-dir "$C" --dataset laion30m --dim 256 --suite laion30m --sweep tags4 \
  --algo linr_v1_filter_mask --algo linr_v2 --interleave --out "$R" --resume
run grid-laion30m-bs1 campaign --config-dir "$C" --dataset laion30m --dim 256 --suite laion30m-bs1 --interleave --out "$R" --resume
cp -a "$L/logs/clocks-gpu$GPU.csv" "$L"/logs/grid-*.log "$L"/logs/retry-*.log "$R/logs/"
$PY "$HERE/cal_table.py" "$R" | tee "$R/table.md"
run upload-retry upload --results "$R" --path-in-repo campaign-v2.5/laion30m-filter --verify
grep -E "MANIFEST|round trip" "$L/logs/upload-retry.log"
echo "$(date -Is) retry rc=0"
