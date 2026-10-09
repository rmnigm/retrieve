#!/usr/bin/env bash
# V-LAION30 SilverTorch re-time at campaign-v2.9 (ST-WIDE: per-row probe table at B * n_probe >= 512), pod d GPU 0, one process per
# (suite, sweep), tags4 first (the IVF-vs-exact-at-0.95 cell, n_probe 4096 bs 16). Part 1: laion30m + laion30m-bs1 SilverTorch
# (4 builds) -> campaign-v2.9/laion30m-filter. Part 2, only when part 1 took <= 1.5 h (controller): laion30m-synth SilverTorch
# (4 builds) -> campaign-v2.9/laion30m-synth-synth. Oracle blobs are the v2.5-built ones (harness-only, data-determined; d-run's
# running leg reads them too, so they are not moved). Rerun after a crash: --resume skips finished cells.
. "$(dirname "$(readlink -f "$0")")/common.sh"
TABLE=$W/docs/artifacts/campaign-v2.5/laion30m/cal_table.py
R=/scratch/campaign-v29/laion30m-filter
mkdir -p "$R/logs"
t0=$(date +%s)
for sw in tags4 c0_domain; do
  for s in laion30m laion30m-bs1; do
    run "st-$s-$sw" run --dataset laion30m --dim 256 --suite $s --sweep $sw --algo silvertorch --out "$R" --resume
  done
done
cp -a "$L/logs/clocks-gpu$GPU.csv" "$L"/logs/st-*.log "$R/logs/"
$PY "$TABLE" "$R" | tee "$R/table.md"
run upload-filter upload --results "$R" --path-in-repo campaign-v2.9/laion30m-filter --verify
grep -E "MANIFEST|round trip" "$L/logs/upload-filter.log"
part1=$(( $(date +%s) - t0 ))
echo "$(date -Is) part 1 done in ${part1}s"
[ $part1 -le 5400 ] || { echo "$(date -Is) part 1 over 1.5 h: synth part skipped (controller)"; exit 0; }
R=/scratch/campaign-v29/laion30m-synth-synth
mkdir -p "$R/logs"
for sw in p01 p02 p05 p1; do
  run "synth-st-$sw" run --dataset laion30m-synth --dim 256 --suite laion30m-synth --sweep $sw --algo silvertorch --out "$R" --resume
done
cp -a "$L/logs/clocks-gpu$GPU.csv" "$L"/logs/synth-st-*.log "$R/logs/"
$PY "$TABLE" "$R" | tee "$R/table.md"
run upload-synth upload --results "$R" --path-in-repo campaign-v2.9/laion30m-synth-synth --verify
grep -E "MANIFEST|round trip" "$L/logs/upload-synth.log"
echo "$(date -Is) st rc=0"
