#!/usr/bin/env bash
# V-LAION30 grid at campaign-v2.5: suites `laion30m` (V1 / V2 interleaved, SilverTorch; bs 16) and `laion30m-bs1`
# (SilverTorch; bs 1), n_probe {24, n95, 4096} from cal.sh, one tree $R, uploaded as campaign-v2.5/laion30m-filter.
# Rerun after a crash: --resume skips finished cells.
. "$(dirname "$(readlink -f "$0")")/common.sh"
R=/scratch/campaign-v25/laion30m-filter
mkdir -p "$R/logs"
for s in laion30m laion30m-bs1; do
  run "grid-$s" campaign --dataset laion30m --dim 256 --suite $s --interleave --out "$R" --resume
done
cp -a "$L/logs/clocks.csv" "$R/logs/" && cp -a "$L"/logs/grid-*.log "$R/logs/"
$PY -c 'import sys; from pathlib import Path; from bench import records; print("aggregated", records.aggregate(Path(sys.argv[1])))' "$R"
$PY "$HERE/cal_table.py" "$R" | tee "$R/table.md"
run upload upload --results "$R" --path-in-repo campaign-v2.5/laion30m-filter --verify
grep -E "MANIFEST|round trip" "$L/logs/upload.log"
echo "$(date -Is) grid rc=0"
