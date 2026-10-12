#!/usr/bin/env bash
# V-LAION30 v2.9: the 30 M probe curve on the real filters (F3, controller): SilverTorch n_probe {64, 256} added to laion30m (bs 16)
# and laion30m-bs1 (bs 1), one process per (suite, sweep), into the campaign-v2.9/laion30m-filter tree (--resume skips the 24 / 1024 /
# 4096 cells already there; each job rebuilds its index), then the tree re-uploaded. Pod d GPU 0.
. "$(dirname "$(readlink -f "$0")")/common.sh"
R=/scratch/campaign-v29/laion30m-filter
for sw in tags4 c0_domain; do
  for s in laion30m laion30m-bs1; do
    run "probe-$s-$sw" run --dataset laion30m --dim 256 --suite $s --sweep $sw --algo silvertorch --out "$R" --resume
  done
done
cp -a "$L/logs/clocks-gpu$GPU.csv" "$L"/logs/probe-*.log "$R/logs/"
$PY "$W/docs/artifacts/campaign-v2.5/laion30m/cal_table.py" "$R" | tee "$R/table.md"
run upload-probe upload --results "$R" --path-in-repo campaign-v2.9/laion30m-filter --verify
grep -E "MANIFEST|round trip" "$L/logs/upload-probe.log"
echo "$(date -Is) probe rc=0"
