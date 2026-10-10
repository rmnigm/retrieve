#!/usr/bin/env bash
# ST-TOPK ledger re-time at campaign-v2.11 (controller): laion30m-filter's SilverTorch triton cells, laion30m (bs 16) + laion30m-bs1 (bs 1),
# n_lists 16384 n_probe {24, 64, 256, 1024, 4096}, tags4 + c0_domain, eager + graph, seed 0, k 100; one process per (suite, sweep): 4 builds,
# 20 cells. Pod d GPU 0 -> campaign-v2.11/laion30m-filter. Rerun after a crash: --resume skips finished cells.
. "$(dirname "$(readlink -f "$0")")/common.sh"
R=/scratch/campaign-v211/laion30m-filter
mkdir -p "$R/logs"
for sw in tags4 c0_domain; do
  for s in laion30m laion30m-bs1; do
    run "fst-$s-$sw" run --dataset laion30m --dim 256 --suite $s --sweep $sw --algo silvertorch --out "$R" --resume
  done
done
cp -a "$L/logs/clocks-gpu$GPU.csv" "$L"/logs/fst-*.log "$R/logs/"
run upload-fst upload --results "$R" --path-in-repo campaign-v2.11/laion30m-filter --verify
grep -E "MANIFEST|round trip" "$L/logs/upload-fst.log"
echo "$(date -Is) filter-st rc=0"
