#!/usr/bin/env bash
# V-GR-FILTER driver: goodreads `filter` suite (E1c, 3 sweeps, 3 seeds) at campaign-v2, GPU 0, one
# sequential process group. Explicit `bench run` streams instead of `bench campaign` (the
# orchestrator held linr_v3 back while the pilot's V3 seed question was open, then released it), each
# stream one campaign child; the campaign's per-child log, clock blocks, _parity pruning and final
# aggregate are reproduced here. Stops on the first non-zero stream.
# Run 1: setsid nohup bash driver.sh > /scratch/v-gr-filter/driver.log 2>&1 &
# Run 2: setsid nohup bash driver.sh 'linr_v3|triton' > /scratch/v-gr-filter/driver-v3.log 2>&1 &
set -u
REPO=/workspace/retrieve
PY=/venvs/retrieve/bin/python
R=/scratch/campaign-v2/results
LOG=/scratch/v-gr-filter
EXPECT=408b1188d542634b3d18a2f5077bd23537a845fc
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v-gr-filter HF_HOME=/scratch/hf
PIN="taskset -c 0-63,128-191"
mkdir -p "$R/_logs" "$LOG" "$TORCHINDUCTOR_CACHE_DIR"
cd "$REPO/evaluation"

cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }

nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$LOG/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

clocks() { $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; }

t0=$(date +%s)
$PIN $PY -m bench.cli oracle --dataset goodreads --suite filter
rc=$?
echo "$(date -Is) step oracle rc=$rc s=$(( $(date +%s) - t0 ))"
[ $rc -eq 0 ] || exit $rc

# algos|backends per stream, in suite order
[ $# -gt 0 ] || set -- "linr_v1_filter_mask linr_v2|triton" "silvertorch|triton official" \
                       "silvertorch|torch" "postfilter|torch"
for stream in "$@"; do
  algos=${stream%|*}; backends=${stream#*|}
  $PY - "$R/_parity" $algos <<'EOF'
import shutil, sys
from pathlib import Path
from bench import run
keep = {run.parity_group("goodreads", 128, a) for a in sys.argv[2:]}
for p in Path(sys.argv[1]).glob("*"):
    if p.name not in keep:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
EOF
  args=(); for a in $algos; do args+=(--algo "$a"); done; for b in $backends; do args+=(--backend "$b"); done
  name="filter_goodreads-d128_${algos// /+}_${backends// /+}.log"
  cmd=($PIN $PY -m bench.cli run --dataset goodreads --dim 128 --suite filter "${args[@]}" --out "$R" --resume --interleave)
  t0=$(date +%s)
  { echo "=== ${cmd[*]}"; echo "=== clocks at start"; clocks; } >> "$R/_logs/$name"
  "${cmd[@]}" >> "$R/_logs/$name" 2>&1
  rc=$?
  { echo "=== clocks at end"; clocks; } >> "$R/_logs/$name"
  echo "$(date -Is) stream ${algos// /+} ${backends// /+} rc=$rc s=$(( $(date +%s) - t0 )) log=$name"
  [ $rc -eq 0 ] || exit $rc
done
rm -rf "$R/_parity"
$PY -c 'import sys; from pathlib import Path; from bench import records; print("=== aggregated", records.aggregate(Path(sys.argv[1])))' "$R"
echo "$(date -Is) driver done"
