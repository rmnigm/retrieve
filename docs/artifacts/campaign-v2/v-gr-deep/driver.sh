#!/usr/bin/env bash
# V-GR-DEEP driver: one (dataset, suite) at campaign-v2 on GPU 0, one sequential process group, from
# the V-GR-FILTER template. Explicit `bench run --resume --interleave` streams, each one campaign
# child (per-stream log under _logs/ with start / end clock blocks, then the aggregate). `deep` has
# no interleave groups, so --interleave leaves its cells one unit each. _parity is not pruned: its
# spills are keyed by suite and params, and the tree holds other legs' spills. Stops on the first
# non-zero stream.
# Step 1: setsid nohup bash driver.sh goodreads deep > /scratch/v-gr-deep/driver-deep.log 2>&1 &
# Step 3: setsid nohup bash driver.sh goodreads filter 'silvertorch|triton official' > .../driver-filter-n95.log 2>&1 &
#         setsid nohup bash driver.sh goodreads-synth synth 'silvertorch|triton official' > .../driver-synth-n95.log 2>&1 &
set -u
DS=$1 SUITE=$2; shift 2
REPO=/workspace/retrieve
PY=/venvs/retrieve/bin/python
R=/scratch/campaign-v2/results
LOG=/scratch/v-gr-deep
HERE=$(cd "$(dirname "$0")" && pwd)
EXPECT=408b1188d542634b3d18a2f5077bd23537a845fc
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v-gr-deep HF_HOME=/scratch/hf
PIN="taskset -c 0-63,128-191"
mkdir -p "$R/_logs" "$LOG" "$TORCHINDUCTOR_CACHE_DIR"
cd "$REPO/evaluation"

cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }

nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$LOG/clocks-$SUITE.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

clocks() { $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; }

t0=$(date +%s)
$PIN $PY -m bench.cli oracle --dataset "$DS" --suite "$SUITE"
rc=$?
echo "$(date -Is) step oracle rc=$rc s=$(( $(date +%s) - t0 ))"
[ $rc -eq 0 ] || exit $rc

# algos|backends per stream, in suite order; silvertorch's triton and official share one stream
[ $# -gt 0 ] || set -- "silvertorch|triton official" "linr_v3|triton"
for stream in "$@"; do
  algos=${stream%|*}; backends=${stream#*|}
  $PY "$HERE/pending.py" "$R" "$cv" "$DS" "$SUITE" $algos 2>/dev/null | tail -1
  args=(); for a in $algos; do args+=(--algo "$a"); done; for b in $backends; do args+=(--backend "$b"); done
  name="${SUITE}_${DS}-d128_${algos// /+}_${backends// /+}.log"
  cmd=($PIN $PY -m bench.cli run --dataset "$DS" --dim 128 --suite "$SUITE" "${args[@]}" --out "$R" --resume --interleave)
  t0=$(date +%s)
  { echo "=== ${cmd[*]}"; echo "=== clocks at start"; clocks; } >> "$R/_logs/$name"
  "${cmd[@]}" >> "$R/_logs/$name" 2>&1
  rc=$?
  { echo "=== clocks at end"; clocks; } >> "$R/_logs/$name"
  echo "$(date -Is) stream ${algos// /+} ${backends// /+} rc=$rc s=$(( $(date +%s) - t0 )) log=$name"
  [ $rc -eq 0 ] || exit $rc
done
$PY -c 'import sys; from pathlib import Path; from bench import records; print("=== aggregated", records.aggregate(Path(sys.argv[1])))' "$R"
echo "$(date -Is) driver done"
