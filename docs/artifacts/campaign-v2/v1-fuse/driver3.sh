#!/usr/bin/env bash
# V1-FUSE third window: clause only, on a fresh inductor cache keyed by code_version (window 2 shared one cache dir
# across two code versions and its graph cells replayed window 1's 512-lane clause kernel). Under the pod's lock:
#   setsid nohup flock /scratch/gpu0.lock bash driver3.sh > /scratch/v1-fuse/driver3.log 2>&1 &
# The library suite on the final tree; the clause gate on both datasets (ids vs the v2.1 records, equality to the old composition, capture,
# interleaved keep-rule and graph-vs-eager timing). Each step's status line lands in this log, its output in OUT/<log>.
set -u
REPO=/scratch/wt/v1-fuse
PY=/venvs/v1-fuse/bin/python
OUT=/scratch/v1-fuse/out3
A=$REPO/docs/artifacts/campaign-v2/v1-fuse
CV=$(git -C "$REPO" rev-parse HEAD:retrieve/src/retrieve)
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v1-fuse-$CV HF_HOME=/scratch/hf V1FUSE_KINDS=clause
PIN="taskset -c 64-127"
mkdir -p "$OUT" "$TORCHINDUCTOR_CACHE_DIR"
cd "$REPO/evaluation"
case "$($PY -c 'import retrieve; print(retrieve.__file__)')" in
  "$REPO"/*) ;;
  *) echo "$(date -Is) retrieve not imported from $REPO, refusing"; exit 2 ;;
esac
$PY -m bench.cli env > "$OUT/env.json"
echo "$(date -Is) code_version $($PY -c 'import json; print(json.load(open("'"$OUT"'/env.json"))["code_version"])')"
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$OUT/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
run() {
  local name=$1 log=$2; shift 2
  local t0; t0=$(date +%s)
  "$@" > "$OUT/$log" 2>&1
  local rc=$?
  echo "$(date -Is) step $name rc=$rc s=$(( $(date +%s) - t0 ))"
}
run library_suite library-suite.log $PIN $PY -m pytest "$REPO/retrieve/tests" -q -p no:cacheprovider
tail -1 "$OUT/library-suite.log"
run gate_arxiv gate-arxiv-synth.log $PIN $PY "$A/v1fuse_gpu.py" gate arxiv-synth \
  /scratch/campaign-v2.1/results-arxiv-synth-synth/synth/arxiv-synth-d128.jsonl "$OUT/gate-arxiv-synth.json"
run gate_goodreads gate-goodreads-synth.log $PIN $PY "$A/v1fuse_gpu.py" gate goodreads-synth \
  /scratch/v1-fuse/hub/gr-synth/synth/goodreads-synth-d128.jsonl "$OUT/gate-goodreads-synth.json"
echo "$(date -Is) driver3 done"
