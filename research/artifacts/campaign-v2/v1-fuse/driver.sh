#!/usr/bin/env bash
# V1-FUSE GPU window on pod c (a100-x1-c), GPU 0, under the pod's lock, its own venv, every check in one hold:
#   setsid nohup flock /scratch/gpu0.lock bash driver.sh > /scratch/v1-fuse/driver.log 2>&1 &
# Steps: the library suite; v1fuse_gpu.py gate (ids vs the campaign-v2.1 records + the old composition, graph capture,
# interleaved keep-rule and graph-vs-eager timing) on arxiv-synth then goodreads-synth; v1fuse_gpu.py fused (item 1,
# experiment only). Outputs in /scratch/v1-fuse/out (to the Hub as artifacts/v1-fuse).
set -u
REPO=/scratch/wt/v1-fuse
PY=/venvs/v1-fuse/bin/python
OUT=/scratch/v1-fuse/out
G=$REPO/docs/artifacts/campaign-v2/v1-fuse/v1fuse_gpu.py
AX_REC=/scratch/campaign-v2.1/results-arxiv-synth-synth/synth/arxiv-synth-d128.jsonl
GR_REC=/scratch/v1-fuse/hub/gr-synth/synth/goodreads-synth-d128.jsonl
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v1-fuse HF_HOME=/scratch/hf
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

step() {
  local name=$1; shift
  local t0; t0=$(date +%s)
  "$@"
  local rc=$?
  echo "$(date -Is) step $name rc=$rc s=$(( $(date +%s) - t0 ))"
}

step library_suite $PIN $PY -m pytest "$REPO/retrieve/tests" -q -p no:cacheprovider --junitxml="$OUT/library-suite.xml" \
  > "$OUT/library-suite.log" 2>&1
tail -3 "$OUT/library-suite.log"
step gate_arxiv $PIN $PY "$G" gate arxiv-synth "$AX_REC" "$OUT/gate-arxiv-synth.json" > "$OUT/gate-arxiv-synth.log" 2>&1
step gate_goodreads $PIN $PY "$G" gate goodreads-synth "$GR_REC" "$OUT/gate-goodreads-synth.json" > "$OUT/gate-goodreads-synth.log" 2>&1
step fused_arxiv $PIN $PY "$G" fused arxiv-synth "$OUT/fused-arxiv-synth.json" > "$OUT/fused-arxiv-synth.log" 2>&1
step fused_goodreads $PIN $PY "$G" fused goodreads-synth "$OUT/fused-goodreads-synth.json" > "$OUT/fused-goodreads-synth.log" 2>&1
