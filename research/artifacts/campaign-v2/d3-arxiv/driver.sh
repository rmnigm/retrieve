#!/usr/bin/env bash
# D3 arXiv driver on pod c (a100-x1-c): one bloomwidth suite on arxiv d128 at campaign-v2, GPU 0, its own
# results tree. Run under the pod's GPU lock, one suite per hold:
#   flock /scratch/gpu0.lock bash driver.sh bloomwidth           (quality-only, perf: false)
#   flock /scratch/gpu0.lock bash driver.sh bloomwidth-timed     (--interleave)
# The shared venv imports `retrieve` from $LIB while code_version is read from $REPO's git tree:
# the driver refuses unless both library trees are identical and at $EXPECT.
set -u
SUITE=$1; DS=arxiv; TAG=$DS-$SUITE
REPO=/scratch/wt/d3-arxiv
LIB=/workspace/retrieve
PY=/venvs/retrieve/bin/python
R=/scratch/campaign-v2/results-$TAG
LOG=/scratch/$TAG
EXPECT=408b1188d542634b3d18a2f5077bd23537a845fc
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/$TAG HF_HOME=/scratch/hf
PIN="taskset -c 64-127"
mkdir -p "$R" "$LOG" "$TORCHINDUCTOR_CACHE_DIR"
cd "$REPO/evaluation"

[ "$($PY -c 'import retrieve; print(retrieve.__file__)')" = "$LIB/retrieve/src/retrieve/__init__.py" ] \
  || { echo "$(date -Is) retrieve not imported from $LIB, refusing"; exit 2; }
diff -rq --exclude=__pycache__ "$REPO/retrieve/src/retrieve" "$LIB/retrieve/src/retrieve" \
  || { echo "$(date -Is) library trees differ, refusing"; exit 2; }
cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }

nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$LOG/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
nvidia-smi -q -d CLOCK > "$LOG/clock-q-start.txt"

t0=$(date +%s)
$PIN $PY -m bench.cli oracle --dataset $DS --dim 128 --suite "$SUITE"
rc=$?
echo "$(date -Is) step oracle rc=$rc s=$(( $(date +%s) - t0 ))"
[ $rc -eq 0 ] || exit $rc

IL=(); [ "$SUITE" = bloomwidth-timed ] && IL=(--interleave)
t0=$(date +%s)
$PIN $PY -m bench.cli campaign --suite "$SUITE" --dataset $DS --dim 128 --resume "${IL[@]}" --timeout 48 --out "$R"
rc=$?
echo "$(date -Is) step campaign rc=$rc s=$(( $(date +%s) - t0 ))"
nvidia-smi -q -d CLOCK > "$LOG/clock-q-end.txt"
exit $rc
