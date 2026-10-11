#!/usr/bin/env bash
# V-PILOT driver: goodreads-synth `synth` suite at campaign-v2, GPU 0, one sequential process group.
# Launch: setsid nohup bash driver.sh > /scratch/v-pilot/driver.log 2>&1 &
set -u
REPO=/workspace/retrieve
PY=/venvs/retrieve/bin/python
R=/scratch/campaign-v2/results
LOG=/scratch/v-pilot
EXPECT=408b1188d542634b3d18a2f5077bd23537a845fc
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v-pilot HF_HOME=/scratch/hf
PIN="taskset -c 0-63,128-191"
mkdir -p "$R" "$LOG" "$TORCHINDUCTOR_CACHE_DIR"
cd "$REPO/evaluation"

cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }

nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 > "$LOG/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

t0=$(date +%s)
$PIN $PY -m bench.cli oracle --dataset goodreads-synth --suite synth
rc=$?
echo "$(date -Is) step oracle rc=$rc s=$(( $(date +%s) - t0 ))"
[ $rc -eq 0 ] || exit $rc

t0=$(date +%s)
$PIN $PY -m bench.cli campaign --suite synth --dataset goodreads-synth --resume --interleave --out "$R"
rc=$?
echo "$(date -Is) step campaign rc=$rc s=$(( $(date +%s) - t0 ))"
exit $rc
