#!/usr/bin/env bash
# D3 PubMed `bloomwidth-timed` at campaign-v2.3 on pod a100-x1-b (1x A100-SXM4-80GB, GPU 0 shared through
# /scratch/gpu0.lock, NUMA node 0 cores): its oracle rebuilt at v2.1, then the timed campaign
# (`--interleave`), pubmed d768 bloom c0_mesh, seeds 0-2, into a fresh v2.1 results tree.
# setsid nohup bash driver.sh > /scratch/d3-pubmed-timed-v23/driver.log 2>&1 &
set -u
W=/scratch/wt/v-pubmed-v23
PY=/venvs/retrieve/bin/python
R=/scratch/campaign-v2.3/results
L=/scratch/d3-pubmed-timed-v23
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/d3-pubmed-timed-v23
PIN="flock /scratch/gpu0.lock taskset -c 0-95"
mkdir -p "$R/_logs" "$L" "$TORCHINDUCTOR_CACHE_DIR"
cd $W/evaluation
yield_gpu() { while [ -e /scratch/gpu0.st-dloop-wants ] || [ -e /scratch/gpu0.st-dloop-phase ]; do sleep 20; done; }   # st-dloop first
EXPECT=$($PY -c "import yaml; print(yaml.safe_load(open('$W/evaluation/campaign.yaml'))['default']['perf']['code_version'])")

yield_gpu
cv=$($PIN $PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv (campaign.yaml $EXPECT)"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }

nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$L/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null; rm -f /scratch/gpu0.timed' EXIT
nvidia-smi -q -d CLOCK > "$L/nvidia-smi-clock-start.txt"

step() { local t0=$(date +%s) c="$*"; "$@"; local rc=$?; echo "$(date -Is) step ${c#$PIN } rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc; }
yield_gpu
step $PIN $PY -m bench.cli oracle --dataset pubmed --suite bloomwidth-timed
yield_gpu
touch /scratch/gpu0.timed   # pod b: the neighbour worker pauses CPU-heavy work while it exists
step $PIN $PY -m bench.cli campaign --suite bloomwidth-timed --dataset pubmed --dim 768 --out "$R" \
  --resume --interleave --timeout 48
rm -f /scratch/gpu0.timed
nvidia-smi -q -d CLOCK > "$L/nvidia-smi-clock-end.txt"
echo "$(date -Is) driver done"
