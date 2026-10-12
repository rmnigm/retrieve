#!/usr/bin/env bash
# V3-BITS-PUBMED (C2's bit budget at d768) on pod a100-x1-b, GPU 0 shared (st-dloop first): `v3bits` on pubmed d768,
# V3 triton at k_bits {256, 768} x pool {1 %, 5 %}, clause, kept sweeps, seed 0 (12 cells), at the current tag's
# code_version (campaign.yaml), into its own results tree. v3bits has no interleave group: 256 and 768 time apart.
# setsid nohup bash driver.sh > /scratch/v3-bits-pubmed/driver.log 2>&1 &
set -u
W=/scratch/wt/v3-bits-run
PY=/venvs/retrieve/bin/python
R=/scratch/campaign-v2.5/results
L=/scratch/v3-bits-pubmed
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v3-bits-pubmed-v25
PIN="flock /scratch/gpu0.lock taskset -c 0-95"
mkdir -p "$R/_logs" "$L" "$TORCHINDUCTOR_CACHE_DIR"
cd $W/evaluation
EXPECT=$($PY -c "import yaml; print(yaml.safe_load(open('campaign.yaml'))['default']['perf']['code_version'])")
yield_gpu() { while [ -e /scratch/gpu0.st-dloop-wants ] || [ -e /scratch/gpu0.st-dloop-phase ]; do sleep 20; done; }

yield_gpu
cv=$($PIN $PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv (campaign.yaml $EXPECT)"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$L/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null; rm -f /scratch/gpu0.timed' EXIT
step() { local t0=$(date +%s) c="$*"; "$@"; local rc=$?; echo "$(date -Is) step ${c#$PIN } rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc; }
yield_gpu
step $PIN $PY -m bench.cli oracle --dataset pubmed --dim 768 --suite v3bits
yield_gpu
touch /scratch/gpu0.timed
{ echo "=== clocks at start"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$R/_logs/v3bits_pubmed-d768.log"
step $PIN $PY -m bench.cli run --dataset pubmed --dim 768 --suite v3bits --seed 0 --out "$R" --resume --interleave >> "$R/_logs/v3bits_pubmed-d768.log" 2>&1
{ echo "=== clocks at end"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$R/_logs/v3bits_pubmed-d768.log"
rm -f /scratch/gpu0.timed
echo "$(date -Is) driver done"
