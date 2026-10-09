#!/usr/bin/env bash
# C7 at d768, the v2.8 repeat (controller, 2026-10-11): SilverTorch triton vs official on PubMed bloom c0_mesh, n_lists 4096,
# n_probe {24, 1024}, seed 0, at campaign-v2.8 (final official adapter) with Meta's extension built -O3 (build_official_o3.sh;
# env.official_build), pod a100-x1-b GPU 0 (st-dloop first). The real `filter` suite, selected (records ok, keyed to the suite),
# --interleave --profile, one process, own venv, fresh inductor dir, own results tree.
# setsid nohup bash driver.sh > /scratch/c7-pubmed-v28/driver.log 2>&1 &
set -u
W=/scratch/wt/c7-pubmed-v28
PY=/venvs/c7-v28/bin/python
R=/scratch/campaign-v2.8/results
L=/scratch/c7-pubmed-v28
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/c7-pubmed-v28
PIN="flock /scratch/gpu0.lock taskset -c 0-95"
SEL="--dataset pubmed --dim 768 --suite filter --algo silvertorch --backend triton --backend official --filter-kind bloom --sweep c0_mesh --seed 0"
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
step $PIN $PY -m bench.cli oracle --dataset pubmed --dim 768 --suite filter --sweep c0_mesh
yield_gpu
touch /scratch/gpu0.timed
{ echo "=== bench run $SEL --out $R --resume --interleave --profile"; echo "=== clocks at start"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$R/_logs/filter_pubmed-d768_c7.log"
step $PIN $PY -m bench.cli run $SEL --out "$R" --resume --interleave --profile >> "$R/_logs/filter_pubmed-d768_c7.log" 2>&1
{ echo "=== clocks at end"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$R/_logs/filter_pubmed-d768_c7.log"
rm -f /scratch/gpu0.timed
echo "$(date -Is) driver done"
