#!/usr/bin/env bash
# PubMed F3 / C6 10 M panel at campaign-v2.9 (controller, 2026-10-11), one version, tree and box: the `deep` PubMed slot
# (SilverTorch triton clause, n_lists 4096, n_probe 24 / 64 / 256 / 1024, kept sweeps) and the exact V1 / V2 references from
# `filter` (clause, interleaved V1 vs V2), seed 0, full variants, own worktree venv (Triton only), fresh inductor dir, pod
# a100-x1-b GPU 0 (st-dloop first). One process per arm group.
# setsid nohup bash driver.sh > /scratch/pubmed-f3-v29/driver.log 2>&1 &
set -u
W=/scratch/wt/pubmed-f3-v29
PY=/venvs/f3-v29/bin/python
R=/scratch/campaign-v2.9/results
L=/scratch/pubmed-f3-v29
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/pubmed-f3-v29
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
run() {  # $1 log name, rest: bench run selection
  local log="$R/_logs/$1.log"; shift
  yield_gpu; touch /scratch/gpu0.timed
  { echo "=== bench run $* --out $R --resume --interleave"; echo "=== clocks at start"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$log"
  step $PIN $PY -m bench.cli run "$@" --out "$R" --resume --interleave >> "$log" 2>&1
  { echo "=== clocks at end"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$log"
  rm -f /scratch/gpu0.timed
}
yield_gpu
step $PIN $PY -m bench.cli oracle --dataset pubmed --dim 768 --suite deep
run deep_pubmed-d768_silvertorch_triton --dataset pubmed --dim 768 --suite deep --seed 0
run filter_pubmed-d768_linr_v1_filter_mask+linr_v2_triton --dataset pubmed --dim 768 --suite filter \
  --algo linr_v1_filter_mask --algo linr_v2 --backend triton --filter-kind clause --seed 0
echo "$(date -Is) driver done"
