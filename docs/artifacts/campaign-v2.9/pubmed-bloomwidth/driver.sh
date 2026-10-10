#!/usr/bin/env bash
# D3 PubMed bloomwidth at campaign-v2.9 (C4 at 10 M d768; controller night queue 3): the real `bloomwidth` (quality only) and
# `bloomwidth-timed` suites, pubmed c0_mesh, both blooms, seed 0; then C7 seeds 1, 2 (the C7 v2.9 `filter`
# selection at --seed 1, 2), then D3 seeds 1-2.
# Library e8958bd2 + H-ARMFREE harness, Meta -O3 venv (/venvs/c7-v28), fresh inductor, pod a100-x1-b GPU 0 (st-dloop first).
# setsid nohup bash driver.sh > /scratch/pubmed-bw-v29/driver.log 2>&1 &
set -u
W=/scratch/wt/pubmed-bw-v29
PY=/venvs/c7-v28/bin/python
R=/scratch/campaign-v2.9/results
L=/scratch/pubmed-bw-v29
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/pubmed-bw-v29
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
run() {  # $1 log name, $2 timed 0|1, rest: bench run selection
  local log="$R/_logs/$1.log" t=$2; shift 2
  yield_gpu; [ $t = 1 ] && touch /scratch/gpu0.timed
  { echo "=== bench run $* --out $R --resume --interleave"; echo "=== clocks at start"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$log"
  step $PIN $PY -m bench.cli run "$@" --out "$R" --resume --interleave >> "$log" 2>&1
  { echo "=== clocks at end"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$log"
  rm -f /scratch/gpu0.timed
}
for s in bloomwidth bloomwidth-timed; do yield_gpu; step $PIN $PY -m bench.cli oracle --dataset pubmed --dim 768 --suite $s; done
run bloomwidth_pubmed-d768 0 --dataset pubmed --dim 768 --suite bloomwidth --seed 0
run bloomwidth-timed_pubmed-d768 1 --dataset pubmed --dim 768 --suite bloomwidth-timed --seed 0
c7() { run filter_pubmed-d768_c7-seed$1 1 --dataset pubmed --dim 768 --suite filter --algo silvertorch --backend triton \
  --backend official --filter-kind bloom --sweep c0_mesh --seed $1 --profile; }
c7 1
c7 2
for seed in 1 2; do  # the follow-on (controller GO): D3 seeds 1-2
  run bloomwidth_pubmed-d768 0 --dataset pubmed --dim 768 --suite bloomwidth --seed $seed
  run bloomwidth-timed_pubmed-d768 1 --dataset pubmed --dim 768 --suite bloomwidth-timed --seed $seed
done
echo "$(date -Is) driver done"
