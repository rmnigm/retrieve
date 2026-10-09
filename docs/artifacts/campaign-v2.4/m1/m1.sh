#!/usr/bin/env bash
# M1 (roadmap): does a loaded neighbour GPU move timed cells? Pod d, 2x A100-SXM4-80GB, both GPUs on NUMA node 1.
# Measured: arxiv filter d128 clause c0_maincat k 100 seed 0, bs {1, 16}, eager + graph, silvertorch/triton
# (n_probe 24, 256) and linr_v2, perf only, on GPU 0 pinned to cores 64-95; 8 runs in ABBA order
# (A = alone, L = GPU 1 running a neighbour). Neighbour: the same suite's all4 cells with quality, looped, GPU 1, cores 96-127.
#   flock /scratch/gpu0.lock flock /scratch/gpu1.lock bash m1.sh
set -u
PY=/venvs/retrieve/bin/python
R=/scratch/campaign-v2.4/m1
LOG=$R/logs
EXPECT=d67d6263c1f4387d7acc769a42b44532941913d5
ORDER="A L L A A L L A"
export HF_HOME=/scratch/hf
mkdir -p "$LOG"
cd /workspace/retrieve/evaluation

cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }
nvidia-smi -q -d CLOCK > "$LOG/clocks-start.txt"
nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$LOG/clocks.csv" &
SMI=$!
NB=
trap 'kill $SMI 2>/dev/null; [ -n "$NB" ] && kill -- -$NB 2>/dev/null' EXIT

step() {
  local name=$1; shift
  local t0; t0=$(date +%s)
  "$@"
  local rc=$?
  echo "$(date -Is) step $name rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || exit $rc
}

gpu0() { CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/m1-v24-gpu0 taskset -c 64-95 "$@"; }
measure() {
  gpu0 $PY -m bench.cli run --dataset arxiv --suite filter --dim 128 --filter-kind clause --sweep c0_maincat \
    --k 100 --seed 0 --algo silvertorch --algo linr_v2 --backend triton --skip-quality --force \
    --out "$R/$1" > "$LOG/$1.log" 2>&1
}
neighbour() {
  while :; do
    CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/m1-v24-gpu1 taskset -c 96-127 \
      $PY -m bench.cli run --dataset arxiv --suite filter --dim 128 --filter-kind clause --sweep all4 --seed 0 \
      --algo silvertorch --algo linr_v2 --backend triton --force --out "$R/neighbour" >> "$LOG/neighbour.log" 2>&1 || exit 1
  done
}
export -f neighbour; export PY R LOG

step oracle gpu0 $PY -m bench.cli oracle --dataset arxiv --suite filter --dim 128 --sweep c0_maincat --sweep all4
i=0
for c in $ORDER; do
  i=$((i + 1)); tag=$c$i
  if [ $c = L ]; then
    setsid bash -c neighbour & NB=$!
    sleep 90
    kill -0 $NB 2>/dev/null || { echo "$(date -Is) neighbour died"; exit 3; }
  fi
  step "$tag" measure "$tag"
  if [ $c = L ]; then kill -- -$NB; wait $NB 2>/dev/null; NB=; sleep 10; fi
done
nvidia-smi -q -d CLOCK > "$LOG/clocks-end.txt"
step gate $PY /scratch/wt/m1/docs/artifacts/campaign-v2.4/m1/gate.py "$R"
