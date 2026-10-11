#!/usr/bin/env bash
# v2.6 regression split (controller 2026-10-10 224500000): campaign-v2.5 (472f2fc6) vs v2.6 (20e83bfc) on group ax3, v2.6 twice —
# attrs straight from the pool (v26) and through run.prepare_pool, the harness path (v26p); fresh inductor dir per variant, 3 rotating
# rounds, then one torch.profiler process per variant (bs 16, V1 + V2, eager + graph), then the 8-cell `bench run` from each tree.
# Launch: setsid nohup flock -n /scratch/gpu0.lock bash crosstree4.sh > /scratch/v26/crosstree4/driver.log 2>&1 &
set -u
if flock -n /scratch/gpu0.lock true; then echo "$(date -Is) not launched under /scratch/gpu0.lock, refusing"; exit 3; fi
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/venvs/retrieve/bin/python
OUT=/scratch/v26/crosstree4
PIN="taskset -c 0-63,128-191"
declare -A TREE=([v25]=/scratch/wt/tree-v25 [v26]=/scratch/wt/tree-v26 [v26p]=/scratch/wt/tree-v26) FLAG=([v25]= [v26]= [v26p]=--prep)
declare -A WANT=([v25]=472f2fc68c697179b463d3b5a6b19ade194e6e2f [v26]=20e83bfc3c2875258d504233311e30d278d21438)
WANT[v26p]=${WANT[v26]}
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hf
mkdir -p "$OUT"
for t in v25 v26 v26p; do
  got=$(git -C "${TREE[$t]}" rev-parse HEAD:retrieve/src/retrieve)
  [ "$got" = "${WANT[$t]}" ] && [ -z "$(git -C "${TREE[$t]}" status --porcelain retrieve/src evaluation/bench)" ] || { echo "$t tree $got, refusing"; exit 2; }
  rm -rf "/scratch/inductor/ct4-$t" && mkdir -p "/scratch/inductor/ct4-$t"
done
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$OUT/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
inrun() {
  local t=$1; shift
  (cd "${TREE[$t]}/evaluation" && PYTHONPATH="${TREE[$t]}/evaluation:${TREE[$t]}/retrieve/src" \
    TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/ct4-$t $PIN $PY "$@")
}
step() {
  local name=$1; shift
  local t0=$(date +%s)
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$(date -Is) $name rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || exit $rc
}
i=0
for order in "v25 v26 v26p" "v26p v26 v25" "v26 v25 v26p"; do
  i=$((i + 1))
  for t in $order; do step "ax3-$t-r$i" inrun "$t" "$HERE/crosstree.py" ax3 "$OUT/ax3-$t-r$i.json" ${FLAG[$t]}; done
done
for t in v25 v26 v26p; do step "prof-$t" inrun "$t" "$HERE/crosstree.py" ax3 "$OUT/prof-$t.json" --profile ${FLAG[$t]}; done
NARROW="--dataset arxiv-synth --dim 128 --suite synth --sweep p0001 --sweep p001 --sweep p01 --sweep p1 --algo linr_v1_filter_mask
  --algo linr_v2 --backend triton --filter-kind clause --seed 0 --k 100 --interleave"
for t in v25 v26; do step "bench-$t" inrun "$t" -m bench.cli run $NARROW --out "$OUT/bench-$t"; done
$PY "$HERE/crosstree4_summary.py" "$OUT" | tee "$OUT/summary.txt"
echo "$(date -Is) driver done"
