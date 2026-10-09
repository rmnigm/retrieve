#!/usr/bin/env bash
# V2 graph cross-tree check over three tags (controller, 2026-10-10, surprise from exhibits): campaign-v2.1 (f01255f1), v2.4
# (d67d6263), v2.5 (472f2fc6), each its own worktree and a fresh inductor dir, one process per (tree, round), 4 rounds in rotating
# order on GPU 0; group ax3: arxiv-synth clause p0001 / p01 / p1, V1 + V2 triton, eager + graph, bs 1 + 16, k 100, seed 0, ids hashed.
# The trees read a shadow data root with the 7-rate synth attrs their configs index (/scratch/data-7rates).
# Launch: setsid nohup flock -n /scratch/gpu0.lock bash crosstree3.sh > /scratch/v25/crosstree3/driver.log 2>&1 &
set -u
if flock -n /scratch/gpu0.lock true; then echo "$(date -Is) not launched under /scratch/gpu0.lock, refusing"; exit 3; fi
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/venvs/retrieve/bin/python
OUT=/scratch/v25/crosstree3
PIN="taskset -c 0-63,128-191"
declare -A TREE=([v21]=/scratch/wt/tree-v21 [v24]=/scratch/wt/tree-v24 [v25]=/scratch/wt/tree-v25) WANT=(
  [v21]=f01255f106214ec0f540352d0e2a5cd90b84b6a1 [v24]=d67d6263c1f4387d7acc769a42b44532941913d5
  [v25]=472f2fc68c697179b463d3b5a6b19ade194e6e2f)
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hf
mkdir -p "$OUT"
for t in v21 v24 v25; do
  got=$(git -C "${TREE[$t]}" rev-parse HEAD:retrieve/src/retrieve)
  [ "$got" = "${WANT[$t]}" ] && [ -z "$(git -C "${TREE[$t]}" status --porcelain retrieve/src)" ] || { echo "$t tree $got, refusing"; exit 2; }
  rm -rf "/scratch/inductor/ct3-$t" && mkdir -p "/scratch/inductor/ct3-$t"  # fresh inductor dir per tree
done
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$OUT/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
inrun() {
  local t=$1; shift
  (cd "${TREE[$t]}/evaluation" && PYTHONPATH="${TREE[$t]}/evaluation:${TREE[$t]}/retrieve/src" \
    TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/ct3-$t $PIN $PY "$@")
}
step() {
  local name=$1; shift
  local t0=$(date +%s)
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$(date -Is) $name rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || exit $rc
}
step oracle inrun v25 -m bench.cli oracle --dataset arxiv-synth --suite synth --sweep p0001 --sweep p01 --sweep p1
i=0
for order in "v21 v24 v25" "v25 v24 v21" "v24 v25 v21" "v21 v25 v24"; do
  i=$((i + 1))
  for t in $order; do step "ax3-$t-r$i" inrun "$t" "$HERE/crosstree.py" ax3 "$OUT/ax3-$t-r$i.json"; done
done
$PY "$HERE/crosstree3_summary.py" "$OUT" | tee "$OUT/summary.txt"
echo "$(date -Is) driver done"
