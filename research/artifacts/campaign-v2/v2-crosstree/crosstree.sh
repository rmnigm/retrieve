#!/usr/bin/env bash
# V2 cross-tree check driver: old (tag campaign-v2, library 408b1188) vs new (campaign-v2.1, f01255f1), one
# process per (tree, group, repeat), ABAB for 4 pairs per group, then one profile process per tree and group.
# Oracles are built first by the new tree (both trees read the same v4 blobs: the fingerprint ignores code).
# Launch: setsid nohup flock -n /scratch/gpu0.lock bash crosstree.sh > /scratch/v21/v2-crosstree/driver.log 2>&1 &
set -u
if flock -n /scratch/gpu0.lock true; then echo "$(date -Is) not launched under /scratch/gpu0.lock, refusing"; exit 3; fi
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/venvs/retrieve/bin/python
OUT=/scratch/v2-crosstree/out
LOG=/scratch/v21/v2-crosstree
PAIRS=${PAIRS:-4}
GROUPS_=${GROUPS_:-gs pm}
declare -A TREE=([old]=/scratch/wt/tree-408b [new]=/scratch/wt/tree-v21) WANT=(
  [old]=408b1188d542634b3d18a2f5077bd23537a845fc [new]=f01255f106214ec0f540352d0e2a5cd90b84b6a1)
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hf
PIN="taskset -c 0-63,128-191"
mkdir -p "$OUT" "$LOG"
for t in old new; do
  got=$(git -C "${TREE[$t]}" rev-parse HEAD:retrieve/src/retrieve)
  [ "$got" = "${WANT[$t]}" ] && [ -z "$(git -C "${TREE[$t]}" status --porcelain retrieve/src)" ] \
    || { echo "$(date -Is) $t tree is $got (dirty?), refusing"; exit 2; }
done
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$LOG/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
inrun() {  # tree, then a command run from the tree's evaluation/ with its packages first on the path
  local t=$1; shift
  (cd "${TREE[$t]}/evaluation" && PYTHONPATH="${TREE[$t]}/evaluation:${TREE[$t]}/retrieve/src" \
    TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v2-crosstree-$t $PIN $PY "$@")
}
step() {
  local name=$1; shift
  local t0=$(date +%s)
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "$(date -Is) $name rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || exit $rc
}
step move-oracles inrun new "$HERE/../v-pod1-run/move_old_oracles.py" "${WANT[new]}" goodreads-synth pubmed
step oracle-gs inrun new -m bench.cli oracle --dataset goodreads-synth --suite synth --sweep p1 --sweep p0001
step oracle-pm inrun new -m bench.cli oracle --dataset pubmed --suite filter --sweep c3_journal_reverse
step move-oracles-ax inrun new "$HERE/../v-pod1-run/move_old_oracles.py" "${WANT[new]}" arxiv
step oracle-ax inrun new -m bench.cli oracle --dataset arxiv --suite filter --sweep c3_nversions
step oracle-axs inrun new -m bench.cli oracle --dataset arxiv-synth --suite synth --sweep p1
for g in $GROUPS_; do
  for i in $(seq 1 "$PAIRS"); do
    if [ $((i % 2)) -eq 1 ]; then order="old new"; else order="new old"; fi
    for t in $order; do step "$g-$t-r$i" inrun "$t" "$HERE/crosstree.py" "$g" "$OUT/$g-$t-r$i.json"; done
  done
  for t in old new; do step "$g-$t-prof" inrun "$t" "$HERE/crosstree.py" "$g" "$OUT/$g-$t-prof.json" --profile; done
done
# programs sweep: the v2.1 tree's variants and the old tree's kernel, old/new/new/old over the cells
for c in pm ax axs; do
  if [ "$c" = ax ]; then order="new old"; else order="old new"; fi
  for t in $order; do step "sweep-$c-$t" inrun "$t" "$HERE/sweep.py" "$c" "$OUT/sweep-$c-$t.json"; done
done
$PY "$HERE/sweep_summary.py" "$OUT" | tee "$LOG/sweep-summary.txt"
$PY "$HERE/crosstree_summary.py" "$OUT" | tee "$LOG/summary.txt"
echo "$(date -Is) driver done"
