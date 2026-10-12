#!/usr/bin/env bash
# V2's 0.21x on YFCC between campaign-v2.5 and v2.9 (controller 2026-10-12): one torch.profiler process per tree over 20 V2 graph replays
# at bs 16 on yfcc10m tags_and (crosstree.py group yf --profile), each tree its own venv and a fresh inductor dir.
# Launch: setsid nohup flock -n /scratch/gpu0.lock bash v2prof-yfcc.sh > /scratch/v29/v2prof-yfcc/driver.log 2>&1 &
set -u
if flock -n /scratch/gpu0.lock true; then echo "$(date -Is) not launched under /scratch/gpu0.lock, refusing"; exit 3; fi
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/scratch/v29/v2prof-yfcc
declare -A TREE=([v25]=/scratch/wt/tree-v25 [v29]=/scratch/wt/tree-v29) PY=([v25]=/venvs/retrieve/bin/python [v29]=/venvs/v29/bin/python)
declare -A WANT=([v25]=472f2fc68c697179b463d3b5a6b19ade194e6e2f [v29]=e8958bd234cf9f814d1afcb2e5eb3f95e25dbec3)
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hf
mkdir -p "$OUT"
for t in v25 v29; do
  got=$(git -C "${TREE[$t]}" rev-parse HEAD:retrieve/src/retrieve)
  [ "$got" = "${WANT[$t]}" ] && [ -z "$(git -C "${TREE[$t]}" status --porcelain retrieve/src evaluation/bench)" ] || { echo "$t tree $got, refusing"; exit 2; }
  rm -rf "/scratch/inductor/v2prof-$t" && mkdir -p "/scratch/inductor/v2prof-$t"
  t0=$(date +%s)
  (cd "${TREE[$t]}/evaluation" && PYTHONPATH="${TREE[$t]}/evaluation:${TREE[$t]}/retrieve/src" TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v2prof-$t \
    taskset -c 0-63,128-191 "${PY[$t]}" "$HERE/crosstree.py" yf "$OUT/prof-$t.json" --profile) > "$OUT/prof-$t.log" 2>&1
  rc=$?; echo "$(date -Is) prof-$t rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc
done
echo "$(date -Is) driver done"
