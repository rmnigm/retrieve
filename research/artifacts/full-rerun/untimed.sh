#!/usr/bin/env bash
# FULL-RERUN untimed steps at one seed (recipe.yaml `untimed`), on the pod's own datasets, outside every timed call:
# grade_bloom (the bloom's false positives on the probed docs at the timed points), ceiling_adapted (C6 split); at seed 0 also
# fp_adapted (seed-free: the bloom does not depend on the k-means) on pode and score_gaps (CPU) on pod1. Each to fr/untimed-*.
# usage: untimed.sh pod1|pode SEED "CEILING GROUPS"     env as queue.sh
set -u
POD=$1 S=$2 CEILG=$3 FR=${FR:-/scratch/fr} PY=$VENV/bin/python H=$(cd "$(dirname "$0")" && pwd)
RD=$TREE/research/artifacts/results-draft LOCK=${LOCK:-/scratch/gpu0.lock}
O=$FR/out/untimed-$POD-s$S; mkdir -p "$O"; [ -f "$O/.done" ] && exit 0
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hf RETRIEVE_DATA_ROOT=/data SEED=$S
cd "$TREE/evaluation"
if [ "$POD" = pod1 ]; then GB="d768:pubmed d256:laion1m,laion3m,laion10m"; else GB="d128:goodreads,arxiv d256:laion30m"; fi
for g in $GB; do
  dim=${g%%:*} ds=${g#*:}
  "$PY" "$H/points.py" "$FR/picks.yaml" ${ds//,/ } > "$O/points-$dim.json"
  flock "$LOCK" "$PY" "$RD/grade_bloom.py" "$CFG" "fr-curve-$dim" "$O/points-$dim.json" "$O/grade_bloom-$dim.csv" > "$O/grade_bloom-$dim.log" 2>&1 \
    || echo "grade_bloom $dim" >> "$O/FAILED"
done
SUITE=fr-ceil-adapted flock "$LOCK" "$PY" "$RD/ceiling_adapted.py" "$CFG" "$O/ceiling" $CEILG > "$O/ceiling.log" 2>&1 \
  || echo "ceiling_adapted" >> "$O/FAILED"
if [ "$S" = 0 ] && [ "$POD" = pode ]; then
  SUITE=fr-bloomwidth flock "$LOCK" "$PY" "$RD/fp_adapted.py" "$CFG/suites.yaml" "$O/fp_adapted.csv" > "$O/fp_adapted.log" 2>&1 \
    || echo "fp_adapted" >> "$O/FAILED"
fi
if [ "$S" = 0 ] && [ "$POD" = pod1 ]; then
  "$PY" "$RD/score_gaps.py" "$O/score_gaps.json" > "$O/score_gaps.log" 2>&1 || echo "score_gaps" >> "$O/FAILED"
fi
"$PY" -m bench.cli upload --results "$O" --path-in-repo "fr/untimed-$POD-s$S" --verify > "$O/upload.log" 2>&1 || echo upload >> "$O/FAILED"
[ -s "$O/FAILED" ] || touch "$O/.done"
