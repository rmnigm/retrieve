#!/usr/bin/env bash
# V1-FUSE, second hold of the same window (after driver.sh, whose goodreads timed gate was cut for time):
#   setsid nohup flock /scratch/gpu0.lock bash followup.sh > /scratch/v1-fuse/followup.log 2>&1 &
# goodreads-synth gate with hashes and equality only (V1FUSE_NO_TIMING=1), the per-kernel profile of new vs old V1
# (arxiv-synth p001, clause and bloom), and test_export_kernel_ref.py with the two new ops' input builders.
set -u
REPO=/scratch/wt/v1-fuse
PY=/venvs/v1-fuse/bin/python
OUT=/scratch/v1-fuse/out
A=$REPO/docs/artifacts/campaign-v2/v1-fuse
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v1-fuse HF_HOME=/scratch/hf
PIN="taskset -c 64-127"
cd "$REPO/evaluation"
step() {
  local name=$1; shift
  local t0; t0=$(date +%s)
  "$@"
  local rc=$?
  echo "$(date -Is) step $name rc=$rc s=$(( $(date +%s) - t0 ))"
}
step export_tests $PIN $PY -m pytest "$REPO/retrieve/tests/compile/test_export_kernel_ref.py" -q -p no:cacheprovider \
  > "$OUT/export-tests.log" 2>&1
tail -1 "$OUT/export-tests.log"
step prof_clause $PIN $PY "$A/v1fuse_prof.py" clause "$OUT/prof-clause.json" > "$OUT/prof-clause.log" 2>&1
step prof_bloom $PIN $PY "$A/v1fuse_prof.py" bloom "$OUT/prof-bloom.json" > "$OUT/prof-bloom.log" 2>&1
