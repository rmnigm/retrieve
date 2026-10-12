#!/usr/bin/env bash
# One GPU block on pod b GPU 0 (< 15 min): the phase flag set and trap-removed here, the job under
# flock /scratch/gpu0.lock, /scratch/gpu0.timed set around timed jobs; the build tree's library,
# harness and tests on PYTHONPATH, its own inductor cache.
#   gpu.sh <timed 0|1> <log> <cmd...>
set -u
TIMED=$1 LOG=$2; shift 2
TREE=${TREE:-/scratch/wt/meta-fork-build}
PHASE=/scratch/gpu0.fork-coder-phase WANT=/scratch/gpu0.fork-coder-wants
touch "$PHASE" "$WANT"
trap 'rm -f "$PHASE" "$WANT"' EXIT
export CUDA_VISIBLE_DEVICES=0 RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf \
  TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/meta-fork-$(git -C "$TREE" rev-parse --short HEAD) \
  PYTHONPATH="$TREE/retrieve/src:$TREE/retrieve:$TREE/evaluation:/scratch/meta-fork/pkgs"
flock /scratch/gpu0.lock bash -c '[ "$0" = 1 ] && touch /scratch/gpu0.timed; "${@}"; rc=$?; [ "$0" = 1 ] && rm -f /scratch/gpu0.timed; exit $rc' \
  "$TIMED" "$@" > "$LOG" 2>&1
rc=$?
echo "$(date -Is) rc=$rc: $*" | tee -a "$LOG"
exit $rc
