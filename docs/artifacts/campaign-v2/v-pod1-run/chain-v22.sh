#!/usr/bin/env bash
# Pod 1's v2.2 queue after V-V3BITS (orchestrator, 2026-10-09): V-GR-DEEP -> V-YFCC deep -> V-YFCC synth chunks, one lock
# hold. Stops on the first non-zero leg, or between legs when /scratch/v22/chain.stop exists (a surprise found while
# reviewing the previous leg). Launch: setsid nohup flock /scratch/gpu0.lock bash chain-v22.sh > /scratch/v22/chain.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
for leg in "v-gr-deep $HERE/v-gr-deep.sh" "v-yfcc-deep $HERE/v-yfcc-deep.sh" "v-yfcc-synth $HERE/../v-yfcc/synth-chunks.sh"; do
  set -- $leg
  [ -e /scratch/v22/chain.stop ] && { echo "$(date -Is) stop file, not starting $1"; exit 0; }
  mkdir -p "/scratch/v22/$1"
  echo "$(date -Is) start $1"
  bash "$2" > "/scratch/v22/$1/driver.log" 2>&1
  rc=$?
  echo "$(date -Is) end $1 rc=$rc"
  [ $rc -eq 0 ] || exit $rc
done
