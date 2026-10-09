#!/usr/bin/env bash
# Pod 1's queue (orchestrator, 2026-10-09): V-GR-DEEP -> V-YFCC deep -> V-YFCC synth chunks, one lock hold, at the tag
# common.sh defaults to. Stops on the first non-zero leg, or between legs when /scratch/<tv>/chain.stop exists (a surprise
# found while reviewing a leg, or a gate between legs). Launch: setsid nohup flock /scratch/gpu0.lock bash chain.sh > /scratch/v23/chain.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:-campaign-v2.3}
D=/scratch/$(echo "${TAG#campaign-}" | tr -d .)
for leg in "v-gr-deep $HERE/v-gr-deep.sh" "v-yfcc-deep $HERE/v-yfcc-deep.sh" "v-yfcc-synth $HERE/../v-yfcc/synth-chunks.sh"; do
  set -- $leg
  [ -e $D/chain.stop ] && { echo "$(date -Is) stop file, not starting $1"; exit 0; }
  mkdir -p "$D/$1"
  echo "$(date -Is) start $1"
  bash "$2" > "$D/$1/driver.log" 2>&1
  rc=$?
  echo "$(date -Is) end $1 rc=$rc"
  [ $rc -eq 0 ] || exit $rc
done
