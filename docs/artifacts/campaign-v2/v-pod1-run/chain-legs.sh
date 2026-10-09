#!/usr/bin/env bash
# Pod 1's queue as arguments (orchestrator, 2026-10-10): the named legs in order, one lock hold, at the tag common.sh
# defaults to. Legs: v-gr-deep v-router v-yfcc-deep v-yfcc-synth gr-synth-mid yfcc-int8. Stops on the first non-zero leg, or between legs when
# /scratch/<tv>/chain.stop exists. Launch: setsid nohup flock -n /scratch/gpu0.lock bash chain-legs.sh LEG... > /scratch/v25/chain.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:-campaign-v2.5}
D=/scratch/$(echo "${TAG#campaign-}" | tr -d .)
declare -A DRIVER=([v-gr-deep]=$HERE/v-gr-deep.sh [v-router]=$HERE/v-router.sh [v-yfcc-deep]=$HERE/v-yfcc-deep.sh
                   [v-yfcc-synth]=$HERE/v-yfcc-synth.sh [gr-synth-mid]=$HERE/gr-synth-mid.sh [yfcc-int8]=$HERE/yfcc-int8.sh)
# legs named in $D/prepend (one line) run first: a check inserted ahead of an already-queued chain
if [ -s "$D/prepend" ]; then set -- $(cat "$D/prepend") "$@"; rm -f "$D/prepend"; fi
for leg in "$@"; do [ -n "${DRIVER[$leg]:-}" ] || { echo "unknown leg $leg"; exit 2; }; done
# the chain starts between legs (the lock is ours, nothing runs from the main checkout): take staging's harness now
git -C /workspace/retrieve pull -q --ff-only origin staging && echo "$(date -Is) main checkout $(git -C /workspace/retrieve log --oneline -1)"
for leg in "$@"; do
  [ -e "$D/chain.stop" ] && { echo "$(date -Is) stop file, not starting $leg"; exit 0; }
  mkdir -p "$D/$leg"
  echo "$(date -Is) start $leg"
  bash "${DRIVER[$leg]}" > "$D/$leg/driver.log" 2>&1
  rc=$?
  echo "$(date -Is) end $leg rc=$rc"
  [ $rc -eq 0 ] || exit $rc
done
