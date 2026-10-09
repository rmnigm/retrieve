#!/usr/bin/env bash
# The YFCC int8-precision check (controller 2026-10-10): quality only, yfcc10m-synth p 1 and p 0.01, at the tag in common.sh.
LEG=yfcc-int8
. "$(dirname "$(readlink -f "$0")")/common.sh"
step oracle oracle --dataset yfcc10m-synth --suite synth --sweep p1 --sweep p001
for sw in p1 p001; do
  t0=$(date +%s)
  $PIN $PY "$HERE/../../campaign-v2.5/yfcc-int8/int8_check.py" yfcc10m-synth "$sw" "$LOG/int8-$sw.json" > "$LOG/int8-$sw.log" 2>&1
  rc=$?; echo "$(date -Is) int8 $sw rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc
done
finish
