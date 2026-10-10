#!/usr/bin/env bash
# C6 recall-ceiling check on goodreads + arXiv at campaign-v2.10 (controller 2026-10-12): for each sweep in points.json (dataset, suite,
# sweep, n_lists, n_probe = the sweep's best clause point in the v2.10 tuning grid), ceiling_check.py scores the probed, filter-passing
# items four ways (shipped, global int8 recomputed, per-row int8, fp16); quality only. Under common.sh (lock, code_version, clocks).
LEG=ceiling-v210
TAG=${TAG:-campaign-v2.10}
REPO=${REPO:-/scratch/wt/tree-v210}
PY=${PY:-/venvs/v210/bin/python}
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2/v-pod1-run/common.sh"
CK="$(dirname "$(readlink -f "$0")")"
$PY - "$CK/points.json" <<'PY' > "$LOG/points.txt"
import json, sys
for ds, suite, sw, nl, npb, _ in json.load(open(sys.argv[1])): print(ds, suite, sw, nl, npb)
PY
while read -r ds suite sw nl npb; do
  t0=$(date +%s)
  $PIN $PY "$CK/ceiling_check.py" $ds $suite $sw $nl $npb "$LOG/ceiling-$ds-$sw.json" > "$LOG/ceiling-$ds-$sw.log" 2>&1
  rc=$?; echo "$(date -Is) step $ds/$sw rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc
done < "$LOG/points.txt"
finish
