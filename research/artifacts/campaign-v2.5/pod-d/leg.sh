#!/usr/bin/env bash
# One whole synth-style leg on one pod-d GPU, child by child (children.py: the --interleave campaign children, checked
# to partition the suite), each `bench run --resume --interleave` into one tree, uploaded cumulatively as
# $GRP<N> after child N; at the end the tree must have nothing pending and goes up as $LEGUP. Rerun after a
# crash: resume skips finished cells, a child with an upload marker is skipped whole. Stops on any non-zero step,
# and between children when /scratch/v25/$LEG.stop exists. SKIP_PERF=1 (M1 failed, a quality-only GPU): every child
# --skip-perf (partial records, which a later timed pass reruns), uploads as $GRP<N>-quality, no leg upload.
#   GPU=0 LEG=v-ax-synth DS=arxiv-synth SUITE=synth GRP=artifacts/v-ax-synth-v25-group LEGUP=campaign-v2.5/arxiv-synth-synth \
#     setsid nohup flock -n /scratch/gpu0.lock bash leg.sh > /scratch/v25/v-ax-synth.driver.log 2>&1 &
: "${DS:?}" "${SUITE:?}" "${GRP:?}" "${LEGUP:?}"
R=${R:-/scratch/campaign-v25/$DS-$SUITE}
. "$(dirname "$(readlink -f "$0")")/common.sh"
PENDING=$HERE/../../campaign-v2/v-gr-deep/pending.py

$PY "$HERE/children.py" "$DS" "$SUITE" "$(dim "$DS")" > "$LOG/children.txt" 2>> "$LOG/children.err" || {
  echo "$(date -Is) children do not partition the suite, refusing"; exit 5; }
old_oracles "$DS"
step check check --dataset "$DS" --dim "$(dim "$DS")"
step oracle oracle --dataset "$DS" --suite "$SUITE" --dim "$(dim "$DS")"
n=0
while IFS= read -r ch; do
  n=$((n + 1))
  [ -e "$LOG/uploaded-$n${SKIP_PERF:+-quality}" ] && continue
  [ -e "/scratch/v25/$LEG.stop" ] && { echo "$(date -Is) stop file, ending before child $n"; finish; exit 0; }
  t0=$(date +%s)
  stream "$DS" "$SUITE" "$ch"
  upload "$GRP$n${SKIP_PERF:+-quality}"
  touch "$LOG/uploaded-$n${SKIP_PERF:+-quality}"
  msg="$LEG child $n ($ch) done on pod d GPU $GPU in $(( $(date +%s) - t0 ))s at $cv; Hub $GRP$n${SKIP_PERF:+-quality} (cumulative): $UP"
  note pod-d "$msg"; note control "$msg"
  herdr agent prompt controller "d-run driver: $msg" > /dev/null 2>&1 < /dev/null
done < "$LOG/children.txt"
[ -n "${SKIP_PERF:-}" ] && { finish; exit 0; }
left=$($PY "$PENDING" "$R" "$cv" "$DS" "$SUITE" 2>/dev/null | tail -1)
echo "$(date -Is) $left"
grep -q ": 0 to run" <<< "$left" || { note pod-d "$LEG: cells still pending after the last child ($left); driver stopped"; exit 6; }
upload "$LEGUP"
msg="$LEG leg done on pod d GPU $GPU at $cv ($left); Hub $LEGUP: $UP"
note pod-d "$msg"; note control "$msg"
herdr agent prompt controller "d-run driver: $msg" > /dev/null 2>&1 < /dev/null
finish
