#!/usr/bin/env bash
# V-PROF3 at campaign-v2.1 (roadmap): (a) official partial vs full codesign, (b) V1 bs 1 Triton vs
# torch.compile, (c) V1 bs 16 graph vs eager at p 0.001 and 1. Each profile in a fresh process (one
# profiler session per process: H-PROFILE saw device events vanish over sessions), then each pair timed
# interleaved in one process. Outputs /scratch/v-prof3/out; not a campaign leg.
LEG=v-prof3
R=/scratch/v-prof3
. "$(dirname "$(readlink -f "$0")")/../v-pod1-run/common.sh"
OUT=$R/out
mkdir -p "$OUT"
old_oracles goodreads-synth
step oracle-codesign oracle --dataset goodreads --suite codesign --sweep c0_genre
step oracle-synth oracle --dataset goodreads-synth --suite synth
p3() {
  local t0=$(date +%s)
  $PIN $PY "$HERE/../v-prof3/prof3.py" "$@" "$OUT" > "$LOG/prof3-${1}-${2}${3:+-$3}.log" 2>&1
  local rc=$?
  echo "$(date -Is) prof3 $* rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || exit $rc
}
for iv in "a partial" "a full" "b triton" "b compile" "c0001 eager" "c0001 graph" "c1 eager" "c1 graph"; do
  p3 profile $iv
done
for it in a b c0001 c1; do p3 time $it; done
finish
