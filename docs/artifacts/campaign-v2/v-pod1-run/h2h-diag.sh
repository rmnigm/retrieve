#!/usr/bin/env bash
# H2H protocol diagnostic (controller addendum 2026-10-09): goodreads c0_genre bloom bs16 k100 seed 0,
# the h2h arms (triton, official fp16, official int32) timed (a) interleaved + --profile, (b)
# interleaved, (c) one process per arm; the sequence three times in orders abc, cba, bac. Scratch
# trees /scratch/h2h-diag/<protocol>/r<rep>, not a campaign leg.
LEG=h2h-diag
R=/scratch/h2h-diag
. "$(dirname "$(readlink -f "$0")")/common.sh"
C=$R/config
$PY "$HERE/h2h_diag_config.py" "$C" || exit 1
step oracle oracle --config-dir "$C/all" --dataset goodreads --suite h2h
run=(run --dataset goodreads --dim 128 --suite h2h --resume)
a() { step "a-r$1" "${run[@]}" --config-dir "$C/all" --out "$R/a/r$1" --interleave --profile; }
b() { step "b-r$1" "${run[@]}" --config-dir "$C/all" --out "$R/b/r$1" --interleave; }
c() { for arm in triton fp16 int32; do step "c-$arm-r$1" "${run[@]}" --config-dir "$C/$arm" --out "$R/c/r$1"; done; }
rep=0
for order in "a b c" "c b a" "b a c"; do
  rep=$((rep + 1))
  for p in $order; do $p $rep; done
done
$PY "$HERE/h2h_diag_summary.py" "$R" | tee "$LOG/summary.txt"
finish
