#!/usr/bin/env bash
# Leg 2, goodreads reruns at campaign-v2.1 (the tag note's stale list): SilverTorch arms of `filter`
# and goodreads-synth `synth` whole at IVF-TUNE's n_lists; V2 and V3 Triton whole on both (Fix A);
# goodreads `bloomwidth-timed` whole. triton + official share one stream (_parity, jaccard_vs_first).
# V2 runs alone unless the tag note says to pair it with V1 again: V2="linr_v1_filter_mask linr_v2".
LEG=gr-reruns
V2=${V2:-linr_v2}
. "$(dirname "$(readlink -f "$0")")/common.sh"
step oracle-filter oracle --dataset goodreads --suite filter
step oracle-synth oracle --dataset goodreads-synth --suite synth
step oracle-bloomwidth-timed oracle --dataset goodreads --suite bloomwidth-timed
for ds_suite in "goodreads filter" "goodreads-synth synth"; do
  set -- $ds_suite
  stream "$1" "$2" "silvertorch|triton official"
  stream "$1" "$2" "$V2|triton"
  stream "$1" "$2" "linr_v3|triton"
done
stream goodreads filter "silvertorch|torch"
step campaign-bloomwidth-timed campaign --suite bloomwidth-timed --dataset goodreads --resume --interleave --out "$R"
finish
