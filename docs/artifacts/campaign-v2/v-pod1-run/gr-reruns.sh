#!/usr/bin/env bash
# V-RERUN-V21, goodreads reruns at campaign-v2.1 (the tag note's stale list): SilverTorch arms of `filter`
# and goodreads-synth `synth` whole at IVF-TUNE's n_lists; V2 and V3 Triton whole on both (Fix A).
# triton + official share one stream (_parity, jaccard_vs_first);
# V2 reruns with V1 in their interleave group (C1's paired V2/V1 ratio needs both at one code_version);
# V3 has no group and runs alone.
LEG=gr-reruns
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles goodreads goodreads-synth
step oracle-filter oracle --dataset goodreads --suite filter
step oracle-synth oracle --dataset goodreads-synth --suite synth
for ds_suite in "goodreads filter" "goodreads-synth synth"; do
  set -- $ds_suite
  stream "$1" "$2" "silvertorch|triton official"
  stream "$1" "$2" "linr_v1_filter_mask linr_v2|triton"
  stream "$1" "$2" "linr_v3|triton"
done
stream goodreads filter "silvertorch|torch"
finish
