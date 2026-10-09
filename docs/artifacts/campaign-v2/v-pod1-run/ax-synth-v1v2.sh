#!/usr/bin/env bash
# arXiv-synth V1 + V2 on pod 1 (controller 2026-10-10 220000000): the synth suite's 7 arXiv-synth rates (14 cells, V1 + V2 triton clause interleaved,
# bs 1 + 16, k 100, seed 0) at the tag in common.sh. C1's 3 M crossover on the box of goodreads / YFCC deep / the crosstree.
LEG=ax-synth-v1v2
. "$(dirname "$(readlink -f "$0")")/common.sh"
SW="--sweep p0001 --sweep p001 --sweep p005 --sweep p01 --sweep p02 --sweep p05 --sweep p1"
old_oracles arxiv-synth
step oracle oracle --dataset arxiv-synth --suite synth $SW
step run run --dataset arxiv-synth --dim 128 --suite synth $SW --algo linr_v1_filter_mask --algo linr_v2 --backend triton \
  --filter-kind clause --seed 0 --k 100 --resume --interleave --out "$R"
finish
