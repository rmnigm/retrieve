#!/usr/bin/env bash
# goodreads-synth's SYNTH-TRIM middle (controller 2026-10-10): p 0.05 / 0.2 / 0.5, V1 + V2 interleaved and SilverTorch triton clause
# (n_lists 4096, n_probe {24 ... 1024}), seed 0: 24 cells, at the tag in common.sh. C1's crossover and C6 at 0.8 M.
LEG=gr-synth-mid
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles goodreads-synth
step oracle oracle --dataset goodreads-synth --suite synth --sweep p005 --sweep p02 --sweep p05
step run run --dataset goodreads-synth --dim 128 --suite synth --sweep p005 --sweep p02 --sweep p05 --resume --interleave \
  --out "$R"
finish
