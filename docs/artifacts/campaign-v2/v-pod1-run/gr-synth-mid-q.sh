#!/usr/bin/env bash
# goodreads-synth middle cells, quality only (controller 2026-10-10 234500000): p 0.05 / 0.2 / 0.5, V1 + V2 and SilverTorch triton clause,
# seed 0, `--skip-perf`, at the tag in common.sh. Nothing timed on clause kernels over 10-clause synth tables before v2.7 (CLAUSE-SKIP);
# the timed half runs then.
LEG=gr-synth-mid-q
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles goodreads-synth
step oracle oracle --dataset goodreads-synth --suite synth --sweep p005 --sweep p02 --sweep p05
step run run --dataset goodreads-synth --dim 128 --suite synth --sweep p005 --sweep p02 --sweep p05 --resume --skip-perf \
  --out "$R"
finish
