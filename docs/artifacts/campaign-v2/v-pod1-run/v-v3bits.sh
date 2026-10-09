#!/usr/bin/env bash
# V-V3BITS at campaign-v2.2 (roadmap): the `v3bits` suite (V3 triton, k_bits {64, 128} x pool frac {0.01, 0.05}) on
# goodreads-synth and goodreads. Quality first: a --skip-perf pass into a scratch tree and its recall table (the leg
# goes on to the timed cells unless a cell fails), then the whole suite timed into the campaign tree.
LEG=v-v3bits
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles goodreads goodreads-synth
step oracle-goodreads oracle --dataset goodreads --suite v3bits
step oracle-synth oracle --dataset goodreads-synth --suite v3bits
Q=/scratch/$TV/v-v3bits/quality
step quality campaign --suite v3bits --dataset goodreads-synth --dataset goodreads --resume --skip-perf --out "$Q"
$PY "$HERE/v3bits_quality.py" "$Q" | tee "$LOG/summary-quality.txt"
step timed campaign --suite v3bits --dataset goodreads-synth --dataset goodreads --resume --interleave --out "$R"
finish
