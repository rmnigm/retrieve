#!/usr/bin/env bash
# V-V3BITS at campaign-v2.2 (roadmap): the `v3bits` suite (V3 triton, k_bits {64, 128} x pool frac {0.01, 0.05}) on
# goodreads-synth and goodreads, timed into the campaign tree (its records carry the quality too). QUALITY=1 adds a
# --skip-perf pass into a scratch tree first; it gates nothing and V3's quality is not reused across trees, so it is
# off (orchestrator, 2026-10-09: the first run's pass was stopped at 128 / 168 goodreads-synth cells).
LEG=v-v3bits
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles goodreads goodreads-synth
step oracle-goodreads oracle --dataset goodreads --suite v3bits
step oracle-synth oracle --dataset goodreads-synth --suite v3bits
if [ "${QUALITY:-0}" = 1 ]; then
  Q=/scratch/$TV/v-v3bits/quality
  step quality campaign --suite v3bits --dataset goodreads-synth --dataset goodreads --resume --skip-perf --out "$Q"
  $PY "$HERE/v3bits_quality.py" "$Q" | tee "$LOG/summary-quality.txt"
fi
step timed campaign --suite v3bits --dataset goodreads-synth --dataset goodreads --resume --interleave --out "$R"
finish
