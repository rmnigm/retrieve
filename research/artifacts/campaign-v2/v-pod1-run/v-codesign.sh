#!/usr/bin/env bash
# V-CODESIGN rerun at campaign-v2.1, goodreads half (pod c runs arXiv): the whole `codesign` suite
# on goodreads, partial vs full interleaved (the 408b1188 run is stale: quantize fix).
LEG=v-codesign
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles goodreads
step oracle-goodreads oracle --dataset goodreads --suite codesign
step campaign campaign --suite codesign --dataset goodreads --resume --interleave --out "$R"
finish
