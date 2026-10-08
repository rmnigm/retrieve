#!/usr/bin/env bash
# Leg 1, V-CODESIGN rerun at campaign-v2.1: the whole `codesign` suite on arXiv and goodreads,
# partial vs full interleaved (the 408b1188 run is stale: quantize fix).
LEG=v-codesign
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles arxiv goodreads
step oracle-arxiv oracle --dataset arxiv --suite codesign
step oracle-goodreads oracle --dataset goodreads --suite codesign
step campaign campaign --suite codesign --dataset arxiv --dataset goodreads --resume --interleave --out "$R"
finish
