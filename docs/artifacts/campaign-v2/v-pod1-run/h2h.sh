#!/usr/bin/env bash
# H-PROFILE + H2H-FINAL at campaign-v2.1: the whole `h2h` suite (official fp16 + int32 + triton, one
# interleaved group per cell, seeds 0-4, --profile with H-PROFILE's fix) on goodreads and arXiv.
LEG=h2h
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles goodreads arxiv
step oracle-goodreads oracle --dataset goodreads --suite h2h
step oracle-arxiv oracle --dataset arxiv --suite h2h
step campaign campaign --suite h2h --dataset goodreads --dataset arxiv --resume --interleave --profile --out "$R"
finish
