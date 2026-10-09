#!/usr/bin/env bash
# H-PROFILE + H2H-FINAL at campaign-v2.1: the whole `h2h` suite (official fp16 + int32 + triton, one
# interleaved group per cell, seeds 0-4) on goodreads and arXiv. Timing runs without --profile into the
# campaign tree; the kernel-only split is a separate --profile pass into $P (same record keys, so its
# own tree), after the H2H protocol diagnostic showed --profile moving interleaved eager times.
LEG=h2h
P=/scratch/campaign-v21/h2h-profile
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles goodreads arxiv
step oracle-goodreads oracle --dataset goodreads --suite h2h
step oracle-arxiv oracle --dataset arxiv --suite h2h
step campaign campaign --suite h2h --dataset goodreads --dataset arxiv --resume --interleave --out "$R"
step campaign-profile campaign --suite h2h --dataset goodreads --dataset arxiv --resume --interleave --profile --out "$P"
finish
