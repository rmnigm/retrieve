#!/usr/bin/env bash
# Leg 5, V-SEEDS YFCC half at campaign-v2.1: yfcc10m `filter`, every arm, seeds 0-2, with the n95
# column from IVF-TUNE.
LEG=v-seeds-yfcc
. "$(dirname "$(readlink -f "$0")")/common.sh"
step oracle-filter oracle --dataset yfcc10m --suite filter
step campaign-filter campaign --suite filter --dataset yfcc10m --resume --interleave --out "$R"
finish
