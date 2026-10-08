#!/usr/bin/env bash
# Leg 3, V-GR-DEEP at campaign-v2.1: goodreads `deep` (n_lists and n_probe from suites.yaml after
# IVF-TUNE), every arm.
LEG=v-gr-deep
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles goodreads
step oracle-deep oracle --dataset goodreads --suite deep
step campaign-deep campaign --suite deep --dataset goodreads --resume --interleave --out "$R"
finish
