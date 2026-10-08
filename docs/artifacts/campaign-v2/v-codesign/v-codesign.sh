#!/usr/bin/env bash
# V-CODESIGN driver: the `codesign` suite (official bloom, partial vs full interleaved, n_probe
# {8, 32, 128}, 3 sweeps, 3 seeds) on arXiv and goodreads at campaign-v2, GPU 0, one process group.
# Launch: setsid nohup bash v-codesign.sh > /scratch/v-codesign/driver.log 2>&1 &
LEG=v-codesign
. "$(dirname "$(readlink -f "$0")")/../h2h-final/common.sh"
step oracle-arxiv oracle --dataset arxiv --suite codesign
step oracle-goodreads oracle --dataset goodreads --suite codesign
step campaign campaign --suite codesign --dataset arxiv --dataset goodreads --resume --interleave --out "$R"
echo "$(date -Is) driver done"
