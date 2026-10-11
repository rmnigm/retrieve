#!/usr/bin/env bash
# D3 driver, goodreads share: `bloomwidth` (quality-only, perf: false) on goodreads at
# campaign-v2, GPU 0, one process group. arXiv and PubMed run on other pods, and
# `bloomwidth-timed` waits for campaign-v2.1 (controller, 2026-10-08). No interleave groups.
# Launch: setsid nohup flock /scratch/gpu0.lock bash d3.sh > /scratch/d3/driver.log 2>&1 &
LEG=d3
. "$(dirname "$(readlink -f "$0")")/../h2h-final/common.sh"
step oracle-goodreads oracle --dataset goodreads --suite bloomwidth
step bloomwidth campaign --suite bloomwidth --dataset goodreads --resume --out "$R"
echo "$(date -Is) driver done"
