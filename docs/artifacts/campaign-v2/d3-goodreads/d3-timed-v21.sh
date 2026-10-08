#!/usr/bin/env bash
# D3 goodreads `bloomwidth-timed` (bs 16, k 100, one timed point per width) at campaign-v2.1
# (code_version f01255f1), GPU 0, into the fresh v2.1 results tree; uploads as
# campaign-v2.1/goodreads-bloomwidth-timed. No interleave groups in the suite.
# Launch: setsid nohup flock /scratch/gpu0.lock bash d3-timed-v21.sh > /scratch/d3-timed-v21/driver.log 2>&1 &
LEG=d3-timed-v21
EXPECT=f01255f106214ec0f540352d0e2a5cd90b84b6a1
R=/scratch/campaign-v21/results
. "$(dirname "$(readlink -f "$0")")/../h2h-final/common.sh"
step oracle-goodreads oracle --dataset goodreads --suite bloomwidth-timed
step bloomwidth-timed campaign --suite bloomwidth-timed --dataset goodreads --resume --out "$R"
echo "$(date -Is) driver done"
