#!/usr/bin/env bash
# H2H-FINAL driver: the `h2h` suite (official fp16 + int32 + triton, one interleaved group per cell,
# seeds 0-4, --profile) on goodreads E1c and arXiv at campaign-v2, GPU 0, one sequential process group.
# Launch: setsid nohup bash h2h-final.sh > /scratch/h2h-final/driver.log 2>&1 &
LEG=h2h-final
. "$(dirname "$(readlink -f "$0")")/common.sh"
step oracle-goodreads oracle --dataset goodreads --suite h2h
step oracle-arxiv oracle --dataset arxiv --suite h2h
step campaign campaign --suite h2h --dataset goodreads --dataset arxiv --resume --interleave --profile --out "$R"
echo "$(date -Is) driver done"
