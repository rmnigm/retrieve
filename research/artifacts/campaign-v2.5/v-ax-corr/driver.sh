#!/usr/bin/env bash
# V-AX-CORR (arxiv-corr-synth synth d128, 198 cells, 4 children, quality + timed in one pass) at campaign-v2.5 on pod d GPU 1:
#   mkdir -p /scratch/v25 && setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v25/v-ax-corr.driver.log 2>&1 &
export GPU=${GPU:-1} LEG=v-ax-corr DS=arxiv-corr-synth SUITE=synth GRP=artifacts/v-ax-corr-v25-group LEGUP=campaign-v2.5/arxiv-corr-synth-synth
exec bash "$(dirname "$(readlink -f "$0")")/../pod-d/leg.sh"
