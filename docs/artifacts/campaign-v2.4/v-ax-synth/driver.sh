#!/usr/bin/env bash
# V-AX-SYNTH whole (arxiv-synth synth d128, 534 cells, 5 children) at campaign-v2.4 on pod d GPU 0:
#   mkdir -p /scratch/v24 && setsid nohup flock -n /scratch/gpu0.lock bash driver.sh > /scratch/v24/v-ax-synth.driver.log 2>&1 &
export GPU=${GPU:-0} LEG=v-ax-synth DS=arxiv-synth SUITE=synth GRP=artifacts/v-ax-synth-v24-group LEGUP=campaign-v2.4/arxiv-synth-synth
exec bash "$(dirname "$(readlink -f "$0")")/../pod-d/leg.sh"
