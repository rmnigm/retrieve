#!/usr/bin/env bash
# V-YFCC synth (yfcc10m-synth synth d192, 25 cells, 2 children: V1 + V2 triton clause at p {0.01, 0.1, 0.2, 0.5, 1}, SilverTorch triton
# clause n_probe {24, 256, 1024}; seed 0, k 100) at campaign-v2.5 on pod d GPU 0 (controller 2026-10-10 16:00; C1 and C6 at 10 M):
#   mkdir -p /scratch/v25 && setsid nohup flock -n /scratch/gpu0.lock bash driver.sh > /scratch/v25/v-yfcc-synth.driver.log 2>&1 &
export GPU=${GPU:-0} LEG=v-yfcc-synth DS=yfcc10m-synth SUITE=synth GRP=artifacts/v-yfcc-synth-v25-group LEGUP=campaign-v2.5/yfcc10m-synth-synth
exec bash "$(dirname "$(readlink -f "$0")")/../pod-d/leg.sh"
