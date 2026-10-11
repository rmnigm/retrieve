#!/usr/bin/env bash
# AFTER-QUEUE 1-2 on pod d GPU 0, under one lock: (a) synth-st.sh, then (b) probe.sh. Launch:
#   GPU=0 setsid nohup flock -n /scratch/gpu0.lock bash after.sh > /scratch/laion30m/v29/after.driver.log 2>&1 &
D=$(dirname "$(readlink -f "$0")")
bash "$D/synth-st.sh" && bash "$D/probe.sh"
