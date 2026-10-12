#!/usr/bin/env bash
# YFCC synth, SilverTorch child only (controller 2026-10-10 17:00: V1 + V2 wait on the exact-gate ruling), into the leg's tree; uploads
# the tree as artifacts/v-yfcc-synth-v25-st (the leg upload waits for the exact child).
#   setsid nohup flock -n /scratch/gpu0.lock bash st.sh > /scratch/v25/v-yfcc-synth-st.driver.log 2>&1 &
export GPU=${GPU:-0} LEG=v-yfcc-synth DS=yfcc10m-synth
R=/scratch/campaign-v25/$DS-synth
. "$(dirname "$(readlink -f "$0")")/../pod-d/common.sh"
stream $DS synth "silvertorch|triton"
upload artifacts/v-yfcc-synth-v25-st
msg="v-yfcc-synth SilverTorch child (15 cells) done on pod d GPU $GPU at $cv; Hub artifacts/v-yfcc-synth-v25-st: $UP"
note pod-d "$msg"; note control "$msg"
finish
