#!/usr/bin/env bash
# C7 at 30 M, interleaved: ours (triton) and Meta (official, -O3 build in /venvs/d-run-v28) in one interleave group per cell, one process per
# sweep, --interleave --profile --mode eager, campaign-v2.8, pod d GPU 1. The suite lives in a scratch config dir (config/ + c7_suite.yaml).
#   GPU=1 setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v28/c7-laion30m.driver.log 2>&1 &
export TAG=campaign-v2.8 LEG=c7-laion30m REPO=/scratch/wt/v28-official PY=/venvs/d-run-v28/bin/python GPU=${GPU:-1}
DS=laion30m SUITE=c7-laion30m
R=/scratch/campaign-v28/$DS-$SUITE
CFG=/scratch/v28/c7-config
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
rm -rf "$CFG" && cp -r "$REPO/evaluation/config" "$CFG" && cat "$HERE/../../campaign-v2.8/c7-laion30m/c7_suite.yaml" >> "$CFG/suites.yaml"
step check check --dataset $DS --dim 256
for sw in c0_domain tags4; do NARROW="--config-dir $CFG --sweep $sw --mode eager --profile" stream $DS $SUITE "silvertorch|triton official"; done
upload artifacts/c7-laion30m-interleaved
msg="$LEG (8 cells, interleaved triton vs official -O3, --profile) done on pod d GPU $GPU at $cv; Hub artifacts/c7-laion30m-interleaved: $UP"
note pod-d "$msg"; note control "$msg"
finish
