#!/usr/bin/env bash
# Publish one finished leg: driver log, clock trace and nvidia-smi -q into the tree's _logs, then
# `bench upload --verify` under campaign.yaml's default perf `hub` pattern, or HUB if given (private, NOT CITABLE:
# no --gate). Usage: bash upload.sh arxiv-synth synth [HUB]
set -eu
TAG=$1-$2
R=${RROOT:-/scratch/campaign-v2.1}/results-$TAG
mkdir -p "$R/_logs/driver"
cp /scratch/v21/$TAG/{driver.log,clocks.csv,clock-q-start.txt,clock-q-end.txt} "$R/_logs/driver/" 2>/dev/null || true
cd ${REPO:-/scratch/wt/v-ax-synth}/evaluation
HUB=${3:-$(/venvs/retrieve/bin/python -c 'import sys,yaml; print(yaml.safe_load(open("campaign.yaml"))["default"]["perf"]["hub"].format(dataset=sys.argv[1], suite=sys.argv[2]))' "$1" "$2")}
echo "upload $R -> $HUB"
HF_HOME=/scratch/hf /venvs/retrieve/bin/python -m bench.cli upload --results "$R" --path-in-repo "$HUB" --verify
