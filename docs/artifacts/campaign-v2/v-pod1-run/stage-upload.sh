#!/usr/bin/env bash
# Stage one (suite, dataset) slice of the v2.1 tree plus the leg's logs and publish it as
# campaign-v2.1/<dataset>-<suite> (`bench upload --verify`; the MANIFEST sha256 goes in hub-index).
# Usage: stage-upload.sh LEG SUITE DATASET DIM
set -eu
LEG=$1 SUITE=$2 DS=$3 DIM=$4
R=${R:-/scratch/campaign-v21/results}
S=/scratch/v21/$LEG/hub/$DS-$SUITE
rm -rf "$S"; mkdir -p "$S/$SUITE" "$S/logs"
cp -a "$R/$SUITE/$DS-d$DIM".* "$S/$SUITE/"   # .jsonl, .samples.jsonl, .perquery/
cp -a /scratch/v21/$LEG/*.log /scratch/v21/$LEG/clocks* /scratch/v21/$LEG/summary* "$S/logs/"
cp -a "$R/_logs/"*"${SUITE}_$DS-d$DIM"*.log "$R/_logs/campaign.log" "$S/logs/" 2>/dev/null || true
cd /workspace/retrieve/evaluation
/venvs/retrieve/bin/python -m bench.cli upload --results "$S" --path-in-repo "campaign-v2.1/$DS-$SUITE" --verify \
  2>&1 | tee "/scratch/v21/$LEG/upload-$DS-$SUITE.log"
