#!/usr/bin/env bash
# Stage one (suite, dataset) slice of a tag's tree plus the leg's logs and publish it as
# <TAG>/<dataset>-<suite> (`bench upload --verify`; the MANIFEST sha256 goes in hub-index). TAG defaults to campaign-v2.5.
# Usage: stage-upload.sh LEG SUITE DATASET DIM
set -eu
LEG=$1 SUITE=$2 DS=$3 DIM=$4
TAG=${TAG:-campaign-v2.5}
TV=$(echo "${TAG#campaign-}" | tr -d .)
R=${R:-/scratch/campaign-$TV/results}
L=/scratch/$TV/$LEG
S=$L/hub/$DS-$SUITE
rm -rf "$S"; mkdir -p "$S/$SUITE" "$S/logs"
cp -a "$R/$SUITE/$DS-d$DIM".* "$S/$SUITE/"   # .jsonl, .samples.jsonl, .perquery/
cp -a "$L"/*.log "$L"/clocks* "$L"/summary* "$S/logs/"
cp -a "$R/_logs/"*"${SUITE}_$DS-d$DIM"*.log "$R/_logs/campaign.log" "$S/logs/" 2>/dev/null || true
cd /workspace/retrieve/evaluation
/venvs/retrieve/bin/python -m bench.cli upload --results "$S" --path-in-repo "$TAG/$DS-$SUITE" --verify \
  2>&1 | tee "$L/upload-$DS-$SUITE.log"
