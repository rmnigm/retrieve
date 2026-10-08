#!/usr/bin/env bash
# Stage one (suite, dataset) slice of the shared campaign-v2 tree and publish it as
# campaign-v2/<dataset>-<suite> (`bench upload --verify`; the MANIFEST sha256 goes in hub-index).
# Usage: stage-upload.sh LEG SUITE DATASET DIM
set -eu
LEG=$1 SUITE=$2 DS=$3 DIM=$4
R=/scratch/campaign-v2/results
S=/scratch/$LEG/hub/$DS-$SUITE
rm -rf "$S"; mkdir -p "$S/$SUITE"
cp -a "$R/$SUITE/$DS-d$DIM".* "$S/$SUITE/"   # .jsonl, .samples.jsonl, .perquery/
cd /workspace/retrieve/evaluation
/venvs/retrieve/bin/python -m bench.cli upload --results "$S" --path-in-repo "campaign-v2/$DS-$SUITE" --verify \
  2>&1 | tee "/scratch/$LEG/upload-$DS-$SUITE.log"
