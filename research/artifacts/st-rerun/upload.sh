#!/usr/bin/env bash
# Publish one dataset's rerun tree (records, samples, per-query sidecars, run log under logs/: bench skips _ paths) to
# pinkmeme/eval-results under artifacts/st-rerun/<dataset>/ (upload.py). usage: upload.sh DATASET | fu-DATASET
set -eu
ds=$1
up=/scratch/strerun/upload/$ds
rm -rf "$up" && mkdir -p "$up/logs"
cp -r /scratch/strerun/results/strerun-$ds "$up/"
cp /scratch/strerun/logs/run-$ds.log /scratch/strerun/logs/run-strerun-$ds.log "$up/logs/" 2>/dev/null || true
cd "$(dirname "$0")/../../../evaluation"
uv run python ../research/artifacts/st-rerun/upload.py "$up" "$ds"
