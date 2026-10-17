#!/usr/bin/env bash
# Check a rebuilt LAION subset against pod g's manifest. usage: check_subset.sh laion1m
set -u
cd /data/$1 && sha256sum -c --quiet /data/laion30m/subsets/$1.MANIFEST.sha256 2>&1 | sed "s/^/$1: /"
echo "$1: $(wc -l < /data/laion30m/subsets/$1.MANIFEST.sha256) files listed, exit ${PIPESTATUS[0]}"
