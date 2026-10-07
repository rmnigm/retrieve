#!/usr/bin/env bash
# Pubmed 10 M slice restage for D1-E (w-h2): docs/artifacts/e2-pubmed/run.sh's ETL steps, CPU
# only, niced on NUMA node 1, at most 4 NCBI connections (16-way bursts draw HTTP 503).
# Stops before encode_queries (GPU). cwd evaluation/, RETRIEVE_DATA_ROOT=/data.
set -euo pipefail
L=/scratch/h2-pubmed/logs; mkdir -p $L
P="nice -n 19 taskset -c 64-127,192-255 /venvs/h2/bin/python -m eval_datasets.etl.pubmed"
date -u +%FT%TZ > $L/start
$P download --what pmids --parallel 2 > $L/download-pmids.log 2>&1
$P download --what mesh --parallel 2 > $L/download-mesh.log 2>&1
$P convert --keep-items 10000000 --seed 0 --fetch --prefetch 1 --delete-raw > $L/convert.log 2>&1 &
c=$!
$P medline --stream --delete-raw --workers 2 > $L/medline.log 2>&1 &
m=$!
wait $c; wait $m
$P attrs --mesh-desc /data/_raw/pubmed/mesh_desc2026.xml.gz > $L/attrs.log 2>&1
$P queries > $L/queries.log 2>&1
date -u +%FT%TZ > $L/end
