#!/usr/bin/env bash
# V-PUBMED on pod a100-x1-b: docs/artifacts/pubmed-restage/restage.sh with this pod's NUMA node 1
# (96-191; the GPU is on node 0) and the shared venv. cwd evaluation/, RETRIEVE_DATA_ROOT=/data.
set -euo pipefail
L=/scratch/v-pubmed/logs; mkdir -p $L
export PYTHONPATH=/scratch/wt/v-pubmed/evaluation:/scratch/wt/v-pubmed/retrieve/src
P="nice -n 19 taskset -c 96-191 /venvs/retrieve/bin/python -m eval_datasets.etl.pubmed"
date -u +%FT%TZ > $L/restage-start
$P download --what pmids --parallel 2 > $L/download-pmids.log 2>&1
$P download --what mesh --parallel 2 > $L/download-mesh.log 2>&1
$P convert --keep-items 10000000 --seed 0 --fetch --prefetch 1 --delete-raw > $L/convert.log 2>&1 &
c=$!
$P medline --stream --delete-raw --workers 2 > $L/medline.log 2>&1 &
m=$!
wait $c; wait $m
$P attrs --mesh-desc /data/_raw/pubmed/mesh_desc2026.xml.gz > $L/attrs.log 2>&1
$P queries > $L/queries.log 2>&1
date -u +%FT%TZ > $L/restage-end
