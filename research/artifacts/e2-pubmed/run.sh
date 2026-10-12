#!/usr/bin/env bash
# Roadmap E2: the PubMed + MedCPT 10 M slice as staged on 2026-09-26 (A100 box).
# cwd evaluation/, RETRIEVE_DATA_ROOT=/data, evaluation/data -> /data.
set -euo pipefail
A=../docs/artifacts/e2-pubmed
P="uv run --no-sync python -m eval_datasets.etl.pubmed"
$P plan --keep-items 10000000 --medline --report $A/plan-10m.json
$P download --what pmids
$P download --what mesh
$P convert --keep-items 10000000 --seed 0 --fetch --prefetch 1 --delete-raw > $A/convert.log 2>&1 &
$P medline --stream --delete-raw --workers 16 > $A/medline.log 2>&1 &
wait
$P attrs --mesh-desc /data/_raw/pubmed/mesh_desc2026.xml.gz > $A/attrs.log 2>&1
$P queries > $A/queries.log 2>&1
$P encode_queries > $A/encode_queries.log 2>&1
uv run --no-sync python -m bench.cli check --dataset pubmed > $A/bench-check.log 2>&1
cell() {  # $1 algo, $2 backend
  TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/e2 uv run --no-sync python -m bench.cli run \
    --dataset pubmed --dim 768 --suite filter --algo "$1" --backend "$2" \
    --filter-kind clause --sweep c0_mesh --mode eager --skip-perf \
    --output $A/filter-pubmed-d768-cell.jsonl
}
cell silvertorch triton > $A/cell-silvertorch-triton.log 2>&1 || true    # OOM in quantize_int8_global
cell linr_v1_filter_mask triton > $A/cell-linr_v1.log 2>&1
cell silvertorch official > $A/cell-silvertorch-official.log 2>&1 || true  # same OOM
