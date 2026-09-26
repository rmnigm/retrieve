#!/usr/bin/env bash
# Roadmap E3: the OpenAlex 15 M catalog as staged on 2026-09-26 (A100 box).
# cwd evaluation/, RETRIEVE_DATA_ROOT=/data, evaluation/data -> /data.
set -euo pipefail
A=../docs/artifacts/e3-openalex
O="python -m eval_datasets.etl.openalex"
$O plan --sample-row-groups 64 --report $A/plan-2026-09-23.json > $A/plan-2026-09-23.log 2>&1
$O download
$O convert --sample-rate 0.35 --workers 64 > $A/run15m-convert.log 2>&1
$O prep --keep-items 15000000 > $A/run15m-prep.log 2>&1
python $A/verify_prep.py /data/openalex > $A/run15m-verify.log 2>&1
$O encode_text > $A/run15m-encode.log 2>&1
$O encode_queries >> $A/run15m-encode.log 2>&1
$O attrs >> $A/run15m-encode.log 2>&1
python -m bench.cli check --dataset openalex > $A/run15m-bench-check.log 2>&1
# The cell needs openalex in suites.yaml's filter suite (added for the run, reverted after).
# First attempt: OOM in layout.load_text_items (cell-linr_v1-oom.log), fixed by the per-shard
# normalise. Second: OOM in bench/oracle.py's item_embs.t().contiguous() (cell-linr_v1.log).
TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/e3 python -m bench.cli run \
  --dataset openalex --dim 768 --suite filter --algo linr_v1_filter_mask --backend triton \
  --filter-kind clause --sweep field_era --mode eager --skip-perf \
  --output $A/filter-openalex-d768-cell.jsonl > $A/cell-linr_v1.log 2>&1
