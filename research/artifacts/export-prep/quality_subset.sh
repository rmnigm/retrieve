#!/usr/bin/env bash
# The records-don't-move GPU check of the export branch: a quality-only subset of the FR cells rerun on the merged tree
# (dev/export-harness = harness half on top of the library half), compared byte for byte against the FR records with
# compare_quality.py. Adapted SilverTorch none / exact / bloom and LiNR exact cells, seed 0, no timing.
#
# usage: TREE=/scratch/wt/export-harness CFG=/scratch/fr/config OUT=/scratch/export-prep/quality-subset quality_subset.sh
#   env  VENV (default /venvs/retrieve; its silvertorch_fork extension must be built from TREE)  LOCK (default /scratch/gpu0.lock)
#        RETRIEVE_DATA_ROOT (default /data); TREE/evaluation/data must point at the data root (the FR configs use data/<dataset>/...)
set -eu
TREE=${TREE:?} CFG=${CFG:?} OUT=${OUT:?} VENV=${VENV:-/venvs/retrieve} LOCK=${LOCK:-/scratch/gpu0.lock}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} RETRIEVE_DATA_ROOT=${RETRIEVE_DATA_ROOT:-/data}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/scratch/inductor/export-prep} HF_HOME=${HF_HOME:-/scratch/hf}
export PYTHONPATH="$TREE/retrieve/src:$TREE/evaluation"
cd "$TREE/evaluation"
run() { flock "$LOCK" "$VENV/bin/python" -m bench.cli run --config-dir "$CFG" --out "$OUT" --seed 0 --skip-perf --mode eager "$@"; }
# adapted SilverTorch, exact and bloom, the whole quality curve leg of goodreads + arXiv (84 cells, ~3 min in the FR)
run --suite fr-curve-d128 --dataset goodreads --dim 128
run --suite fr-curve-d128 --dataset arxiv --dim 128
# adapted SilverTorch unfiltered + exact LiNR V1 / V3 on Yambda (quality leg)
run --suite fr-yambda --dataset yambda-500m --dim 128
# exact LiNR V1 / V2 / V3 on the real filters (the timed final-pass suite, quality only -> partial records)
run --suite filter --dataset goodreads --dim 128 --algo linr_v1_filter_mask --algo linr_v2 --algo linr_v3
run --suite filter --dataset arxiv --dim 128 --algo linr_v1_filter_mask --algo linr_v2 --algo linr_v3
"$VENV/bin/python" "$TREE/research/artifacts/export-prep/compare_quality.py" /scratch/results-draft/fr "$OUT"
