#!/bin/bash
# The LAION 1M-30M inputs, rebuilt as the final run built them (docs/system/datasets.md § laion30m,
# § Subsets, § Synthetic selectivity attrs). One GPU. Run from a checkout's evaluation/ with
# evaluation/data -> $RETRIEVE_DATA_ROOT. Configs are written to $CFG and diffed against the tree's.
set -euo pipefail
PY=${PY:-/venvs/retrieve/bin/python}
CFG=${CFG:-/scratch/laion/configs}
export HF_HOME=${HF_HOME:-/scratch/hf}
mkdir -p "$CFG"
nvidia-smi -q -d CLOCK | head -25
$PY -m eval_datasets.etl.laion download --parts 0-1
$PY -m eval_datasets.etl.laion prep
$PY -m eval_datasets.etl.laion encode_text
$PY -m eval_datasets.etl.laion encode_queries
$PY -m eval_datasets.etl.laion ingest --config-dir "$CFG"
for d in 0 1 2; do $PY -m eval_datasets.cli synth-filter --dataset laion30m-synth --draw $d; done
for n in 1 3 10; do $PY -m eval_datasets.cli subset laion30m laion${n}m --n-items ${n}000000 --config-dir "$CFG"; done
for f in "$CFG"/*.yaml; do diff -q "$f" "config/$(basename "$f")"; done
for f in config/laion*.yaml; do $PY -m bench.cli check --dataset "$(basename "$f" .yaml)"; done
echo REBUILD-OK
