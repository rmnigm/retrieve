#!/bin/bash
# Run a: the head-to-head SilverTorch points (gen_a.py) on current code, one dataset at a time on one GPU.
# Run from a checkout's evaluation/ (evaluation/data -> the data root); CFG holds the tree's dataset yamls
# plus gen_a.py's suites as suites.yaml.
set -uo pipefail
PY=${PY:-/venvs/retrieve/bin/python}
CFG=${CFG:-/scratch/gpu-followups/a/config}
OUT=${OUT:-/scratch/gpu-followups/a/out}
export HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/gf-a CUDA_VISIBLE_DEVICES=0
mkdir -p "$OUT"
for ds in goodreads-synth arxiv-synth pubmed-synth; do
  [ -f "data/$($PY -c "import yaml;print(yaml.safe_load(open('$CFG/$ds.yaml'))['data_dir'].split('/',1)[1])")/synth_filter.json" ] \
    || $PY -m eval_datasets.cli synth-filter --dataset $ds --draw 0
done
nvidia-smi -q -d CLOCK | head -25
for pair in ${CELLS:-goodreads:gf-a-d128 goodreads-synth:gf-a-d128 arxiv:gf-a-d128 arxiv-synth:gf-a-d128 \
            pubmed:gf-a-d768 pubmed-synth:gf-a-d768 laion1m:gf-a-d256 laion3m:gf-a-d256 laion10m:gf-a-d256 \
            laion30m:gf-a-d256 laion30m-synth:gf-a-d256}; do
  ds=${pair%%:*}; suite=${pair##*:}
  echo "=== $ds $suite $(date -u +%FT%TZ)"
  $PY -m bench.cli run --dataset "$ds" --suite "$suite" --config-dir "$CFG" --out "$OUT" || echo "FAILED $ds rc=$?"
done
nvidia-smi -q -d CLOCK | head -25
echo RUN-A-DONE
