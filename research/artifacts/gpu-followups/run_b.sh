#!/bin/bash
# Run b: the head-to-head LiNR arms (gen_b.py) timed again at seeds 1 and 2, interleaved by algo, one dataset at a time.
set -uo pipefail
PY=${PY:-/venvs/retrieve/bin/python}
CFG=${CFG:-/scratch/gpu-followups/b/config}
OUT=${OUT:-/scratch/gpu-followups/b/out}
export HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/gf-b CUDA_VISIBLE_DEVICES=0
cd /scratch/wt/laion-inputs/evaluation
mkdir -p "$OUT"
nvidia-smi -q -d CLOCK | head -25
for pair in goodreads:gf-b-d128 arxiv:gf-b-d128 pubmed:gf-b-d768 laion1m:gf-b-d256 laion3m:gf-b-d256 \
            laion10m:gf-b-d256 laion30m:gf-b-d256; do
  ds=${pair%%:*}; suite=${pair##*:}
  for seed in 1 2; do
    echo "=== $ds $suite seed $seed $(date -u +%FT%TZ)"
    $PY -m bench.cli run --dataset "$ds" --suite "$suite" --seed $seed --interleave --config-dir "$CFG" --out "$OUT" \
      || echo "FAILED $ds seed $seed rc=$?"
  done
done
nvidia-smi -q -d CLOCK | head -25
echo RUN-B-DONE
