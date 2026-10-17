#!/bin/bash
# GPU check of the rebuilt LAION inputs: exact LiNR V1 (bs 16, k 100) on both real filters at 1M-30M; its oracle
# recall, held-out counts and result-id sha256 are compared with the final run's records (compare_exact.py).
cd /scratch/wt/laion-inputs/evaluation
export HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/laion-check CUDA_VISIBLE_DEVICES=0
nvidia-smi -q -d CLOCK | head -25
for ds in laion1m laion3m laion10m laion30m; do
  /venvs/retrieve/bin/python -m bench.cli run --dataset $ds --suite lc-exact --mode graph \
    --config-dir /scratch/laion/check/config --out /scratch/laion/check/exact || echo "FAILED $ds"
done
echo EXACT-DONE
