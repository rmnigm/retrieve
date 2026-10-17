#!/bin/bash
# Run d on one GPU: run_d.py over Goodreads, then PubMed.
cd /scratch/wt/laion-inputs/evaluation
export HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/gf-d CUDA_VISIBLE_DEVICES=0 GF_D_OUT=/scratch/gpu-followups/d/out
nvidia-smi -q -d CLOCK | head -25
/venvs/retrieve/bin/python ../research/artifacts/gpu-followups/run_d.py goodreads pubmed || echo "FAILED rc=$?"
nvidia-smi -q -d CLOCK | head -25
echo RUN-D-DONE
