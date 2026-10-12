#!/usr/bin/env bash
# self-match GPU smoke on the rerun tree (a9b8951 + dev/selfmatch harness, library eb600da1, /venvs/meta-fork fork build):
# arXiv d128 c0_maincat, every arm at k 1000 (oracle 1001), fork + V2 at k 2048 (oracle 2049); quality only. Run under flock /scratch/gpu0.lock.
set -u
FT=/scratch/wt/selfmatch-run V=/venvs/meta-fork O=/scratch/selfmatch-smoke
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/selfmatch-smoke/inductor RETRIEVE_DATA_ROOT=/data
export PYTHONPATH="$FT/retrieve/src:$FT/retrieve:$FT/evaluation"
cd $FT/evaluation
for s in sm-smoke sm-smoke-k2048; do
  t0=$(date +%s)
  timeout 540 $V/bin/python -m bench.cli run --config-dir $O/config --out $O/out --suite $s --dataset arxiv --skip-perf
  echo "suite $s exit $? s=$(( $(date +%s) - t0 ))"
done
