#!/bin/bash
# One build point, final-freeze (clause-base) vs CLAUSE-OPT (clause-opt): bit-exact dumps of the
# exact filter and the rechecked bloom (bs 1 / 16 / picks', k 100 / 1000), compare, then the
# exact-vs-bloom profile per tree. Run under the GPU lock:
#   flock /scratch/gpu0.lock point.sh TAG DATASET DIM N_LISTS "N_PROBES" "SWEEPS" [BS]
set -u
TAG=$1 DS=$2 DIM=$3 NL=$4 NP=$5 SW=$6 BS=${7:-"1 16 64"}
A=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/scratch/clause-opt/out}
mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} TORCHINDUCTOR_CACHE_DIR=/scratch/clause-opt/inductor
COMMON="--dataset $DS --dim $DIM --n-lists $NL --n-probe $NP --sweeps $SW --bs $BS"
for t in base opt; do
  (cd /scratch/wt/clause-$t/evaluation && /venvs/clause-$t/bin/python $A/clause_bench.py $COMMON \
    --k 100 1000 --mode dump --out $OUT/$TAG-$t.pt) > $OUT/$TAG-dump-$t.log 2>&1 || { tail -n 20 $OUT/$TAG-dump-$t.log; exit 1; }
done
/venvs/clause-opt/bin/python $A/compare.py $OUT/$TAG-base.pt $OUT/$TAG-opt.pt | tail -n 4
for t in ${PROFILE:-base opt}; do
  (cd /scratch/wt/clause-$t/evaluation && /venvs/clause-$t/bin/python $A/clause_bench.py $COMMON \
    --k 100 --mode profile --json $OUT/$TAG-prof-$t.json) > $OUT/$TAG-prof-$t.log 2>&1 || tail -n 20 $OUT/$TAG-prof-$t.log
done
echo "$TAG done $(date -u +%H:%M:%S)"
