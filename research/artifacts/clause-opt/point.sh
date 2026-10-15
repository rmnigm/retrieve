#!/bin/bash
# One build point, final-freeze (BASE) vs CLAUSE-OPT (OPT), one index build per tree: the
# bit-exact dumps (exact + rechecked bloom, k 100 / 1000) and the exact-vs-bloom profile (k 100,
# bloom recheck off), then compare. Run under the GPU lock:
#   flock /scratch/gpu0.lock point.sh TAG DATASET DIM N_LISTS "N_PROBES" "SWEEPS" ["BS"]
set -u
TAG=$1 DS=$2 DIM=$3 NL=$4 NP=$5 SW=$6 BS=${7:-"1 16 64"}
A=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/scratch/clause-opt/out}
BASE_TREE=${BASE_TREE:-/scratch/wt/clause-base} BASE_VENV=${BASE_VENV:-/venvs/clause-base}
OPT_TREE=${OPT_TREE:-/scratch/wt/clause-opt} OPT_VENV=${OPT_VENV:-/venvs/clause-opt}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} TORCHINDUCTOR_CACHE_DIR=/scratch/clause-opt/inductor
for t in base opt; do
  [ -s $OUT/$TAG-$t.pt ] && [ -s $OUT/$TAG-prof-$t.json ] && continue  # done in an earlier block
  if [ $t = base ]; then tree=$BASE_TREE venv=$BASE_VENV; else tree=$OPT_TREE venv=$OPT_VENV; fi
  (cd $tree/evaluation && $venv/bin/python $A/clause_bench.py --dataset $DS --dim $DIM \
    --n-lists $NL --n-probe $NP --sweeps $SW --bs $BS --k 100 1000 --mode both \
    --out $OUT/$TAG-$t.pt --json $OUT/$TAG-prof-$t.json) > $OUT/$TAG-$t.log 2>&1 \
    || { echo "$TAG $t FAILED"; tail -n 20 $OUT/$TAG-$t.log; exit 1; }
done
$OPT_VENV/bin/python $A/compare.py $OUT/$TAG-base.pt $OUT/$TAG-opt.pt | tail -n 4
echo "$TAG done $(date -u +%H:%M:%S)"
