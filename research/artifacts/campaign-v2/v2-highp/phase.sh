#!/usr/bin/env bash
# V2-HIGHP GPU phase on pod b (GPU 0): SASS identity at D 128 / 192, the fmkt bit-exact gate and the keep-rule timing at
# 0.8 M / 3 M d128 and 10 M d768, the LiNR compile tests and the library suite, then V2 on the real PubMed cells
# (bench rounds, before = campaign-v2.2; BENCH_ROUNDS / BENCH_ARMS narrow them). Each job under the want-flag + flock.
# Usage: phase.sh <out_dir>
set -u
OUT=$1; mkdir -p "$OUT"
W=/scratch/wt/v2-highp BASE=/scratch/wt/v22 ART=$W/docs/artifacts/campaign-v2/v2-highp
export CUDA_VISIBLE_DEVICES=0 PY=/venvs/retrieve/bin/python RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
gpu() {
  touch /scratch/gpu0.st-dloop-wants
  flock /scratch/gpu0.lock "$@"; local rc=$?
  rm -f /scratch/gpu0.st-dloop-wants; echo "$(date -Is) rc=$rc: $*"; return $rc
}
for t in v22 v2-highp; do
  TRITON_CACHE_DIR=$(mktemp -d -p /scratch/v2-highp) PYTHONPATH= gpu $PY $ART/sass_fmkt.py --src /scratch/wt/$t/retrieve/src \
    > $OUT/sass_$t.txt 2> $OUT/sass_$t.err
done
# No fixed TORCHINDUCTOR_CACHE_DIR: the FX-graph cache keys a graph on the custom op, not its @triton_op body, so a
# directory reused across library edits replays stale kernels (testing.md § Running). pytest gets a fresh one; bench
# keys its own by code_version.
export PYTHONPATH=/scratch/v2-highp/pkgs:$W/retrieve/src:$W/retrieve:$ART
for nd in "800000 128" "3000000 128" "10000000 768"; do
  set -- $nd
  gpu $PY $ART/fmkt_gate.py exact $1 $2 $OUT/exact_n$1_d$2.json > $OUT/exact_n$1_d$2.log 2>&1
  gpu $PY $ART/fmkt_gate.py time $1 $2 $OUT/time_n$1_d$2.json > $OUT/time_n$1_d$2.log 2>&1
done
cd $W/retrieve
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d -p /scratch/v2-highp)
gpu $PY -m pytest tests/compile/test_linr_compile.py tests/parity/test_fused_masked_knn_topk.py -q -p no:cacheprovider > $OUT/compile_parity.log 2>&1
gpu $PY -m pytest tests/ -q -p no:cacheprovider > $OUT/pytest_retrieve.log 2>&1
unset TORCHINDUCTOR_CACHE_DIR
for r in ${BENCH_ROUNDS:-1 2}; do
  for arm in ${BENCH_ARMS:-before after}; do
    T=$([ $arm = before ] && echo $BASE || echo $W)
    (cd $T/evaluation && PYTHONPATH=$T/evaluation:$T/retrieve/src \
      gpu taskset -c 0-95 $PY -m bench.cli run --config-dir /scratch/v2-highp/config --dataset pubmed --dim 768 \
      --suite v2-highp --out $OUT/bench-$arm-r$r > $OUT/bench-$arm-r$r.log 2>&1)
  done
done
echo "$(date -Is) phase done"
