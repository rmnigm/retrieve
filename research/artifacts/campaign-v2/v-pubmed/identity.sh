#!/usr/bin/env bash
# V-PUBMED step 1 on pod a100-x1-b: the embedding identity check, after restage.sh. GPU 0,
# NUMA-local cores. encode_queries, bench check, then one V1 cell (it builds only its own oracle)
# (clause c0_mesh, seed 0, --skip-perf) into a side file, compared with d1/pubmed by compare_v1.py.
# setsid nohup bash identity.sh > /scratch/v-pubmed/identity.log 2>&1 &
set -u
W=/scratch/wt/v-pubmed
PY=/venvs/retrieve/bin/python
EXPECT=408b1188d542634b3d18a2f5077bd23537a845fc
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v-pubmed
PIN="flock /scratch/gpu0.lock taskset -c 0-95"   # pod b shares GPU 0: every GPU command takes the lock
L=/scratch/v-pubmed; mkdir -p $L/identity $TORCHINDUCTOR_CACHE_DIR
cd $W/evaluation
cv=$($PIN $PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }
step() { local t0=$(date +%s) c="$*"; "$@"; local rc=$?; echo "$(date -Is) step ${c#$PIN } rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc; }
step $PIN $PY -m eval_datasets.etl.pubmed encode_queries > $L/logs/encode_queries.log 2>&1
step $PIN $PY -m bench.cli check --dataset pubmed > $L/logs/bench-check.log 2>&1
step $PIN $PY -m bench.cli run --dataset pubmed --dim 768 --suite filter --algo linr_v1_filter_mask \
  --backend triton --filter-kind clause --sweep c0_mesh --seed 0 --mode eager --skip-perf \
  --output $L/identity/v1-c0_mesh.jsonl > $L/logs/identity-cell.log 2>&1
step $PY $W/docs/artifacts/campaign-v2/v-pubmed/compare_v1.py /scratch/v-pubmed/d1/filter/pubmed-d768.jsonl $L/identity/v1-c0_mesh.jsonl
echo "$(date -Is) identity done"
