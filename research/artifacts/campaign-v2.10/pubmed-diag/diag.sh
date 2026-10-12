#!/usr/bin/env bash
# PubMed diagnostic (a) on pod a100-x1-b GPU 0 at campaign-v2.10 (library a3bec4a5, detached tag tree), quality only, one short step
# under the GPU lock behind st-dloop. The k 1000 oracles build at v2.10 in the mirror data dir's own gt_d768 (pubmed-tune's), then
# are compared with torch.equal against the shared v2.9 blobs. (b) is CPU only: bloom_seeds.py.
# setsid nohup bash diag.sh > /scratch/pubmed-diag/diag.log 2>&1 &
set -u
FROM=${FROM:-all}  # FROM=rescore runs only the V2 rescoring step (exact_miss.csv already written)
T=/scratch/wt/tree-v210
D=/scratch/wt/pubmed-diag/docs/artifacts/campaign-v2.10/pubmed-diag
PY=/venvs/c7-v28/bin/python
O=/scratch/pubmed-diag
export PYTHONPATH=$T/evaluation:$T/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/pubmed-diag
mkdir -p $O $TORCHINDUCTOR_CACHE_DIR
cd $T/evaluation
yield_gpu() { while [ -e /scratch/gpu0.st-dloop-wants ] || [ -e /scratch/gpu0.st-dloop-phase ]; do sleep 20; done; }
step() { local t0=$(date +%s); "$@"; local rc=$?; echo "$(date -Is) step ${*:1:4} rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc; }
[ $FROM = rescore ] || { yield_gpu; step flock /scratch/gpu0.lock taskset -c 0-95 $PY $D/exact_miss.py $O/cfg $O/exact_miss; }
yield_gpu
step flock /scratch/gpu0.lock taskset -c 0-95 $PY $D/v2_rescore.py $O/exact_miss.csv $O/cfg $O/exact_miss_classes
step $PY $D/oracle_compare.py /data/pubmed-medcpt/gt_d768 /scratch/pubmed-tune-v210/data/pubmed-medcpt/gt_d768
echo "$(date -Is) diag done"
