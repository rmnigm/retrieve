#!/bin/bash
# Fresh-inductor-cache recheck: ST-IDS and V2-HIGHP compile tests + library suites, then V2-HIGHP's bench after-round
# under the harness's own per-code_version cache.
set -u
export CUDA_VISIBLE_DEVICES=0 PY=/venvs/retrieve/bin/python RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
gpu() { touch /scratch/gpu0.st-dloop-wants; flock /scratch/gpu0.lock "$@"; local rc=$?; rm -f /scratch/gpu0.st-dloop-wants; echo "$(date -Is) rc=$rc: $*"; return $rc; }
for t in st-ids v2-highp; do
  out=$([ $t = st-ids ] && echo /scratch/st-ids/recheck || echo /scratch/v2-highp/phase4)
  (cd /scratch/wt/$t/retrieve && PYTHONPATH=/scratch/wt/$t/retrieve/src:/scratch/wt/$t/retrieve TORCHINDUCTOR_CACHE_DIR=$(mktemp -d -p /scratch/st-ids) \
    gpu $PY -m pytest tests/ -q -p no:cacheprovider > $out/pytest_fresh_cache.log 2>&1)
done
(cd /scratch/wt/v2-highp/evaluation && PYTHONPATH=/scratch/wt/v2-highp/evaluation:/scratch/wt/v2-highp/retrieve/src \
  gpu taskset -c 0-95 $PY -m bench.cli run --config-dir /scratch/v2-highp/config --dataset pubmed --dim 768 \
  --suite v2-highp --out /scratch/v2-highp/phase4/bench-after-r1 > /scratch/v2-highp/phase4/bench-after-r1.log 2>&1)
echo "$(date -Is) recheck done"
