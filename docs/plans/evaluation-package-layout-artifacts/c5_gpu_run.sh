#!/bin/bash
# C5's GPU stages (evaluation-package-layout.md §9): (1) the one-cell gate — goodreads-d128
# c0_genre silvertorch/triton, the suite's cells, both modes, full perf; (2) L4-b — linr_v4/triton
# on the same sweep, quality only, chunk 64 (l4b_chunk64.py). One GPU job at a time under the lock.
set -euo pipefail
WT=/workspace/wt/c5
ART=$WT/docs/plans/evaluation-package-layout-artifacts/c5
export UV_PROJECT_ENVIRONMENT=/venvs/c5 RETRIEVE_DATA_ROOT=/workspace/data TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-c5
cd $WT/evaluation
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,utilization.gpu --format=csv > $ART/provenance.txt
git -C $WT rev-parse HEAD HEAD:retrieve/src/retrieve >> $ART/provenance.txt
git -C $WT status --porcelain -- retrieve/src/retrieve >> $ART/provenance.txt || true
echo "=== gate cell $(date -u +%FT%TZ)" | tee -a $ART/driver.log
uv run --no-sync bench run --dataset goodreads --dim 128 --suite filter --filter-kind clause --sweep c0_genre \
    --algo silvertorch --backend triton --out $ART/results > $ART/gate-silvertorch-triton.log 2>&1 || echo "gate rc=$?" | tee -a $ART/driver.log
echo "=== L4-b linr_v4 chunk 64 $(date -u +%FT%TZ)" | tee -a $ART/driver.log
uv run --no-sync python $WT/docs/plans/evaluation-package-layout-artifacts/l4b_chunk64.py \
    --dataset goodreads --dim 128 --suite filter --filter-kind clause --sweep c0_genre \
    --algo linr_v4 --backend triton --skip-perf --out $ART/results-l4b-chunk64 > $ART/l4b-linr_v4-chunk64.log 2>&1 || echo "l4b rc=$?" | tee -a $ART/driver.log
echo "=== done $(date -u +%FT%TZ)" | tee -a $ART/driver.log
