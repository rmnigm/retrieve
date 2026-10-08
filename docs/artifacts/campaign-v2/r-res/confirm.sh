#!/usr/bin/env bash
# R-RES GPU confirmation: reproduce each golden residual from the frozen tree by undoing the one
# difference the CPU analysis names. Quality only, eager, seed 0, one GPU.
#   A: arXiv SilverTorch triton (n_probe 24) run at k = 100 alone, as the old harness did,
#      instead of the k_max = 1000 prefix.
#   B: goodreads linr_v2 / linr_v3 triton with fused_masked_knn_topk reverted to the pre-L5
#      fp16 operand sum (pre_l5_fp16_sum.diff, undone on exit), one run per k as the old
#      harness did.
# Run from the repository root:  OUT=/scratch/r-res bash docs/artifacts/campaign-v2/r-res/confirm.sh
set -euo pipefail
OUT=${OUT:?}
HERE=$(cd "$(dirname "$0")" && pwd)
# Never inherit: the pod profile exports the shared /venvs/retrieve, editable onto another checkout.
export UV_PROJECT_ENVIRONMENT=/venvs/r-res
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCHINDUCTOR_CACHE_DIR=$OUT/inductor TRITON_CACHE_DIR=$OUT/triton
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,clocks.sm --format=csv > "$OUT/gpu.txt"
git rev-parse HEAD > "$OUT/commit.txt"
bench() { (cd evaluation && uv run --no-sync python -c "import retrieve, sys; print(sys.executable, retrieve.__file__)" && uv run --no-sync bench run "$@"); }
CKPT=data/goodreads-work-id/checkpoints/gsasrec-d128-drop0.5-id/best_model.pt

bench --dataset arxiv --dim 128 --suite filter --algo silvertorch --backend triton \
  --filter-kind clause --sweep c0_maincat --seed 0 --mode eager --skip-perf --k 100 \
  --out "$OUT/A" 2>&1 | tee "$OUT/A.log"

git apply "$HERE/pre_l5_fp16_sum.diff"
trap 'git apply -R "$HERE/pre_l5_fp16_sum.diff"' EXIT
grep -c "to(tl.float32)" retrieve/src/retrieve/ops/triton/fused_masked_knn_topk.py | sed "s/^/widen casts left: /" | tee "$OUT/patch.txt" || true
for k in 100 500 1000; do
  bench --dataset goodreads --dim 128 --suite filter --algo linr_v2 --algo linr_v3 \
    --backend triton --filter-kind clause --sweep c0_genre --seed 0 --mode eager --skip-perf \
    --checkpoint "$CKPT" --k "$k" --out "$OUT/B-k$k" 2>&1 | tee "$OUT/B-k$k.log"
done
