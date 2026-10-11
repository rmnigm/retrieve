#!/usr/bin/env bash
# Re-derive the golden goodreads linr_v2 / linr_v3 triton cells at the frozen campaign-v2 library,
# with the golden README's method: the old harness (tmp/golden-rederive-l1l2 @ c72bfe5) with
# retrieve/ replaced by tag campaign-v2's (tree 408b1188), on throwaway branch
# tmp/golden-rederive-cv2, plus one port (golden_harness_port.diff: LiNR V4 out of the
# registry). Each cell runs twice, one process and a fresh inductor cache each. Clocks are
# sampled, not locked.
#   WT=/scratch/wt/golden-cv2 OUT=/scratch/r-res/golden-rederive bash golden_rederive.sh
set -euo pipefail
WT=${WT:?} OUT=${OUT:?}
export UV_PROJECT_ENVIRONMENT=/venvs/r-res-golden  # pinned: never the profile's /venvs/retrieve
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$OUT"
{
  echo "worktree: $WT @ $(git -C "$WT" rev-parse HEAD) ($(git -C "$WT" branch --show-current))"
  echo "library tree: $(git -C "$WT" rev-parse HEAD:retrieve/src/retrieve) (code_version; = campaign-v2:retrieve/src/retrieve)"
  echo "dirty: $(git -C "$WT" status --short --untracked-files=no | wc -l) tracked changes"
  echo "host: $(hostname)  utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi
  (cd "$WT/evaluation" && uv run --no-sync python -c "import sys, torch, triton, retrieve; print(sys.executable, retrieve.__file__, 'torch', torch.__version__, 'triton', triton.__version__)")
} > "$OUT/provenance.txt"
echo "utc,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu" > "$OUT/clocks.csv"
(while true; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits)" >> "$OUT/clocks.csv"
  sleep 30
done) &
SAMPLER=$!
trap 'kill $SAMPLER' EXIT
for run in 1 2; do
  for algo in linr_v2 linr_v3; do
    cell=goodreads-d128-c0_genre-$algo-triton
    mkdir -p "$OUT/run$run"
    (cd "$WT/evaluation" && TORCHINDUCTOR_CACHE_DIR="$OUT/inductor-run$run-$algo" \
      uv run --no-sync evaluate --config config/goodreads/d128-filter.yaml \
      --algo "$algo" --backend triton --filter-kind clause --sweep c0_genre \
      --output "$OUT/run$run/$cell.json") > "$OUT/run$run/$cell.log" 2>&1
    echo "run$run $cell done"
  done
done
