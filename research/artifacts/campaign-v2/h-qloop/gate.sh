#!/usr/bin/env bash
# H-QLOOP's byte-identical gate (GPU, under the lock): the same cells with --skip-perf under the current harness (A: the
# main checkout) and the H-QLOOP harness (B: this worktree's evaluation/ on PYTHONPATH, the same imported library), into
# two scratch trees, then compare.py. Cells: the freeze's golden set (suites_freeze.py golden) and one cell per arm.
# Launch: setsid nohup flock -n /scratch/gpu0.lock bash gate.sh > /scratch/h-qloop/gate.log 2>&1 &
set -u
if flock -n /scratch/gpu0.lock true; then echo "$(date -Is) not launched under /scratch/gpu0.lock, refusing"; exit 3; fi
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../../../.." && pwd)
MAIN=/workspace/retrieve
PY=/venvs/retrieve/bin/python
OUT=/scratch/h-qloop
PIN="taskset -c 0-63,128-191"
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hf
rm -rf "$OUT/A" "$OUT/B" "$OUT/golden-config"; mkdir -p "$OUT"
(cd "$MAIN/evaluation" && $PY "$MAIN/docs/artifacts/campaign-v2/freeze/suites_freeze.py" golden "$OUT/golden-config") > "$OUT/golden-cells.txt" 2>&1 || exit 1
cells=(  # dataset|suite|config|narrowing (CLI), each one `bench run --skip-perf`
  "goodreads|filter|golden|"
  "arxiv|filter|golden|"
  "goodreads-synth|synth|real|--algo linr_v1_filter_mask --algo linr_v2 --backend triton --filter-kind clause"
  "goodreads-synth|synth|real|--algo linr_v3 --backend triton --filter-kind clause"
  "goodreads-synth|synth|real|--algo silvertorch --backend triton --filter-kind clause"
  "goodreads-synth|synth|real|--algo silvertorch --backend triton --backend official --filter-kind bloom"
  "goodreads-synth|synth|real|--algo postfilter --backend torch --filter-kind clause"
  "goodreads-synth|synth|real|--algo linr_v1_filter_mask --algo linr_v2 --backend torch --filter-kind clause"
  "goodreads|filter|real|--algo silvertorch --backend torch --filter-kind clause --sweep c0_genre"
)
for side in A B; do
  if [ $side = A ]; then dir=$MAIN/evaluation; pp=""; else dir=$WT/evaluation; pp=$WT/evaluation; fi
  for c in "${cells[@]}"; do
    IFS='|' read -r ds suite cfg narrow <<< "$c"
    cd_=config; [ "$cfg" = golden ] && cd_=$OUT/golden-config
    sweep=""; [ "$ds" = goodreads-synth ] && sweep="--sweep p01"
    t0=$(date +%s)
    (cd "$dir" && PYTHONPATH=$pp TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/h-qloop-$side $PIN $PY -m bench.cli run \
       --config-dir "$cd_" --dataset "$ds" --suite "$suite" $narrow $sweep --seed 0 --skip-perf --out "$OUT/$side" --resume) \
       >> "$OUT/run-$side.log" 2>&1
    rc=$?
    echo "$(date -Is) $side $ds/$suite $narrow rc=$rc s=$(( $(date +%s) - t0 ))"
  done
done
$PY "$HERE/compare.py" "$OUT/A" "$OUT/B" | tee "$OUT/compare.txt"
echo "$(date -Is) gate done rc=${PIPESTATUS[0]}"
