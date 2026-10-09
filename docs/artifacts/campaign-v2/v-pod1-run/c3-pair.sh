#!/usr/bin/env bash
# C3's bs-1 pair (exhibits, 2026-10-10): V1 Triton vs V1 torch.compile(max-autotune), goodreads-synth and arxiv-synth clause p01,
# seed 0, k 100, at the tag in common.sh. Records with quality (recall_oracle@100, ids_sha256_canon) from one narrowed `bench run`
# per dataset; the interleaved ratio from prof3.py's round-robin timing (Triton eager, Triton graph, compiled; bs 1 and 16).
LEG=c3-pair
. "$(dirname "$(readlink -f "$0")")/common.sh"
for ds in goodreads-synth arxiv-synth; do
  step "run-$ds" run --dataset "$ds" --suite synth --algo linr_v1_filter_mask --backend triton --backend torch \
    --filter-kind clause --sweep p01 --seed 0 --k 100 --out "$R" --resume
done
for it in bg1 bg16 ba1 ba16; do
  t0=$(date +%s)
  $PIN $PY "$HERE/../v-prof3/prof3.py" time "$it" "$LOG" > "$LOG/time-$it.log" 2>&1
  rc=$?; echo "$(date -Is) time $it rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc
done
finish
