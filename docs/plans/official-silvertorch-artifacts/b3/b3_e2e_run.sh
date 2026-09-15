#!/usr/bin/env bash
# B3 (roadmap) — end-to-end head-to-head, O §9c: silvertorch on {triton, torch, official},
# both datasets at d128, the headline sweeps (goodreads c0_genre, arxiv c0_maincat), seed 0.
# The `filter` suite runs unnarrowed (ks 100/500/1000 × bs 1/8/16 → status ok); the `quality`
# suite is narrowed to bs 1/8/16 at k=100 so the unfiltered arm is timed at the same batch
# sizes (status partial, on purpose).
set -euo pipefail
cd "$(dirname "$0")/../../../.."/evaluation
export UV_PROJECT_ENVIRONMENT=/venvs/b3
export RETRIEVE_DATA_ROOT=/workspace/data
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-b3
OUT="$(cd .. && pwd)/docs/plans/official-silvertorch-artifacts/b3/e2e"
LOGS="$OUT/_logs"
mkdir -p "$LOGS"

run () {  # run <dataset> <suite> <sweep> <extra...>
  local ds=$1 suite=$2 sweep=$3; shift 3
  for backend in triton torch official; do
    local tag="${suite}_${ds}_${backend}"
    echo "=== $(date -Is) $tag" | tee -a "$LOGS/campaign.log"
    uv run python -m bench.cli run \
      --dataset "$ds" --suite "$suite" --dim 128 --algo silvertorch \
      --backend "$backend" ${sweep:+--sweep "$sweep"} --seed 0 \
      --out "$OUT" "$@" >"$LOGS/$tag.log" 2>&1 || echo "rc=$? $tag" | tee -a "$LOGS/campaign.log"
  done
}

run goodreads filter c0_genre
run arxiv    filter c0_maincat
run goodreads quality "" --k 100 --bs 1 --bs 8 --bs 16
run arxiv    quality "" --k 100 --bs 1 --bs 8 --bs 16
echo "done $(date -Is)" | tee -a "$LOGS/campaign.log"
