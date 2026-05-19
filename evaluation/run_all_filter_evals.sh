#!/usr/bin/env bash
# Run all 6 filter-eval configs serially. Designed to run inside tmux so it
# survives a closed shell session.
#
# Logs:
#   results/_runlogs/full.log              — combined stream of every config, tail -f this
#   results/_runlogs/SUMMARY.txt           — one-line-per-config pass/fail + duration
#   results/_runlogs/<dataset>_<dim>.log   — per-config full output
#   results/_runlogs/current.log -> ...    — symlink to the config currently running
set -uo pipefail

cd /workspace/retrieve/evaluation

LOGDIR="results/_runlogs"
mkdir -p "$LOGDIR"

FULL="$LOGDIR/full.log"
SUMMARY="$LOGDIR/SUMMARY.txt"
: > "$FULL"
: > "$SUMMARY"
{
    echo "started at: $(date -u +%FT%TZ)"
    echo "host:       $(hostname)"
    echo "gpu:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo ""
} | tee -a "$SUMMARY" "$FULL" >/dev/null

# Main filter eval — smallest dims first per dataset, arxiv before goodreads
# (no SASRec encode pass on arxiv → faster iteration on the first cells).
# All configs use a single canonical silvertorch and linr_v3 combo;
# arxiv users_limit set to 10k to match goodreads.
# Param-sweep / deep-curve configs (silvertorch params, linr_v3 candidate_pool)
# come LAST after all 6 main configs so the headline cross-algo comparison
# data is on disk before the deep sweeps add hours of single-algo curve work.
CONFIGS=(
    "conf/arxiv/d64-filter.yaml"
    "conf/arxiv/d128-filter.yaml"
    "conf/arxiv/d256-filter.yaml"
    "conf/goodreads/d64-filter.yaml"
    "conf/goodreads/d128-filter.yaml"
    "conf/goodreads/d256-filter.yaml"
    # "conf/deep_sweeps/arxiv-d128-silvertorch.yaml"  # silvertorch dropped 2026-05-19
    "conf/deep_sweeps/goodreads-d128-linr_v3.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    name=$(echo "$cfg" | sed -E 's|conf/||; s|/|_|g; s|\.yaml||')
    log="$LOGDIR/$name.log"
    ln -sfn "$(basename "$log")" "$LOGDIR/current.log"

    banner="=== $(date -u +%FT%TZ) starting $cfg ==="
    echo "$banner" | tee -a "$SUMMARY" "$FULL"

    SECONDS=0
    # tee both to the per-config log and the combined log; preserve exit code
    # via PIPESTATUS so a tee failure doesn't mask the real result.
    uv run evaluate --config "$cfg" 2>&1 | tee "$log" | tee -a "$FULL"
    rc=${PIPESTATUS[0]}
    dur=$SECONDS
    line=$(printf "%s  exit=%d  duration=%ds (%dh%dm)  log=%s" \
        "$cfg" "$rc" "$dur" $((dur/3600)) $(((dur%3600)/60)) "$log")
    echo "$line" | tee -a "$SUMMARY" "$FULL"
done

rm -f "$LOGDIR/current.log"
{
    echo ""
    echo "finished at: $(date -u +%FT%TZ)"
} | tee -a "$SUMMARY" "$FULL" >/dev/null
