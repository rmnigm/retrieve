#!/usr/bin/env bash
# Resume wrapper — only the 3 remaining configs, appends to existing logs.
# Order: finish arxiv (d256 retry), then finish goodreads (d256), then deep sweeps.
set -uo pipefail

cd /workspace/retrieve/evaluation

LOGDIR="results/_runlogs"
FULL="$LOGDIR/full.log"
SUMMARY="$LOGDIR/SUMMARY.txt"
{
    echo ""
    echo "--- resume run @ $(date -u +%FT%TZ) ---"
} | tee -a "$SUMMARY" "$FULL" >/dev/null

CONFIGS=(
    "conf/arxiv/d256-filter.yaml"
    "conf/goodreads/d256-filter.yaml"
    "conf/deep_sweeps/goodreads-d128-linr_v3.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    name=$(echo "$cfg" | sed -E 's|conf/||; s|/|_|g; s|\.yaml||')
    log="$LOGDIR/$name.log"
    ln -sfn "$(basename "$log")" "$LOGDIR/current.log"

    banner="=== $(date -u +%FT%TZ) starting $cfg ==="
    echo "$banner" | tee -a "$SUMMARY" "$FULL"

    SECONDS=0
    ./run_per_algo.sh "$cfg" 2>&1 | tee "$log" | tee -a "$FULL"
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
