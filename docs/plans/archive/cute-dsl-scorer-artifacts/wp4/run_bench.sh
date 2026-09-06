#!/bin/bash
cd /workspace/retrieve/retrieve
W=/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp4
tag=${1:-base}
if [ "${2:-}" = "parity" ]; then
  echo "=== parity ($(date +%T))"; uv run pytest tests/parity/test_codesigned_probe_score_cute.py -q > $W/parity-$tag.txt 2>&1; echo "exit $?"; tail -1 $W/parity-$tag.txt
fi
echo "=== h2h ($(date +%T))"; uv run python $W/h2h.py $W/h2h-$tag.json > $W/h2h-$tag.txt 2>&1; echo "exit $?"
echo "=== kernel_only ($(date +%T))"; uv run python $W/kernel_only.py $W/kernel_only-$tag.json A > $W/kernel_only-$tag.txt 2>&1; echo "exit $?"
echo "=== host_overhead ($(date +%T))"; uv run python $W/host_overhead.py $W/host_overhead-$tag.json > $W/host_overhead-$tag.txt 2>&1; echo "exit $?"
echo "BENCH DONE ($(date +%T))"
