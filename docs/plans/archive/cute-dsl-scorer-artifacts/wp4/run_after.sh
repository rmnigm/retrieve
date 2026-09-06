#!/bin/bash
cd /workspace/retrieve/retrieve
W=/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp4
echo "=== host_overhead after ($(date +%T))"; uv run python -u $W/host_overhead.py $W/host_overhead-after.json > $W/host_overhead-after.txt 2>&1; echo "exit $?"
echo "=== p3 unpinned ($(date +%T))"; uv run python -u $W/p3_bench.py unpinned > $W/p3-unpinned.txt 2>&1; echo "exit $?"
echo "=== parity after trims ($(date +%T))"; uv run pytest tests/parity/test_codesigned_probe_score_cute.py -q > $W/parity-trims.txt 2>&1; echo "exit $?"; tail -1 $W/parity-trims.txt
echo "=== compile cute rows ($(date +%T))"; uv run pytest tests/compile -q -k cute > $W/compile-cute-trims.txt 2>&1; echo "exit $?"; tail -1 $W/compile-cute-trims.txt
echo "AFTER DONE ($(date +%T))"
