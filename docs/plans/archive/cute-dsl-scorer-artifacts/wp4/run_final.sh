#!/bin/bash
cd /workspace/retrieve/retrieve
W=/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp4
echo "=== h2h final ($(date +%T))"; uv run python -u $W/h2h.py $W/h2h-final.json > $W/h2h-final.txt 2>&1; echo "exit $?"
echo "=== kernel_only final ($(date +%T))"; uv run python -u $W/kernel_only.py $W/kernel_only-final.json A > $W/kernel_only-final.txt 2>&1; echo "exit $?"
echo "=== tune cute re-sweep ($(date +%T))"
uv run tune-kernels codesigned-probe-score-cute --json-out $W/cps-cute-posttrim.json > $W/tune-cps-cute-posttrim.txt 2>&1; echo "exit $?"; tail -3 $W/tune-cps-cute-posttrim.txt
uv run tune-kernels codesigned-probe-score-exact-cute --json-out $W/cpse-cute-posttrim.json > $W/tune-cpse-cute-posttrim.txt 2>&1; echo "exit $?"; tail -3 $W/tune-cpse-cute-posttrim.txt
echo "=== compile facts ($(date +%T))"; uv run python -u $W/compile_facts.py > $W/compile_facts.txt 2>&1; echo "exit $?"
for m in none bloom exact; do uv run python -u $W/compile_fresh.py $m > $W/compile_fresh-$m.txt 2>&1; echo "fresh $m exit $?"; done
echo "=== full parity+compile ($(date +%T))"; uv run pytest tests/parity tests/compile -q > $W/pytest-final.txt 2>&1; echo "exit $?"; tail -2 $W/pytest-final.txt
echo "FINAL DONE ($(date +%T))"
