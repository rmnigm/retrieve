#!/bin/bash
# WP-4 step 2: six tune sweeps, sequential (GPU otherwise idle).
set -u
cd /workspace/retrieve/retrieve
W=/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp4
for spec in codesigned-probe-score-cute codesigned-probe-score-exact-cute \
            codesigned-probe-score-cuda codesigned-probe-score-exact-cuda \
            codesigned-probe-score codesigned-probe-score-exact; do
  case $spec in
    codesigned-probe-score) tag=cps-triton;;
    codesigned-probe-score-exact) tag=cpse-triton;;
    codesigned-probe-score-cuda) tag=cps-cuda;;
    codesigned-probe-score-exact-cuda) tag=cpse-cuda;;
    codesigned-probe-score-cute) tag=cps-cute;;
    codesigned-probe-score-exact-cute) tag=cpse-cute;;
  esac
  echo "=== $spec -> $tag ($(date +%T))"
  uv run tune-kernels $spec --json-out $W/$tag.json > $W/tune-$tag.txt 2>&1
  echo "exit $? ($(date +%T))"; tail -3 $W/tune-$tag.txt
done
echo ALL DONE
