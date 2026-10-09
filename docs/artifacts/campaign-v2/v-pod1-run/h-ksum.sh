#!/usr/bin/env bash
# H-KSUM's GPU pass: the `h2h` suite on goodreads + arXiv once more with --profile, now recording every
# kernel (kernels_us, kernels_calls), into a scratch tree; --skip-quality (the v2.1 h2h leg has the
# quality). Not a campaign leg: no timing claims; uploaded as artifacts/h-ksum-h2h.
LEG=h-ksum
R=/scratch/h-ksum/results
. "$(dirname "$(readlink -f "$0")")/common.sh"
grep -q "def kernel_summary" "$REPO/evaluation/bench/measure.py" \
  || { echo "$(date -Is) main checkout lacks H-KSUM (dev/h-ksum not merged / pulled), refusing"; exit 2; }
step campaign campaign --suite h2h --dataset goodreads --dataset arxiv --resume --interleave --profile \
  --skip-quality --out "$R"
finish
