#!/bin/bash
# bloom-runner B1 resume after the hold (2026-10-13 ruling) and the reaper fix: --resume re-runs only laion3m's 3 failed cells, then the rest.
FT=/scratch/wt/bloom-run R=/scratch/numbers/p1/bloom/run_bloom.sh B=/scratch/numbers/p1 V=/venvs/meta-fork
export LIB=eb600da1c0103f7963aab18a97e4ac970aea4883 FORKTREE=$FT INDUCTOR_TAG=-fork NUM=$B/plan-6a52275/research/artifacts/numbers INTERLEAVE=
say(){ echo "$(date -Is) $*" >> $B/logs/driver.log; }
say "bloom B1 (A1-bloom) resume"
NOUPLOAD=1 $R $FT $V A1-bloom n-bloom-curve-d256 numbers/A1-bloom laion3m || exit 1
NOUPLOAD=1 $R $FT $V A1-bloom n-bloom-curve-d768 numbers/A1-bloom pubmed || exit 1
NOUPLOAD=1 $R $FT $V A1-bloom n-bloom-curve-d256 numbers/A1-bloom laion10m laion30m || exit 1
$R $FT $V A1-bloom n-bloom-curve-synth-d128 numbers/A1-bloom goodreads-synth arxiv-synth || exit 1
say "bloom B1 (A1-bloom) done"
