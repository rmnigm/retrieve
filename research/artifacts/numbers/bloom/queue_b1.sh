#!/bin/bash
# bloom-runner B1 = numbers/A1-bloom: quality-only bloom curves, recheck off (plan dev/results-draft 6a52275, run tree run/numbers-bloom a9b8951).
# laion1m-synth / laion3m-synth: n-bloom-curve-synth-d256 has 0 jobs (no bloom sweeps in their dataset yamls), not run.
FT=/scratch/wt/bloom-run R=/scratch/numbers/p1/bloom/run_bloom.sh B=/scratch/numbers/p1 V=/venvs/meta-fork
export LIB=eb600da1c0103f7963aab18a97e4ac970aea4883 FORKTREE=$FT INDUCTOR_TAG=-fork NUM=$B/plan-6a52275/research/artifacts/numbers INTERLEAVE=
say(){ echo "$(date -Is) $*" >> $B/logs/driver.log; }
say "bloom B1 (A1-bloom) start"
NOUPLOAD=1 $R $FT $V A1-bloom n-bloom-curve-d128 numbers/A1-bloom goodreads arxiv || exit 1
NOUPLOAD=1 $R $FT $V A1-bloom n-bloom-curve-d256 numbers/A1-bloom laion1m laion3m || exit 1
NOUPLOAD=1 $R $FT $V A1-bloom n-bloom-curve-d768 numbers/A1-bloom pubmed || exit 1
NOUPLOAD=1 $R $FT $V A1-bloom n-bloom-curve-d256 numbers/A1-bloom laion10m laion30m || exit 1
$R $FT $V A1-bloom n-bloom-curve-synth-d128 numbers/A1-bloom goodreads-synth arxiv-synth || exit 1
say "bloom B1 (A1-bloom) done"
