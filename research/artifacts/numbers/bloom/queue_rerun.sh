#!/bin/bash
# B2 k 100 unstable-cell rerun (controller: all 15): goodreads + arxiv (d128), laion1m / 3m / 10m / 30m (d256), same frozen plan 2b3f662, tree a9b8951, .so 8ae8c68c.
# Stable ok records of numbers/A2-bloom are seeded first, so --resume re-runs only the unstable cells (with their interleave partners).
FT=/scratch/wt/bloom-run R=/scratch/numbers/p1/bloom/run_bloom.sh B=/scratch/numbers/p1 V=/venvs/meta-fork
export LIB=eb600da1c0103f7963aab18a97e4ac970aea4883 FORKTREE=$FT INDUCTOR_TAG=-fork NUM=$B/plan-2b3f662/research/artifacts/numbers SEED=$B/out/A2-bloom
say(){ echo "$(date -Is) $*" >> $B/logs/driver.log; }
while pgrep -f "queue_b2r.sh" > /dev/null; do sleep 30; done
say "A2-bloom-rerun start (all 15 unstable cells: goodreads, arxiv, laion1m, laion3m, laion10m, laion30m)"
NOUPLOAD=1 $R $FT $V A2-bloom-rerun n-bloom-timed-d128 numbers/A2-bloom-rerun goodreads arxiv || exit 1
$R $FT $V A2-bloom-rerun n-bloom-timed-d256 numbers/A2-bloom-rerun laion1m laion3m laion10m laion30m || exit 1
say "A2-bloom-rerun done"
