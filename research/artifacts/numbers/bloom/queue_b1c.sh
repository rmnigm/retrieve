#!/bin/bash
# Controller reorder: B1 real-filter curves first and uploaded (so the picks start), then the GPU window for yambda-runner; synth curves go to the end (queue_synth.sh, after B2).
# Waits for the in-flight pubmed call (run_bloom pid 3762411, orphaned from queue_b1b.sh on purpose).
FT=/scratch/wt/bloom-run R=/scratch/numbers/p1/bloom/run_bloom.sh B=/scratch/numbers/p1 V=/venvs/meta-fork
export LIB=eb600da1c0103f7963aab18a97e4ac970aea4883 FORKTREE=$FT INDUCTOR_TAG=-fork NUM=$B/plan-6a52275/research/artifacts/numbers INTERLEAVE=
say(){ echo "$(date -Is) $*" >> $B/logs/driver.log; }
while kill -0 3762411 2>/dev/null; do sleep 30; done
tail -3 $B/logs/driver.log | grep -q "A1-bloom run pubmed rc=0" || { say "bloom B1c not started: pubmed did not end clean"; exit 1; }
say "bloom B1c: laion10m, laion30m, then upload of the real-filter curves"
$R $FT $V A1-bloom n-bloom-curve-d256 numbers/A1-bloom laion10m laion30m || exit 1
say "bloom B1 real-filter curves uploaded; GPU window for yambda-runner until B2"
herdr agent prompt yambda-runner "bloom-runner: B1 real-filter curves uploaded and verified; GPU 0 is yours now under flock /scratch/gpu0.lock (1 h cap). B2 waits on the lock. Never print secrets." >> $B/bloom/herdr.log 2>&1
