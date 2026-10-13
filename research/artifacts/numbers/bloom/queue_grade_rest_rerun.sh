#!/bin/bash
# After queue_b2r (its grade d768): grade_bloom for the d256 points the OOM left (laion30m 65536 alone, then laion3m), each in a fresh process;
# then the B2 unstable-cell rerun (queue_rerun.sh).
B=/scratch/numbers/p1 FT=/scratch/wt/bloom-run V=/venvs/meta-fork G=$B/bloom/grade GRADE=/scratch/wt/bloom-picks/research/artifacts/results-draft/grade_bloom.py
say(){ echo "$(date -Is) $*" >> $B/logs/driver.log; }
while pgrep -f "queue_b2r.sh" > /dev/null; do sleep 30; done
for r in rest30m rest3m; do
  ( cd $FT/evaluation && PYTHONPATH=$FT/retrieve/src:$FT/retrieve:$FT/evaluation RETRIEVE_DATA_ROOT=/data CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/numbers/inductor-p1-0-fork \
    flock /scratch/gpu0.lock $V/bin/python $GRADE $B/bloom/gate-cfg n-bloom-timed-d256 $G/points-d256-$r.json $G/grade-d256-$r.csv > $G/grade-d256-$r.log 2>&1 ) \
    && say "grade_bloom d256-$r ok" || say "grade_bloom d256-$r failed (see bloom/grade/grade-d256-$r.log)"
done
exec bash $B/bloom/queue_rerun.sh
