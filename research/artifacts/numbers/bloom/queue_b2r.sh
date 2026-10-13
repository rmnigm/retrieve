#!/bin/bash
# bloom-runner B2 = numbers/A2-bloom (restartable copy, queue_b2r: every dataset gets the OOM fallback, not only laion30m), real filters: k 100 (priority 2, with LiNR V3 at matched recall), then k 1000 (3a). Usage: queue_b2.sh PLAN_DIR
# Interleaved per suite group (SilverTorch bloom recheck off + LiNR V1 / V2 / V3 exact). laion30m: if the interleaved process OOMs, its failed cells
# re-run one algo per process (--algo, --resume), as L1b did. check_b2.py gates the suites first. Uploads after k 100, superset after k 1000.
PLAN=$1 GRADE=${GRADE:-/scratch/wt/bloom-picks/research/artifacts/results-draft/grade_bloom.py}
FT=/scratch/wt/bloom-run R=/scratch/numbers/p1/bloom/run_bloom.sh B=/scratch/numbers/p1 V=/venvs/meta-fork
export LIB=eb600da1c0103f7963aab18a97e4ac970aea4883 FORKTREE=$FT INDUCTOR_TAG=-fork NUM=$PLAN/research/artifacts/numbers
say(){ echo "$(date -Is) $*" >> $B/logs/driver.log; }
G=$B/bloom/gate-cfg; rm -rf $G; bash $NUM/prep_config.sh $FT $G > /dev/null
( cd $FT/evaluation && PYTHONPATH=$FT/retrieve/src:$FT/retrieve:$FT/evaluation CUDA_VISIBLE_DEVICES="" $V/bin/python $B/bloom/check_b2.py $G ) > $B/bloom/gate-b2.log 2>&1 \
  || { say "A2-bloom refused by check_b2.py (see bloom/gate-b2.log)"; exit 1; }
touch $B/bloom/kmeans-pause   # the k-means test (kmeans_steps.sh) stops at its next step boundary until B2 k 100 is uploaded
say "A2-bloom start (plan $(git -C $PLAN rev-parse --short HEAD))"
split(){  # suite ds: on a failed interleaved process (OOM), re-run the failed cells: V1 + V2 together (as A2-pubmed), V3 alone, SilverTorch alone;
          # laion30m one algo per process (V1 + V2 together OOMed there in L1b). --resume skips ok cells.
  say "A2-bloom $1 $2: failed cells, re-running split by algo"
  if [ "$2" = laion30m ]; then G1="--algo linr_v1_filter_mask" G2="--algo linr_v2"; else G1="--algo linr_v1_filter_mask --algo linr_v2" G2=""; fi
  for e in "$G1" "$G2" "--algo linr_v3" "--algo silvertorch"; do [ -z "$e" ] && continue
    NOUPLOAD=1 EXTRA="$e" $R $FT $V A2-bloom $1 numbers/A2-bloom $2 || return 1; done
}
one(){ NOUPLOAD=1 $R $FT $V A2-bloom $1 numbers/A2-bloom $2 || split $1 $2; }
gap(){  # after k 100: the rest of the k-means test, then grade_bloom.py at B2's points (untimed), then k 1000
  rm -f $B/bloom/kmeans-pause
  while pgrep -f kmeans_steps.sh > /dev/null; do sleep 60; done
  say "A2-bloom gap: k-means test ended ($(cat $B/bloom/kmeans-done 2>/dev/null || echo 'no rc')); grade_bloom next"
  Gd=$B/bloom/grade; mkdir -p $Gd; uv run -q --no-project --with pyyaml python -I $B/bloom/grade_points.py $G/suites.yaml $Gd > $Gd/points.log 2>&1
  for f in $Gd/points-d*.json; do d=$(basename $f .json); d=${d#points-}
    ( cd $FT/evaluation && PYTHONPATH=$FT/retrieve/src:$FT/retrieve:$FT/evaluation RETRIEVE_DATA_ROOT=/data CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/numbers/inductor-p1-0-fork \
      flock /scratch/gpu0.lock $V/bin/python $GRADE $G n-bloom-timed-$d $f $Gd/grade-$d.csv > $Gd/grade-$d.log 2>&1 ) \
      && say "grade_bloom $d ok" || say "grade_bloom $d failed (see bloom/grade/grade-$d.log); B2 k 1000 continues"
  done
}
for k in "" "-k1000"; do
  for d in d128 d768 d256; do
    s=n-bloom-timed$k-$d; grep -q "^$s:" $G/suites.yaml || continue
    for ds in $(awk "/^$s:/{f=1;next} f&&/^[a-z]/{exit} f&&/^  datasets:/{g=1;next} g&&/^  - /{print \$2;next} g&&/^  [a-z]/{g=0}" $G/suites.yaml | grep -v laion30m) \
              $(awk "/^$s:/{f=1;next} f&&/^[a-z]/{exit} f&&/^  datasets:/{g=1;next} g&&/^  - laion30m/{print \$2}" $G/suites.yaml); do
      one $s $ds || exit 1
    done
  done
  s=n-bloom-timed$k-d128; $R $FT $V A2-bloom $s numbers/A2-bloom || exit 1   # upload only (no dataset): every suite folder so far
  say "A2-bloom k${k:-100} uploaded"
  [ -z "$k" ] && gap
done
say "A2-bloom done"
