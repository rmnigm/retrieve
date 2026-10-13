#!/bin/bash
# k-means test on pod 1 GPU 0: fit the centroids of every variant (fit_centroids.py), then km-sanity, km-4096, km-16384
# through bench_cached.py, each step under the GPU lock (released between steps so B2 can take it), then upload.
#   run_kmeans.sh <tree> <venv> <upload path>
set -u
TREE=$1 VENV=$2 UP=$3 HERE=$(cd "$(dirname "$0")" && pwd)
B=/scratch/kmeans-test OUT=$B/out L=$B/logs C=$B/centroids
mkdir -p $OUT/config $L $C
say(){ echo "$(date -Is) kmeans-test $*" | tee -a $L/driver.log; }
for f in "$TREE"/evaluation/config/*.yaml; do [ "$(basename "$f")" = suites.yaml ] || cp "$f" $OUT/config/; done
cp $HERE/suites.yaml $OUT/config/suites.yaml
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/kmeans-test RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export PYTHONPATH=$TREE/retrieve/src:$TREE/evaluation KMEANS_CACHE=$C
cd $TREE/evaluation
nvidia-smi -q -d CLOCK > $L/clocks-before.txt
for v in "4096 random 10" "4096 kmeans++ 10" "4096 kmeans++ 25" "16384 random 10" "16384 kmeans++ 10" "16384 kmeans++ 25"; do
  set -- $v; [ -f $C/$2-$3-$1-s0.json ] && continue
  t0=$(date +%s)
  flock /scratch/gpu0.lock $VENV/bin/python $HERE/fit_centroids.py data/laion30m/content_d256 $C $1 $2 $3 >> $L/fit.log 2>&1 \
    || { say "fit $v failed"; exit 1; }
  say "fit $v s=$(( $(date +%s) - t0 ))"
done
flock /scratch/gpu0.lock $VENV/bin/python -m bench.cli oracle --dataset laion30m --suite km-4096 --config-dir $OUT/config > $L/oracle.log 2>&1 \
  || { say "oracle failed"; exit 1; }
for s in ${SUITES:-km-sanity km-4096 km-16384}; do
  t0=$(date +%s)
  flock /scratch/gpu0.lock $VENV/bin/python $HERE/bench_cached.py run --config-dir $OUT/config --out $OUT --suite $s --dataset laion30m --resume > $L/run-$s.log 2>&1
  rc=$?; say "run $s rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc = 0 ] || exit 1
done
nvidia-smi -q -d CLOCK > $L/clocks-after.txt
[ -n "${NOUPLOAD:-}" ] && { say "upload deferred"; exit 0; }
T=$(mktemp -d $B/upload.XXXXXX)
for d in $OUT/km-*/; do mkdir -p $T/$(basename $d); cp -al $d* $T/$(basename $d)/; done
mkdir -p $T/centroids $T/logs; cp $C/*.json $T/centroids/; cp $L/*.log $L/clocks-*.txt $T/logs/
$VENV/bin/python -m bench.cli upload --results $T --path-in-repo $UP --verify > $L/upload.log 2>&1 || { say "upload failed"; exit 1; }
say "uploaded $UP $(grep -o 'MANIFEST.json sha256 [0-9a-f]*' $L/upload.log)"
