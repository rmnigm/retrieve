#!/usr/bin/env bash
# Night queue 8 continuation: tune.sh's per-n_lists body for the n_lists given as arguments (65536 when tune_need_65536.py says so).
# Same tree, same jobs, cumulative upload to campaign-v2.9/laion30m-tune. Stop between n_lists: touch $L/tune.stop. Rerun: --resume.
. "$(dirname "$(readlink -f "$0")")/common.sh"
R=/scratch/campaign-v29/laion30m-tune
for nl in "$@"; do
  [ -e "$L/tune.stop" ] && { echo "$(date -Is) stop file: ending before n_lists $nl"; break; }
  C=$L/tune-config-$nl
  $PY "$HERE/tune_config.py" "$C" $nl 2>> "$L/logs/tune-config.log" || exit 5
  for job in "laion30m tags4 clause" "laion30m tags4 bloom" "laion30m c0_domain clause" "laion30m c0_domain bloom" \
             "laion30m-synth p001 clause" "laion30m-synth p01 clause" "laion30m-synth p1 clause"; do
    set -- $job
    run "tune-$nl-$1-$2-$3" run --config-dir "$C" --dataset $1 --dim 256 --suite tune --sweep $2 --filter-kind $3 \
      --mode graph --out "$R" --resume
  done
  cp -a "$L/logs/clocks-gpu$GPU.csv" "$L"/logs/tune-*.log "$R/logs/"
  run "upload-tune-$nl" upload --results "$R" --path-in-repo campaign-v2.9/laion30m-tune --verify
  grep -E "MANIFEST|round trip" "$L/logs/upload-tune-$nl.log"
done
echo "$(date -Is) tune-nl rc=0"
