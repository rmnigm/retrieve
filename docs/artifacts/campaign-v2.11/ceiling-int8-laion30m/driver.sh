#!/usr/bin/env bash
# C6 ceiling check on LAION 30 M d256 at campaign-v2.11 (controller 2026-10-12): laion30m c0_domain + tags4 and laion30m-synth p 0.01 / 0.1 / 1 at
# laion30m-tune's best clause point (n_lists 16384, n_probe 4096 for every sweep; Hub campaign-v2.9/laion30m-tune), scored shipped / global int8 /
# per-row int8 / fp16 by ceiling_check.py; quality only; pod d GPU 1 from the tag tree. Hub artifacts/ceiling-int8-laion30m.
#   GPU=1 setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v211/ceiling-laion30m.driver.log 2>&1 &
export TAG=campaign-v2.11 LEG=ceiling-laion30m REPO=/scratch/wt/v211-tag PY=/venvs/d-run-v211/bin/python GPU=${GPU:-1}
R=/scratch/campaign-v211/ceiling-int8-laion30m
export CEIL_CFG=/scratch/v211/ceil-config
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
rm -rf "$CEIL_CFG" && cp -r "$REPO/evaluation/config" "$CEIL_CFG"
printf 'ceil:\n  datasets: [laion30m, laion30m-synth]\n  dims: [256]\n  filter_kinds: [clause]\n  ks: [100]\n  batch_sizes: [16]\n  seeds: [0]\n  sweeps:\n    laion30m: [c0_domain, tags4]\n    laion30m-synth: [p001, p01, p1]\n  arms:\n    - {algo: silvertorch, backends: [triton], build: {n_lists: [16384]}, query: {n_probe: [4096]}}\n' >> "$CEIL_CFG/suites.yaml"
for pt in laion30m:c0_domain laion30m:tags4 laion30m-synth:p001 laion30m-synth:p01 laion30m-synth:p1; do
  ds=${pt%%:*}; sw=${pt#*:}; t0=$(date +%s)
  $PIN $PY "$HERE/../../campaign-v2.11/ceiling-int8-laion30m/ceiling_check.py" $ds ceil $sw 16384 4096 "$R/ceiling-$ds-$sw.json" > "$R/logs/ceiling-$ds-$sw.log" 2>&1 < /dev/null
  rc=$?; echo "$(date -Is) step $ds/$sw rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || { note pod-d "$LEG $ds/$sw rc=$rc; driver stopped"; exit $rc; }
done
upload artifacts/ceiling-int8-laion30m
msg="$LEG (5 sweeps, quality only) done on pod d GPU $GPU at $cv; Hub artifacts/ceiling-int8-laion30m: $UP"
note pod-d "$msg"; note control "$msg"
finish
