#!/usr/bin/env bash
# codesign-laion30m, triton arm alone in a fresh process (its builds OOMed in build_transposed_sigs after the official arm in one process).
#   GPU=<i> setsid nohup flock -n /scratch/gpu<i>.lock bash triton.sh > /scratch/v26/codesign-laion30m-triton.driver.log 2>&1 &
export TAG=campaign-v2.6 LEG=codesign-laion30m REPO=/scratch/wt/codesign-laion30m PY=/venvs/d-run-v26/bin/python
DS=laion30m SUITE=codesign-laion30m
R=/scratch/campaign-v26/$DS-$SUITE
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
stream $DS $SUITE "silvertorch|triton"
left=$($PY "$HERE/../../campaign-v2/v-gr-deep/pending.py" "$R" "$cv" "$DS" "$SUITE" 2>/dev/null | tail -1)
echo "$(date -Is) $left"
grep -q ": 0 to run" <<< "$left" || { note pod-d "$LEG: cells still pending ($left); driver stopped"; exit 6; }
upload campaign-v2.6/$DS-$SUITE
msg="$LEG (16 cells; triton arm in its own process) done on pod d GPU $GPU at $cv ($left); Hub campaign-v2.6/$DS-$SUITE: $UP"
note pod-d "$msg"; note control "$msg"
finish
