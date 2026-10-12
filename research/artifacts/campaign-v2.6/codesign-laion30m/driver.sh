#!/usr/bin/env bash
# codesign-laion30m (C5 at 30 M: SilverTorch bloom partial vs full, official + triton, d256, n_probe {32, 128}, bs {16, 64}, 1 vs 4
# sub-queries; 8 jobs / 16 cells) at campaign-v2.6 on whichever pod d GPU laion frees first, from this worktree and its own venv
# (/venvs/d-run-v26), fresh inductor dir. triton + official in one stream (the parity spill). laion30m's oracle blobs are laion's
# (its v2.5 legs read them), so they are not moved: the blobs key on the attrs digest, not the library.
#   GPU=<i> setsid nohup flock -n /scratch/gpu<i>.lock bash driver.sh > /scratch/v26/codesign-laion30m.driver.log 2>&1 &
export TAG=campaign-v2.6 LEG=codesign-laion30m REPO=/scratch/wt/codesign-laion30m PY=/venvs/d-run-v26/bin/python
DS=laion30m SUITE=codesign-laion30m
R=/scratch/campaign-v26/$DS-$SUITE
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
step check check --dataset $DS --dim 256
step oracle oracle --dataset $DS --suite $SUITE --dim 256
stream $DS $SUITE "silvertorch|triton official"
left=$($PY "$HERE/../../campaign-v2/v-gr-deep/pending.py" "$R" "$cv" "$DS" "$SUITE" 2>/dev/null | tail -1)
echo "$(date -Is) $left"
grep -q ": 0 to run" <<< "$left" || { note pod-d "$LEG: cells still pending ($left); driver stopped"; exit 6; }
upload campaign-v2.6/$DS-$SUITE
msg="$LEG (16 cells) done on pod d GPU $GPU at $cv ($left); Hub campaign-v2.6/$DS-$SUITE: $UP"
note pod-d "$msg"; note control "$msg"
finish
