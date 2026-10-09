#!/usr/bin/env bash
# codesign-laion30m, triton half (8 cells) at campaign-v2.7 on pod d: one process per sweep until H-ARMFREE frees arms between builds (pod d's
# fit check: 11 GiB stayed allocated after a 30 M SilverTorch module). The official half is the v2.6 artifact
# (artifacts/codesign-laion30m-v26-official); official vs triton compare from the per-query sidecars.
#   GPU=<i> setsid nohup flock -n /scratch/gpu<i>.lock bash triton.sh > /scratch/v27/codesign-laion30m-triton.driver.log 2>&1 &
export TAG=campaign-v2.7 LEG=codesign-laion30m REPO=/scratch/wt/v27-codesign PY=/venvs/d-run-v27/bin/python
DS=laion30m SUITE=codesign-laion30m
R=/scratch/campaign-v27/$DS-$SUITE
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
step check check --dataset $DS --dim 256
for sw in c0_domain tags4; do NARROW="--sweep $sw" stream $DS $SUITE "silvertorch|triton"; done
n=$($PY -c 'import json,sys; rs=[json.loads(l) for l in open(sys.argv[1])]; print(sum(r["backend"]=="triton" and r["status"]=="ok" for r in rs), sum(r["status"]!="ok" for r in rs))' "$R/$SUITE/$DS-d256.jsonl")
echo "$(date -Is) triton ok / not ok: $n"
[ "$n" = "8 0" ] || { note pod-d "$LEG triton half: ok / not-ok $n; driver stopped"; exit 6; }
upload campaign-v2.7/$DS-$SUITE
msg="$LEG triton half (8 cells, one process per sweep) done on pod d GPU $GPU at $cv; Hub campaign-v2.7/$DS-$SUITE: $UP"
note pod-d "$msg"; note control "$msg"
finish
