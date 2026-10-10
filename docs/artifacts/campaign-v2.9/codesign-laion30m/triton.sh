#!/usr/bin/env bash
# C5 ours at 30 M re-read at campaign-v2.9 (ST-WIDE: at d256 the per-row probe table, B · n_probe >= 512): codesign-laion30m's 8 triton cells
# (bloom partial + full, c0_domain + tags4, n_probe 32 / 128, bs 16 / 64), one process per sweep, from a worktree at the campaign-v2.9 tag
# (/scratch/wt/v29-tag, venv /venvs/d-run-v29), fresh inductor dir. Hub campaign-v2.9/laion30m-codesign-laion30m (triton).
#   GPU=0 setsid nohup flock -n /scratch/gpu0.lock bash triton.sh > /scratch/v29/codesign-laion30m-triton.driver.log 2>&1 &
export TAG=campaign-v2.9 LEG=codesign-laion30m-v29 REPO=/scratch/wt/v29-tag PY=/venvs/d-run-v29/bin/python GPU=${GPU:-0}
DS=laion30m SUITE=codesign-laion30m
R=/scratch/campaign-v29/$DS-$SUITE
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
step check check --dataset $DS --dim 256
for sw in c0_domain tags4; do NARROW="--sweep $sw" stream $DS $SUITE "silvertorch|triton"; done
n=$($PY -c 'import json,sys; rs=[json.loads(l) for l in open(sys.argv[1])]; print(sum(r["backend"]=="triton" and r["status"]=="ok" for r in rs), sum(r["status"]!="ok" for r in rs))' "$R/$SUITE/$DS-d256.jsonl")
echo "$(date -Is) triton ok / not ok: $n"
[ "$n" = "8 0" ] || { note pod-d "$LEG: ok / not-ok $n; driver stopped"; exit 6; }
upload campaign-v2.9/$DS-$SUITE
msg="$LEG (8 triton cells) done on pod d GPU $GPU at $cv; Hub campaign-v2.9/$DS-$SUITE: $UP"
note pod-d "$msg"; note control "$msg"
finish
