#!/usr/bin/env bash
# C5 Meta at 30 M on the final adapter: codesign-laion30m official half (8 cells) at campaign-v2.8 on pod d GPU 1, Meta's extension built -O3
# (scripts/build_official_o3.sh into /venvs/d-run-v28), one process per sweep; every record's env.official_build must name the -O3 _C.so.
#   GPU=1 setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v28/codesign-laion30m-official.driver.log 2>&1 &
export TAG=campaign-v2.8 LEG=codesign-laion30m-official REPO=/scratch/wt/v28-official PY=/venvs/d-run-v28/bin/python GPU=${GPU:-1}
DS=laion30m SUITE=codesign-laion30m
R=/scratch/campaign-v28/$DS-$SUITE
SO=015acb05c089d9f2cb7dfd13be3075dc00c0e9d7e9cf0c21de8994bde2870332
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
step check check --dataset $DS --dim 256
for sw in c0_domain tags4; do NARROW="--sweep $sw" stream $DS $SUITE "silvertorch|official"; done
n=$($PY -c 'import json,sys; rs=[json.loads(l) for l in open(sys.argv[1])]; print(sum(r["backend"]=="official" and r["status"]=="ok" and (r["env"].get("official_build") or {}).get("so_sha256")==sys.argv[2] for r in rs), sum(r["status"]!="ok" for r in rs))' "$R/$SUITE/$DS-d256.jsonl" $SO)
echo "$(date -Is) official ok with the -O3 so / not ok: $n"
[ "$n" = "8 0" ] || { note pod-d "$LEG: ok-with-O3 / not-ok $n; driver stopped"; exit 6; }
upload campaign-v2.8/$DS-$SUITE
msg="$LEG (8 cells, -O3 official build so_sha256 ${SO:0:12}) done on pod d GPU $GPU at $cv; Hub campaign-v2.8/$DS-$SUITE: $UP"
note pod-d "$msg"; note control "$msg"
finish
