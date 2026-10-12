#!/usr/bin/env bash
# AFTER-QUEUE 1: arxiv-synth SilverTorch at campaign-v2.9 on pod d GPU 1 (same box as pod d's v2.5 V-AX-SYNTH): the SYNTH-TRIM grid's SilverTorch
# child, 32 cells (triton clause 7 rates x n_probe {24, 64, 256, 1024}; triton + official bloom n_probe 24 at p 0.01 and 1), bs 1 + 16, seed 0, one
# process (H-ARMFREE), triton + official in one stream (the parity spill). Own worktree on staging (library e8958bd2), venv with Meta's -O3 build.
# Hub campaign-v2.9/arxiv-synth-synth (appends to the tree the V1 + V2 re-time does not cover at v2.9).
#   GPU=1 setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v29/ax-synth-st.driver.log 2>&1 &
export TAG=campaign-v2.9 LEG=ax-synth-st REPO=/scratch/wt/v29-axsynth PY=/venvs/d-run-v29-ax/bin/python GPU=${GPU:-1}
DS=arxiv-synth SUITE=synth
R=/scratch/campaign-v29/$DS-$SUITE
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
old_oracles $DS
step check check --dataset $DS --dim 128
step oracle oracle --dataset $DS --suite $SUITE --dim 128
stream $DS $SUITE "silvertorch|triton official"
n=$($PY -c 'import json,sys; rs=[json.loads(l) for l in open(sys.argv[1])]; print(sum(r["status"]=="ok" and (r["backend"]!="official" or (r["env"].get("official_build") or {}).get("nvcc_append_flags")=="-O3 -Xcompiler -O3") for r in rs), sum(r["status"]!="ok" for r in rs))' "$R/$SUITE/$DS-d128.jsonl")
echo "$(date -Is) ok (official at -O3) / not ok: $n"
[ "$n" = "32 0" ] || { note pod-d "$LEG: ok / not-ok $n; driver stopped"; exit 6; }
upload campaign-v2.9/$DS-$SUITE
msg="$LEG (32 SilverTorch cells, official -O3) done on pod d GPU $GPU at $cv; Hub campaign-v2.9/$DS-$SUITE: $UP"
note pod-d "$msg"; note control "$msg"
finish
