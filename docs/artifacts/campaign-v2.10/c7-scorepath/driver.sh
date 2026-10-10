#!/usr/bin/env bash
# C7 30 M score-path cell (controller 2026-10-10): is Meta's bs-64 lead its fp16 top-k keys? ours triton vs Meta -O3 at score_path fp16 and int32,
# one interleave group per sweep {by: [backend, score_path]}, ONE process, --interleave --profile; laion30m d256 bloom partial, n_lists 16384,
# n_probe 128, bs 16 / 64, eager + graph (official eager only), k 100, seed 0; 3 rounds = the group's windows. campaign-v2.10 (tag dc0af407,
# library a3bec4a5) from a worktree at the tag, venv with Meta's -O3 build; laion30m oracle blobs not moved (laion reads them).
#   GPU=1 setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v210/c7-scorepath.driver.log 2>&1 &
export TAG=campaign-v2.10 LEG=c7-scorepath REPO=/scratch/wt/v210-tag PY=/venvs/d-run-v210/bin/python GPU=${GPU:-1}
DS=laion30m SUITE=c7-scorepath
R=/scratch/campaign-v210/$DS-$SUITE
CFG=/scratch/v210/c7-scorepath-config
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
rm -rf "$CFG" && cp -r "$REPO/evaluation/config" "$CFG"
printf 'c7-scorepath:\n  datasets: [laion30m]\n  dims: [256]\n  filter_kinds: [bloom]\n  ks: [100]\n  batch_sizes: [16, 64]\n  seeds: [0]\n  sweeps:\n    laion30m: [c0_domain, tags4]\n  interleave: [{by: [backend, score_path]}]\n  arms:\n    - {algo: silvertorch, backends: [triton], build: {bloom_path: [partial], n_lists: [16384]}, query: {n_probe: [128]}}\n    - {algo: silvertorch, backends: [official], build: {bloom_path: [partial], n_lists: [16384], score_path: [fp16, int32]}, query: {n_probe: [128]}}\n' >> "$CFG/suites.yaml"
step check check --dataset $DS --dim 256
NARROW="--config-dir $CFG --profile" stream $DS $SUITE "silvertorch|triton official"
n=$($PY -c 'import json,sys; rs=[json.loads(l) for l in open(sys.argv[1])]; print(sum(r["status"]=="ok" and (r["backend"]!="official" or (r["env"].get("official_build") or {}).get("nvcc_append_flags")=="-O3 -Xcompiler -O3") for r in rs), sum(r["status"]!="ok" for r in rs))' "$R/$SUITE/$DS-d256.jsonl")
echo "$(date -Is) ok (official at -O3) / not ok: $n"
[ "$n" = "6 0" ] || { note pod-d "$LEG: ok / not-ok $n; driver stopped"; exit 6; }
upload campaign-v2.10/$DS-$SUITE
msg="$LEG (6 cells) done on pod d GPU $GPU at $cv; Hub campaign-v2.10/$DS-$SUITE: $UP"
note pod-d "$msg"; note control "$msg"
finish
