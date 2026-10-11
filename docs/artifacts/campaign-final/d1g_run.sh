#!/usr/bin/env bash
# D1-G's quality subset on pod d (README.md): the picks of `d1g_quality.py plan` split over GPU 0 / GPU 1 by recorded
# build time (longest first, each to the less loaded GPU), each pick a `bench run ... --skip-perf --force` into a scratch
# tree per GPU, one job per GPU under its /scratch/gpu<N>.lock, from the campaign-v2.12 worktree with the campaign-final
# config (same library e16512f5; its suites.yaml adds the extension suites). Official picks use the -O3 venv.
# usage: d1g_run.sh PICKS.json OUT_DIR
set -euo pipefail
PICKS=$1 OUT=$2
WT=/scratch/wt/final/evaluation CFG=/scratch/wt/final-ext/evaluation/config
mkdir -p "$OUT/logs"
python3 - "$PICKS" "$OUT" <<'PY'
import json, sys
picks = [p for p in json.load(open(sys.argv[1])) if "error" not in p]
load, q = [0.0, 0.0], [[], []]
for p in sorted(picks, key=lambda p: -p["build_s"]):
    g = 0 if load[0] <= load[1] else 1
    load[g] += p["build_s"] + 30
    q[g].append(p)
for g in (0, 1):
    with open(f"{sys.argv[2]}/queue{g}.txt", "w") as fh:
        for p in q[g]:
            algos = " ".join(f"--algo {a}" for a in p["algos"])
            be = f" --backend {p['backend']}" if p["backend"] else ""
            venv = "/venvs/final-o3" if p["official"] else "/venvs/final"
            fh.write(f"{p['leg']}\t{venv}\t--dataset {p['dataset']} --dim {p['dim']} --suite {p['suite']} {algos}{be} "
                     f"--filter-kind {p['filter_kind']} --sweep {p['sweep']} --seed {p['seed']}\n")
    print(f"GPU {g}: {len(q[g])} picks, {load[g] / 3600:.2f} h of builds")
PY
run() {
  local g=$1
  while IFS=$'\t' read -r leg venv args; do
    t0=$(date +%s)
    ( flock 9
      cd "$WT" && CUDA_VISIBLE_DEVICES=$g TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/d1g-gpu$g \
        "$venv/bin/python" -m bench.cli run $args --skip-perf --force --config-dir "$CFG" --out "$OUT/gpu$g" \
        > "$OUT/logs/$leg.log" 2>&1 ) 9>/scratch/gpu$g.lock
    echo "$(date -Is) gpu$g $leg rc=$? s=$(( $(date +%s) - t0 ))" >> "$OUT/progress.log"
  done < "$OUT/queue$g.txt"
}
run 0 & run 1 & wait
echo "$(date -Is) done" >> "$OUT/progress.log"
