"""Quality of a `bench run` (new code) against (1) the golden cells, C4's criterion
(`quality.oracle` recall@k / ndcg@k within 1e-6 of the golden rows), and (2) the committed D1
records of the same cells taken on the pre-change code (every oracle and held-out metric).

    python golden_compare.py NEW_RESULTS_DIR
"""

import json
import sys
from pathlib import Path

new_dir = Path(sys.argv[1])
TOL = 1e-6
DEFAULT = {"silvertorch": {"n_probe": 24}}


def records(path):
    return [json.loads(line) for line in open(path)] if path.exists() else []


new = {}
for f in (new_dir / "filter").glob("*.jsonl"):
    for r in records(f):
        new[(r["dataset"], r["sweep"], r["algo"], r["backend"], json.dumps(r["params"], sort_keys=True))] = r

print("## vs golden (quality.oracle, tol 1e-6)\n")
print("| golden cell | k | metric | golden | new | |diff| | verdict |")
print("|---|---|---|---|---|---|---|")
for gf in sorted(Path("evaluation/golden").glob("*.json")):
    ds, _, sweep_algo = gf.stem.partition("-d128-")
    sweep, algo, backend = sweep_algo.split("-")
    dataset = {"goodreads": "goodreads", "arxiv": "arxiv"}[ds]
    key = (dataset, sweep, algo, backend, json.dumps(DEFAULT.get(algo, {}), sort_keys=True))
    rec = new.get(key)
    if rec is None:
        print(f"| {gf.stem} | | | | | | no counterpart in today's campaign config |")
        continue
    seen = set()
    for row in json.load(open(gf)):
        k = row["k"]
        if k in seen:
            continue
        seen.add(k)
        for m in (f"recall@{k}", f"ndcg@{k}"):
            if m not in row:
                continue
            a, b = row[m], rec["quality"]["oracle"][m]
            print(f"| {gf.stem} | {k} | {m} | {a:.9f} | {b:.9f} | {abs(a - b):.1e} | {'PASS' if abs(a - b) <= TOL else 'FAIL'} |")

print("\n## vs committed D1 records (pre-change code, same harness)\n")
old = {}
for r in records(Path("evaluation/results/filter/goodreads-d128.jsonl")):
    old[(r["dataset"], r["sweep"], r["algo"], r["backend"], json.dumps(r["params"], sort_keys=True))] = r
print("| cell | metrics compared | max |diff| | max diff metric |")
print("|---|---|---|---|")
for key, r in sorted(new.items()):
    o = old.get(key)
    if o is None or r["sweep"] != "c0_genre":
        continue
    worst, which, n = 0.0, "", 0
    for split in ("oracle", "heldout"):
        for m, v in r["quality"][split].items():
            if m == "n" or m not in o["quality"][split]:
                continue
            d = abs(v - o["quality"][split][m])
            n += 1
            if d >= worst:
                worst, which = d, f"{split}.{m}"
    print(f"| {key[2]} {key[3]} {key[4]} | {n} | {worst:.1e} | {which} |")
