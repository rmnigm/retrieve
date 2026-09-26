"""Quality of two `bench run`s of the golden cell set — the pre-change tree (BASE) and this
branch (NEW), both taken on this box — against the golden cells (`quality.oracle` recall@k /
ndcg@k, C4's 1e-6 criterion), and NEW against BASE on every oracle and held-out metric.

    python golden_compare.py BASE_RESULTS_DIR NEW_RESULTS_DIR
"""

import json
import sys
from pathlib import Path

TOL = 1e-6
DEFAULT = {"silvertorch": {"n_probe": 24}}


def load(d):
    out = {}
    for f in (Path(d) / "filter").glob("*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            key = (r["dataset"], r["sweep"], r["algo"], r["backend"])
            out[key + (json.dumps(r["params"], sort_keys=True),)] = r
    return out


base, new = load(sys.argv[1]), load(sys.argv[2])

print("## vs golden (quality.oracle, tol 1e-6)\n")
print("| golden cell | metric | golden | base | new | new verdict |")
print("|---|---|---|---|---|---|")
for gf in sorted(Path("evaluation/golden").glob("*.json")):
    ds, _, sweep_algo = gf.stem.partition("-d128-")
    sweep, algo, backend = sweep_algo.split("-")
    key = (ds, sweep, algo, backend, json.dumps(DEFAULT.get(algo, {}), sort_keys=True))
    if key not in new:
        print(f"| {gf.stem} | | | | | no counterpart in today's suites |")
        continue
    seen = set()
    for row in json.load(open(gf)):
        if row["k"] in seen:
            continue
        seen.add(row["k"])
        for m in (f"recall@{row['k']}", f"ndcg@{row['k']}"):
            g, b, n = row[m], base[key]["quality"]["oracle"][m], new[key]["quality"]["oracle"][m]
            verdict = "PASS" if abs(g - n) <= TOL else f"{abs(g - n):.1e}"
            print(f"| {gf.stem} | {m} | {g:.9f} | {b:.9f} | {n:.9f} | {verdict} |")

print("\n## new vs base (every oracle and held-out metric)\n")
print("| cell | metrics | max |new − base| | at |")
print("|---|---|---|---|")
for key in sorted(new):
    worst, at, n = 0.0, "", 0
    for split in ("oracle", "heldout"):
        for m, v in new[key]["quality"][split].items():
            if m == "n":
                continue
            d = abs(v - base[key]["quality"][split][m])
            n += 1
            if d >= worst:
                worst, at = d, f"{split}.{m}"
    print(f"| {key[0]} {key[2]} {key[3]} {key[4]} | {n} | {worst:.1e} | {at} |")
