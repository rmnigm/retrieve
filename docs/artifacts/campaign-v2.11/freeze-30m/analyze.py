"""Freeze-library 30 M gate (e16512f5 vs v2.11), V1 (suite v1g) and SilverTorch (suite stg) cells; from the V1-BS1 gate's analysis: per (dataset, sweep, bs, mode), V1-BS1 (B) / v2.11 (A) mean of per-round median ratios with a t 95 % CI over 3 rounds,
ids_sha256 A = B every round, FLAG where the mean > 1.00; then the score-dump comparison (ids, scores, the bs-1 path per table).
Run: python analyze.py RESULTS_DIR"""

import json
import sys
from pathlib import Path

import numpy as np

T95 = {2: 4.303, 3: 3.182}
root = Path(sys.argv[1])
cells: dict = {}
for f in sorted(root.glob("r*-[AB]/*/*.jsonl")):
    if f.name.endswith("samples.jsonl"):
        continue
    rnd, tree = f.parts[-3].split("-")
    for line in open(f):
        r = json.loads(line)
        if r["status"] not in ("ok", "partial"):
            sys.exit(f"{f}: {r['status']} {r.get('stage')}")
        for p in r["perf"]:
            if p.get("median_ms") is not None:
                cells.setdefault(
                    (r["dataset"], r["sweep"], p["bs"], p["mode"]), {}
                ).setdefault(tree, {})[rnd] = p
print(
    "| dataset | sweep | bs | mode | v2.11 ms | V1-BS1 ms | V1-BS1 / v2.11 | ids A = B | flag |"
)
print("|---|---|---|---|---|---|---|---|---|")
flags = 0
for (ds, sw, bs, mode), t in sorted(cells.items()):
    rounds = sorted(t["B"])
    a = np.array([t["A"][r]["median_ms"] for r in rounds])
    b = np.array([t["B"][r]["median_ms"] for r in rounds])
    x = b / a
    h = T95[len(x) - 1] * x.std(ddof=1) / np.sqrt(len(x))
    ids = all(t["A"][r]["ids_sha256"] == t["B"][r]["ids_sha256"] for r in rounds)
    flag = "**> 1.00**" if x.mean() > 1.0 else ""
    flags += bool(flag)
    print(
        f"| {ds} | {sw} | {bs} | {mode} | {np.median(a):.3f} | {np.median(b):.3f} | **{x.mean():.3f} ({x.mean() - h:.3f}-{x.mean() + h:.3f})** | {ids} | {flag} |"
    )
print(f"\ncells above 1.00 of v2.11: {flags}")
cmp = root / "scores-compare.txt"
if cmp.exists():
    print(
        "\nscore dumps (first 64 kept queries; bs 1 on 8) and the bs-1 path per table:\n"
        + cmp.read_text()
    )
