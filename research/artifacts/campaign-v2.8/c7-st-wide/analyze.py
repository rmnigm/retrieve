"""ST-WIDE cross-tree at 30 M: per cell and round, ours v2.8 (A), ours st-wide (B) and Meta -O3 (A, interleaved with ours v2.8) eager
median ms; ratios B/Meta, A/Meta, B/A per round, their mean with a t 95 % CI over the 3 rounds; the --profile kernel split (scorer / topk /
epilogue / other µs, launches) per arm from round 1. Run: python analyze.py RESULTS_DIR"""

import json
import sys
from pathlib import Path

import numpy as np

T95 = {2: 4.303, 3: 3.182}  # t_{0.975, n-1} for n = 3, 4 rounds
root = Path(sys.argv[1])
cells: dict = {}
for f in sorted(root.glob("r*-[AB]/c7w-laion30m/laion30m-d256.jsonl")):
    rnd, tree = f.parts[-3].split("-")
    for line in open(f):
        r = json.loads(line)
        if r["status"] not in ("ok", "partial"):
            sys.exit(f"{f}: {r['status']} {r.get('stage')}")
        arm = "meta" if r["backend"] == "official" else f"ours_{tree}"
        for p in r["perf"]:
            if p.get("median_ms") is not None and p["mode"] == "eager":
                cells.setdefault((r["sweep"], p["bs"]), {}).setdefault(arm, {})[rnd] = p


def ci(xs):
    xs = np.asarray(xs)
    h = T95[len(xs) - 1] * xs.std(ddof=1) / np.sqrt(len(xs))
    return f"{xs.mean():.2f} ({xs.mean() - h:.2f}-{xs.mean() + h:.2f})"


def split(p):
    ks = p.get("kernel_scopes") or {}
    us = " / ".join(
        f"{ks.get(s, {}).get('us', 0):.0f}"
        for s in ("scorer", "topk", "epilogue", "other")
    )
    return f"{us} ({p.get('kernels_calls')})"


print(
    "| sweep | bs | ours v2.8 ms | ours st-wide ms | Meta ms | st-wide / Meta | v2.8 / Meta | st-wide / v2.8 | scorer / topk / epi / other µs (launches): v2.8 · st-wide · Meta |"
)
print("|---|---|---|---|---|---|---|---|---|")
for (sw, bs), arms in sorted(cells.items()):
    rounds = sorted(arms["ours_B"])
    a = [arms["ours_A"][r]["median_ms"] for r in rounds]
    b = [arms["ours_B"][r]["median_ms"] for r in rounds]
    m = [arms["meta"][r]["median_ms"] for r in rounds]
    r1 = rounds[0]
    print(f"| {sw} | {bs} | {np.median(a):.3f} | {np.median(b):.3f} | {np.median(m):.3f} | **{ci(np.divide(b, m))}** | {ci(np.divide(a, m))} | "
          f"{ci(np.divide(b, a))} | {split(arms['ours_A'][r1])} · {split(arms['ours_B'][r1])} · {split(arms['meta'][r1])} |")  # fmt: skip
