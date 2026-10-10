"""ST-WIDE-2 30 M gate: per (sweep, bs, mode), ours ST-WIDE-2 (B) / ours v2.9 (A) and ours / Meta -O3 (eager; Meta is not capturable), the mean of
the per-round ratios with a t 95 % CI over the 3 rounds, ids_sha256 A vs B (every round), and a FLAG where B / A > 1.00.
Run: python analyze.py RESULTS_DIR"""

import json
import sys
from pathlib import Path

import numpy as np

T95 = {2: 4.303, 3: 3.182}
root = Path(sys.argv[1])
cells: dict = {}
for f in sorted(root.glob("r*-[AB]/stw2-laion30m/laion30m-d256.jsonl")):
    rnd, tree = f.parts[-3].split("-")
    for line in open(f):
        r = json.loads(line)
        if r["status"] not in ("ok", "partial"):
            sys.exit(f"{f}: {r['status']} {r.get('stage')}")
        arm = "meta" if r["backend"] == "official" else tree
        for p in r["perf"]:
            if p.get("median_ms") is not None:
                cells.setdefault((r["sweep"], p["bs"], p["mode"]), {}).setdefault(
                    arm, {}
                )[rnd] = p


def ci(xs):
    xs = np.asarray(xs)
    h = T95[len(xs) - 1] * xs.std(ddof=1) / np.sqrt(len(xs))
    return xs.mean(), f"{xs.mean():.3f} ({xs.mean() - h:.3f}-{xs.mean() + h:.3f})"


print(
    "| sweep | bs | mode | v2.9 ms | ST-WIDE-2 ms | Meta ms | ST-WIDE-2 / v2.9 | ST-WIDE-2 / Meta | v2.9 / Meta | ids A = B | flag |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|")
flags = 0
for (sw, bs, mode), arms in sorted(cells.items()):
    rounds = sorted(arms["B"])
    a = [arms["A"][r]["median_ms"] for r in rounds]
    b = [arms["B"][r]["median_ms"] for r in rounds]
    ids = all(arms["A"][r]["ids_sha256"] == arms["B"][r]["ids_sha256"] for r in rounds)
    mean_ba, ba = ci(np.divide(b, a))
    if "meta" in arms and mode == "eager":
        m = [arms["meta"][r]["median_ms"] for r in rounds]
        bm, am, mm = (
            ci(np.divide(b, m))[1],
            ci(np.divide(a, m))[1],
            f"{np.median(m):.3f}",
        )
    else:
        bm = am = mm = "-"
    flag = "**> 1.00**" if mean_ba > 1.0 else ""
    flags += bool(flag)
    print(
        f"| {sw} | {bs} | {mode} | {np.median(a):.3f} | {np.median(b):.3f} | {mm} | **{ba}** | {bm} | {am} | {ids} | {flag} |"
    )
print(f"\ncells above 1.00 of v2.9: {flags}")
