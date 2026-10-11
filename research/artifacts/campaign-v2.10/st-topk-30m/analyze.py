"""ST-TOPK 30 M gate: per (sweep, bs, mode), ours ST-TOPK (B) / ours v2.10 (A), mean of per-round ratios with a t 95 % CI over 3 rounds, ids_sha256
A vs B every round, FLAG where B / A > 1.00; bonus: B / Meta -O3 fp16 and B / Meta int32 (eager; Meta runs in tree A's process).
Run: python analyze.py RESULTS_DIR"""

import json
import sys
from pathlib import Path

import numpy as np

T95 = {2: 4.303, 3: 3.182}
root = Path(sys.argv[1])
cells: dict = {}
for f in sorted(root.glob("r*-[AB]/stt-laion30m/laion30m-d256.jsonl")):
    rnd, tree = f.parts[-3].split("-")
    for line in open(f):
        r = json.loads(line)
        if r["status"] not in ("ok", "partial"):
            sys.exit(f"{f}: {r['status']} {r.get('stage')}")
        arm = (
            f"meta_{r['params'].get('score_path')}"
            if r["backend"] == "official"
            else tree
        )
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
    "| sweep | bs | mode | v2.10 ms | ST-TOPK ms | ST-TOPK / v2.10 | ids A = B | flag | Meta fp16 ms | ST-TOPK / Meta fp16 | Meta int32 ms | ST-TOPK / Meta int32 |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
flags = 0
for (sw, bs, mode), arms in sorted(cells.items()):
    rounds = sorted(arms["B"])
    a = [arms["A"][r]["median_ms"] for r in rounds]
    b = [arms["B"][r]["median_ms"] for r in rounds]
    ids = all(arms["A"][r]["ids_sha256"] == arms["B"][r]["ids_sha256"] for r in rounds)
    mean_ba, ba = ci(np.divide(b, a))
    meta = []
    for sp in ("fp16", "int32"):
        m = arms.get(f"meta_{sp}")
        if m and mode == "eager":
            mm = [m[r]["median_ms"] for r in rounds]
            meta += [f"{np.median(mm):.3f}", ci(np.divide(b, mm))[1]]
        else:
            meta += ["-", "-"]
    flag = "**> 1.00**" if mean_ba > 1.0 else ""
    flags += bool(flag)
    print(
        f"| {sw} | {bs} | {mode} | {np.median(a):.3f} | {np.median(b):.3f} | **{ba}** | {ids} | {flag} | {' | '.join(meta)} |"
    )
print(f"\ncells above 1.00 of v2.10: {flags}")
cmp = root / "scores-compare.txt"
if cmp.exists():
    print(
        "\nscore dumps (first 64 kept queries per sweep; bs 1 on 8):\n"
        + cmp.read_text()
    )
