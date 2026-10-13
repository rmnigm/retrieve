"""k-means test result table: recall at the probe cap (n_probe = n_lists / 4) and the tuned points (smallest n_probe reaching
0.90 / 0.95) per (variant, n_lists, sweep, kind, k), and whether any 0.90 / 0.95 verdict flips against random-10 (A1 / B1, reused)."""

import glob
import json
import sys
from collections import defaultdict

T = sys.argv[1]


def load(pattern):
    return [json.loads(line) for f in glob.glob(pattern) for line in open(f)]


curves = defaultdict(dict)  # (variant, n_lists, sweep, kind, k) -> {n_probe: recall}
for r in load(f"{T}/km-4096/*.jsonl") + load(f"{T}/km-16384/*.jsonl"):
    p = r["params"]
    v = f"{p['kmeans_init']}-{p['n_iter']}"
    for k in (100, 1000):
        curves[(v, p["n_lists"], r["sweep"], r["filter_kind"], k)][p["n_probe"]] = r[
            "quality"
        ]["oracle"][f"recall@{k}"]
ref = [
    r
    for r in load(
        "/scratch/kmeans-test/ref/A1-laion30m/n-curve-q-all/laion30m-d256.jsonl"
    )
    if r["algo"] == "silvertorch"
    and r["backend"] == "official-fork"
    and r["filter_kind"] == "clause"
]
ref += [
    r
    for r in load(
        "/scratch/kmeans-test/ref/A1-bloom/n-bloom-curve-d256/laion30m-d256.jsonl"
    )
    if r["filter_kind"] == "bloom" and r["params"].get("recheck") is False
]
for r in ref:  # file order, newest per key last
    p = r["params"]
    if (
        p.get("n_lists") in (4096, 16384)
        and r["sweep"] in ("c0_domain", "tags4")
        and r["status"] in ("ok", "partial")
    ):
        for k in (100, 1000):
            q = (r["quality"].get("oracle") or {}).get(f"recall@{k}")
            if q is not None:
                curves[("random-10", p["n_lists"], r["sweep"], r["filter_kind"], k)][
                    p["n_probe"]
                ] = q


def tuned(c, t):
    hit = [n for n in sorted(c) if c[n] >= t]
    return hit[0] if hit else None


rows, flips, gaps = [], [], []
for key in sorted(
    curves, key=lambda x: (x[1], x[2], x[3], x[4], x[0] != "random-10", x[0])
):
    v, nl, sw, kind, k = key
    c = curves[key]
    cap = nl // 4
    grid = [n for n in (8 * 2**i for i in range(20)) if n <= cap]
    missing = [n for n in grid if n not in c]
    if missing:
        gaps.append((key, missing))
    row = {
        "variant": v,
        "n_lists": nl,
        "sweep": sw,
        "kind": kind,
        "k": k,
        "cap_recall": c.get(cap),
        "t90": tuned(c, 0.90),
        "t95": tuned(c, 0.95),
    }
    rows.append(row)
by = {(r["variant"], r["n_lists"], r["sweep"], r["kind"], r["k"]): r for r in rows}
for r in rows:
    if r["variant"] == "random-10":
        continue
    b = by.get(("random-10", r["n_lists"], r["sweep"], r["kind"], r["k"]))
    if b is None:
        gaps.append(
            (
                (r["variant"], r["n_lists"], r["sweep"], r["kind"], r["k"]),
                "no random-10 curve",
            )
        )
        continue
    for t in ("t90", "t95"):
        if (r[t] is None) != (b[t] is None):
            flips.append(
                {
                    **{x: r[x] for x in ("variant", "n_lists", "sweep", "kind", "k")},
                    "target": t,
                    "random10": b[t],
                    "variant_tuned": r[t],
                }
            )
json.dump(
    {"rows": rows, "flips": flips, "gaps": [str(g) for g in gaps]},
    open(sys.argv[2], "w"),
    indent=1,
)


def f(x):
    return "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


print(
    "| n_lists | sweep | kind | k | variant | recall at cap | n_probe @0.90 | n_probe @0.95 |\n|---|---|---|---|---|---|---|---|"
)
for r in rows:
    print(
        f"| {r['n_lists']} | {r['sweep']} | {r['kind']} | {r['k']} | {r['variant']} | {f(r['cap_recall'])} | {f(r['t90'])} | {f(r['t95'])} |"
    )
print("FLIPS", flips)
print("GAPS", gaps)
