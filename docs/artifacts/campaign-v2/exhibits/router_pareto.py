"""The router decision (user, 2026-10-10; README.md § router_pareto): the router is kept only if it is on
the recall-latency Pareto front between IVF and exact search on PubMed 10 M.

usage: router_pareto.py OUT TREE [TREE ...] [--dataset pubmed]

Per (code_version, dataset, sweep, bs): the IVF curve (SilverTorch triton along n_probe), the exact
points (V1, V2 triton) and the router points (one per threshold), each at recall_oracle@100 and p50 at
k 100. A router point is dominated if some IVF or exact point, or the IVF curve interpolated
piecewise-linearly in recall between measured points (never extrapolated), has recall >= and latency
<= it, one of them strictly. Latency is each arm's served mode: graph where it has one, else eager
(the router is eager-only); the verdict is also given eager-vs-eager. The router is kept if at least
one of its points is undominated in the served-mode comparison on every sweep of the leg, at bs 1 or
bs 16 (both reported). Same code_version and same tree only: never across boxes.
"""

import collections
import csv
import sys
from pathlib import Path

from load import cv, load, perf, recall

ROUTER, IVF, EXACT = "router", "silvertorch", ("linr_v1_filter_mask", "linr_v2")


def lat(r, bs, mode):
    if mode == "served":
        e = perf(r, bs, 100, "graph") or perf(r, bs, 100, "eager")
    else:
        e = perf(r, bs, 100, mode)
    return e["median_ms"] if e else None


def ivf_at(curve, rec):
    """Latency of the IVF curve at recall `rec`, interpolated between bracketing points, or None."""
    pts = sorted(curve)
    for (r0, t0), (r1, t1) in zip(pts, pts[1:]):
        if r0 <= rec <= r1 and r1 > r0:
            return t0 + (t1 - t0) * (rec - r0) / (r1 - r0)
    exact = [t for r, t in pts if r == rec]
    return min(exact) if exact else None


def dominated(point, others, curve):
    r, t = point
    for ro, to in others:
        if ro >= r and to <= t and (ro > r or to < t):
            return f"by ({ro:.4f}, {to:.3f} ms)"
    ti = ivf_at(curve, r)
    if ti is not None and ti < t:
        return f"by the IVF curve at recall {r:.4f} ({ti:.3f} ms)"
    return ""


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ds = sys.argv[sys.argv.index("--dataset") + 1] if "--dataset" in sys.argv else "pubmed"
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    trees = [a for a in args[1:] if a != ds]
    recs = [
        r
        for r in load(trees)
        if r["dataset"] == ds and r["status"] in ("ok", "partial") and recall(r) is not None
    ]
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        g[(cv(r), r["_tree"], r["filter_kind"], r["sweep"])][r["algo"]].append(r)
    rows, verdict = [], collections.defaultdict(list)
    for (c, tree, fk, sw), arms in sorted(g.items()):
        if ROUTER not in arms or IVF not in arms:
            continue
        for bs in (1, 16):
            for mode in ("served", "eager"):
                curve = [(recall(r), lat(r, bs, mode)) for r in arms[IVF] if lat(r, bs, mode)]
                exact = [
                    (recall(r), lat(r, bs, mode))
                    for a in EXACT
                    for r in arms.get(a, [])
                    if lat(r, bs, mode)
                ]
                kept = False
                for r in arms[ROUTER]:
                    t = lat(r, bs, mode)
                    if t is None:
                        continue
                    why = dominated((recall(r), t), curve + exact, curve)
                    kept |= not why
                    rows.append(
                        (
                            c,
                            tree,
                            fk,
                            sw,
                            bs,
                            mode,
                            f"threshold={r['params'].get('lq_threshold')}",
                            round(recall(r), 4),
                            round(t, 4),
                            "on the front" if not why else f"dominated {why}",
                        )
                    )
                for name, pts in (("IVF", curve), ("exact", exact)):
                    for rr, tt in sorted(pts):
                        rows.append(
                            (c, tree, fk, sw, bs, mode, name, round(rr, 4), round(tt, 4), "")
                        )
                verdict[(c, bs, mode)].append((sw, kept))
    with open(out / "router-pareto.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "code_version",
                "tree",
                "filter_kind",
                "sweep",
                "bs",
                "mode",
                "arm",
                "recall@100",
                "p50_ms",
                "status",
            ]
        )
        w.writerows(rows)
    lines = [
        "# Router decision: on the recall-latency Pareto front? (NOT CITABLE)\n",
        "| code_version | bs | latency mode | sweeps where a router point is on the front | kept |",
        "|---|---|---|---|---|",
    ]
    for (c, bs, mode), v in sorted(verdict.items()):
        on = [sw for sw, k in v if k]
        lines.append(
            f"| {c} | {bs} | {mode} | {', '.join(on) or 'none'} of {len(v)} | {'yes' if len(on) == len(v) else 'no'} |"
        )
    (out / "router-pareto.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
