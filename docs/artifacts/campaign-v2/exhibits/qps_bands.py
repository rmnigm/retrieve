"""Idea #6a, Big-ANN-style QPS at recall_oracle@100 = 0.95 per selectivity band (README.md § qps_bands).

usage: qps_bands.py OUT TREE [TREE ...]

Two flavours, never mixed:
- per sweep: every (dataset, filter_kind, sweep, arm, n_lists) recall curve of the `deep` suite (along
  n_probe / candidate_pool) is put in the band of its sweep's pass rate; QPS at 0.95 = bs · 1000 /
  latency at 0.95 (`stats.at_recall`, graph, official eager); exact arms (V1, V2 of the `filter`
  suite at the same code_version) at their own latency;
- per query: on suites with per-query sidecars and an n_probe sweep (`codesign`), each band's recall is the
  mean over the sweep's queries whose own pass rate is in the band; latency is the cell's.
A curve whose cheapest point already exceeds 0.95 reports that point's QPS as a lower bound (">=").
"""

import collections
import csv
import statistics as st
import sys
from pathlib import Path

import numpy as np

from load import ALGO, box, cv, load, perf, recall

from bench import stats

BANDS = [
    (0.0, 0.01, "p < 0.01"),
    (0.01, 0.1, "0.01-0.1"),
    (0.1, 0.5, "0.1-0.5"),
    (0.5, 1.01, "> 0.5"),
]
TARGET = 0.95


def band(p):
    return next(name for lo, hi, name in BANDS if lo <= p < hi)


def lat(r, bs):
    e = perf(r, bs, 100, "graph") or perf(r, bs, 100, "eager")
    return (e["median_ms"], e["mode"]) if e else (None, None)


def qps_at(points, bs):
    """points: [(recall, latency_ms, label)] -> (QPS string, bracket)"""
    res = stats.at_recall(points, TARGET)
    if res["latency"] is not None:
        return f"{bs * 1000 / res['latency']:.0f}", "-".join(map(str, res["bracket"]))
    if res["reason"].startswith("first point"):
        r0, t0, l0 = min(points, key=lambda x: x[1])
        return f">= {bs * 1000 / t0:.0f}", f"{l0} (recall {r0:.3f})"
    return "not reached", f"max recall {max(p[0] for p in points):.3f}"


def per_sweep(recs, rows):
    curves = collections.defaultdict(list)
    for r in recs:
        if (
            r["suite"] != "deep"
            or r["status"] not in ("ok", "partial")
            or recall(r) is None
        ):
            continue
        x = r["params"].get(
            "n_probe",
            r["params"].get("candidate_pool", r["params"].get("candidate_pool_frac")),
        )
        nl = r["params"].get("n_lists", "")
        curves[
            (
                cv(r),
                r["dataset"],
                r["filter_kind"],
                r["sweep"],
                f"{ALGO[r['algo']]}/{r['backend']}",
                nl,
                box(r),
            )
        ].append((r, x))
    for (c, ds, fk, sw, arm, nl, bx), lst in sorted(
        curves.items(), key=lambda x: tuple(map(str, x[0]))
    ):
        p = st.median(r["pass_rate"] for r, _ in lst)
        for bs in (1, 16):
            by = collections.defaultdict(list)
            for r, x in lst:
                t, mode = lat(r, bs)
                if t:
                    by[x].append((recall(r), t, mode))
            pts = [
                (st.median(v[0] for v in vs), st.median(v[1] for v in vs), f"{x}")
                for x, vs in sorted(by.items())
            ]
            if not pts:
                continue
            q, br = qps_at(pts, bs)
            rows.append(
                (
                    "per-sweep",
                    c,
                    ds,
                    fk,
                    band(p),
                    sw,
                    round(p, 4),
                    f"{arm} n_lists={nl}",
                    bs,
                    q,
                    br,
                    bx,
                )
            )
    # exact arms of the `filter` suite at the same code_version and box (timings never cross boxes)
    have = {(row[1], row[2], row[11]) for row in rows}
    g = collections.defaultdict(list)
    for r in recs:
        if (
            r["suite"] == "filter"
            and r["algo"] in ("linr_v1_filter_mask", "linr_v2")
            and r["status"] == "ok"
            and (cv(r), r["dataset"], box(r)) in have
        ):
            g[
                (
                    cv(r),
                    r["dataset"],
                    r["filter_kind"],
                    r["sweep"],
                    f"{ALGO[r['algo']]}/{r['backend']}",
                    box(r),
                )
            ].append(r)
    for (c, ds, fk, sw, arm, bx), rs in sorted(g.items()):
        p = st.median(r["pass_rate"] for r in rs)
        for bs in (1, 16):
            ts = [lat(r, bs)[0] for r in rs if lat(r, bs)[0]]
            rc = st.median(recall(r) for r in rs)
            if ts:
                rows.append(
                    (
                        "per-sweep",
                        c,
                        ds,
                        fk,
                        band(p),
                        sw,
                        round(p, 4),
                        f"{arm} (exact, recall {rc:.4f})",
                        bs,
                        f"{bs * 1000 / st.median(ts):.0f}",
                        "exact",
                        bx,
                    )
                )


def per_query(recs, trees, rows):
    curves = collections.defaultdict(list)
    for r in recs:
        if r["suite"] != "codesign" or not r.get("per_query") or r["status"] != "ok":
            continue
        curves[
            (
                cv(r),
                r["dataset"],
                r["filter_kind"],
                r["sweep"],
                f"{ALGO[r['algo']]}/{r['backend']}",
                r["params"].get("bloom_path"),
                r["params"].get("n_lists"),
            )
        ].append(r)
    for (c, ds, fk, sw, arm, bp, nl), rs in sorted(
        curves.items(), key=lambda x: tuple(map(str, x[0]))
    ):
        tree = next(t for t in trees if t.name == rs[0]["_tree"])
        for b_lo, b_hi, bname in BANDS:
            for bs in (1, 16):
                by = collections.defaultdict(list)
                for r in rs:
                    z = np.load(tree / r["per_query"])
                    pq = z["pass_count"] / r["n_items"]
                    m = (pq >= b_lo) & (pq < b_hi) & ~np.isnan(z["recall_oracle@100"])
                    t, mode = lat(r, bs)
                    if m.sum() >= 20 and t:
                        by[r["params"]["n_probe"]].append(
                            (float(z["recall_oracle@100"][m].mean()), t, int(m.sum()))
                        )
                pts = [
                    (st.median(v[0] for v in vs), st.median(v[1] for v in vs), f"{x}")
                    for x, vs in sorted(by.items())
                ]
                if pts:
                    n = st.median(v[2] for vs in by.values() for v in vs)
                    q, br = qps_at(pts, bs)
                    rows.append(
                        (
                            "per-query",
                            c,
                            ds,
                            fk,
                            bname,
                            sw,
                            f"{int(n)} queries",
                            f"{arm} bloom_path={bp} n_lists={nl}",
                            bs,
                            q,
                            br,
                            box(rs[0]),
                        )
                    )


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    trees = [Path(t) for t in sys.argv[2:]]
    recs = load(trees)
    rows = []
    per_sweep(recs, rows)
    per_query(recs, trees, rows)
    with open(out / "qps-bands.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "flavour",
                "code_version",
                "dataset",
                "filter_kind",
                "band",
                "sweep",
                "pass_rate_or_n",
                "arm",
                "bs",
                "qps_at_0.95",
                "bracket",
                "box",
            ]
        )
        w.writerows(rows)
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
