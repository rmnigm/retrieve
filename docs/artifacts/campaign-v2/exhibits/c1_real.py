"""C1 on real filters (README.md § c1_real): does the per-query pass-rate spread explain where real
V2 / V1 crosses 1? Per real clause sweep, the measured graph V2 / V1 next to predictions from the
same-box, same-code uniform synth V1 / V2 curve evaluated on the sweep's own per-query pass rates
(sidecar `pass_count` / N), drawn as the harness draws its perf pool (H §2.5: `n_pool` batches of
bs queries, with replacement; latency is the median over calls):

- `agg`: the curve at the aggregate (mean) pass rate, what a uniform filter of that p would cost;
- `median_q` (bs 1): the curve at each query's p_q, median over the pool;
- `batch_mean` / `batch_max` (bs 16): the curve at each batch's mean / max p_q, median over the pool.

usage: c1_real.py OUT TREE [TREE ...]
"""

import collections
import csv
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

from load import box, cv, load, pass_p, perf

REAL_SYNTH = {
    "yfcc10m": "yfcc10m-synth",
    "arxiv": "arxiv-synth",
    "laion30m": "laion30m-synth",
    "goodreads": "goodreads-synth",
    "pubmed": None,
}
EXACT = ("linr_v1_filter_mask", "linr_v2")
N_POOL = 4096


def curve(recs, algo, bs):
    """(log p, log p50) points of one synth arm, graph mode."""
    pts = sorted(
        (math.log(pass_p(r)), math.log(e["median_ms"]))
        for r in recs
        if r["algo"] == algo and (e := perf(r, bs, 100, "graph"))
    )
    return [p for p, _ in pts], [y for _, y in pts]


def at(c, p):
    """The curve at pass rate(s) p, log-log linear, clamped to its measured range."""
    xs, ys = c
    return np.exp(np.interp(np.log(np.maximum(p, 1e-9)), xs, ys))


def ratio(v1, v2, bs):
    """Measured V2 / V1 graph p50: paired over windows when interleaved, else the p50 ratio."""
    e1, e2 = perf(v1, bs, 100, "graph"), perf(v2, bs, 100, "graph")
    if not e1 or not e2:
        return None, ""
    g1, g2 = ((x.get("interleave") or {}).get("group") for x in (v1, v2))
    if g1 and g1 == g2 and len(e1["window_medians_ms"]) == len(e2["window_medians_ms"]):
        return st.median(
            b / a for a, b in zip(e1["window_medians_ms"], e2["window_medians_ms"])
        ), "paired"
    return e2["median_ms"] / e1["median_ms"], "p50"


def main():
    out = Path(sys.argv[1])
    recs = [
        r
        for r in load(sys.argv[2:])
        if r["status"] == "ok"
        and r["backend"] == "triton"
        and r["algo"] in EXACT
        and r["filter_kind"] == "clause"
        and r["seed"] == 0
    ]
    trees = {Path(t).name: Path(t) for t in sys.argv[2:]}
    synth = collections.defaultdict(list)
    real = collections.defaultdict(dict)
    for r in recs:
        if r["dataset"] in REAL_SYNTH.values():
            synth[(r["dataset"], cv(r), box(r))].append(r)
        elif r["dataset"] in REAL_SYNTH:
            real[(r["dataset"], cv(r), box(r), r["sweep"])][r["algo"]] = r
    rng = np.random.default_rng(0)
    rows = []
    for (ds, c, bx, sw), arms in sorted(real.items()):
        if set(arms) != set(EXACT) or not arms["linr_v2"].get("per_query"):
            continue
        v1, v2 = arms["linr_v1_filter_mask"], arms["linr_v2"]
        z = np.load(trees[v2["_tree"]] / v2["per_query"])
        pq = z["pass_count"] / v2["n_items"]
        s = synth.get((REAL_SYNTH[ds], c, bx), [])
        for bs in (1, 16):
            meas, how = ratio(v1, v2, bs)
            if meas is None:
                continue
            row = {
                "dataset": ds,
                "code_version": c,
                "box": bx,
                "sweep": sw,
                "bs": bs,
                "p_agg": round(float(pq.mean()), 5),
                "p_median_q": round(float(np.median(pq)), 5),
                "p10_q": round(float(np.quantile(pq, 0.1)), 5),
                "p90_q": round(float(np.quantile(pq, 0.9)), 5),
                "cv_q": round(float(pq.std() / pq.mean()), 2) if pq.mean() else "",
                "v1_ms": round(perf(v1, bs, 100, "graph")["median_ms"], 3),
                "v2_ms": round(perf(v2, bs, 100, "graph")["median_ms"], 3),
                "measured": round(meas, 3),
                "measured_how": how,
                "synth": f"{REAL_SYNTH[ds]} same box" if s else "none at this code / box",
            }
            if s:
                c1, c2 = curve(s, "linr_v1_filter_mask", bs), curve(s, "linr_v2", bs)
                batches = pq[rng.integers(0, len(pq), (N_POOL, bs))]
                v1_pred = float(np.median(at(c1, batches.mean(1))))
                row["agg"] = round(float(at(c2, pq.mean()) / at(c1, pq.mean())), 3)
                if bs == 1:
                    row["median_q"] = round(float(np.median(at(c2, batches[:, 0]))) / v1_pred, 3)
                else:
                    row["batch_mean"] = round(
                        float(np.median(at(c2, batches.mean(1)))) / v1_pred, 3
                    )
                    row["batch_max"] = round(float(np.median(at(c2, batches.max(1)))) / v1_pred, 3)
                row["v1_pred_ms"] = round(v1_pred, 3)
            rows.append(row)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "c1-real.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
