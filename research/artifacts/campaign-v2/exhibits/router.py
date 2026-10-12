"""Idea #1, a pass-rate router (README.md § router): a counterfactual per-query policy from the sidecars.

usage: router.py OUT TREE [TREE ...]

Route a query to exact V2 when its own pass rate p_q = pass_count / N is below t, else to IVF (SilverTorch
Triton, n_probe 24). Recall per query comes from the sidecars (IVF: its record; V1 / V2: their own sidecar
where the sweep has one, else 1.0). Latency is a per-cell (batch) p50 in graph mode, k 100, looked up by
p_q on the arm's latency-vs-p curve (`synth` legs; PubMed: d1's real sweeps), log-linear between points
and clamped at the ends. The estimate assumes routing costs ~0 and that each arm runs its own batches.
"""

import collections
import csv
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from load import cv, load, pass_p, perf  # noqa: E402

TARGET = 0.95
# real query sets with an IVF per-query recall: (dataset, suite, IVF params subset)
QUERY_SETS = {
    "goodreads": ("filter", {"n_probe": 24}),
    "arxiv": ("bloomwidth", {"m_bits": 1024, "k_hash": 5}),
    "pubmed": ("bloomwidth", {"m_bits": 1024, "k_hash": 5}),
}
# where each dataset's latency curves come from (dataset of the records, suite)
CURVES = {
    "goodreads": ("goodreads-synth", "synth"),
    "arxiv": ("arxiv-synth", "synth"),
    "pubmed": ("pubmed", "filter"),
}
# IVF (SilverTorch Triton, n_probe 24) latency source per dataset: (dataset, suite, filter_kind, params subset)
IVF_LAT = {
    "goodreads": ("goodreads", "h2h", None, {"n_probe": 24}),
    "arxiv": ("arxiv", "h2h", None, {"n_probe": 24}),
    "pubmed": ("pubmed", "bloomwidth-timed", "bloom", {"m_bits": 1024, "k_hash": 5}),
}


def match(r, sub):
    return all(r["params"].get(k) == v for k, v in sub.items())


def tree_of(r, trees):
    return next(t for t in trees if t.name == r["_tree"])


def per_query(recs, trees):
    """{(dataset, filter_kind, sweep): {"p": array, "ivf": array, "v2": array|None, "v1": array|None, "cv": str}}"""
    out = {}
    for ds, (suite, sub) in QUERY_SETS.items():
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in recs:
            if (
                r["dataset"] != ds
                or r["suite"] != suite
                or not r.get("per_query")
                or r["status"] != "ok"
            ):
                continue
            if (
                r["algo"] == "silvertorch"
                and r["backend"] == "triton"
                and match(r, sub)
                and not r["params"].get("compile")
            ):
                arm = "ivf"
            elif r["algo"] == "linr_v2" and r["backend"] == "triton":
                arm = "v2"
            elif r["algo"] == "linr_v1_filter_mask" and r["backend"] == "triton":
                arm = "v1"
            else:
                continue
            z = np.load(tree_of(r, trees) / r["per_query"])
            g[(ds, r["filter_kind"], r["sweep"])][arm].append(
                (z["rows"], z["pass_count"] / r["n_items"], z["recall_oracle@100"], cv(r))
            )
        for key, arms in g.items():
            if "ivf" not in arms:
                continue
            rows0, p0, _, c = arms["ivf"][0]
            d = {"p": p0, "cv": c}
            for arm, lst in arms.items():
                assert all(np.array_equal(x[0], rows0) for x in lst), key
                d[arm] = np.nanmean(
                    np.stack([x[2] for x in lst]), axis=0
                )  # mean over seeds (builds)
            out[key] = d
    return out


def curves(recs):
    """{dataset: {(arm, filter_kind, bs): [(p, ms)]}} for V1 / V2 Triton, medians over seeds, graph k 100."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for ds, (src, suite) in CURVES.items():
        g = collections.defaultdict(list)
        for r in recs:
            if (
                r["dataset"] != src
                or r["suite"] != suite
                or r["backend"] != "triton"
                or r["status"] != "ok"
            ):
                continue
            if r["algo"] not in ("linr_v1_filter_mask", "linr_v2"):
                continue
            p = pass_p(r) if suite == "synth" else r["pass_rate"]
            for bs in (1, 16):
                e = perf(r, bs, 100, "graph")
                if e:
                    g[
                        (
                            "v1" if r["algo"] == "linr_v1_filter_mask" else "v2",
                            r["filter_kind"],
                            bs,
                            round(p, 6),
                        )
                    ].append((e["median_ms"], cv(r)))
        for (arm, fk, bs, p), v in g.items():
            out[ds][(arm, fk, bs)].append((p, st.median(x for x, _ in v), v[0][1]))
        for k in out[ds]:
            out[ds][k].sort()
    return out


def ivf_latency(recs):
    out = {}
    for ds, (src, suite, fk, sub) in IVF_LAT.items():
        for bs in (1, 16):
            v = [
                (perf(r, bs, 100, "graph") or {}).get("median_ms")
                for r in recs
                if r["dataset"] == src
                and r["suite"] == suite
                and r["algo"] == "silvertorch"
                and r["backend"] == "triton"
                and (fk is None or r["filter_kind"] == fk)
                and match(r, sub)
                and r["status"] == "ok"
            ]
            v = [x for x in v if x]
            fks = {r["filter_kind"] for r in recs if r["dataset"] == src and r["suite"] == suite}
            if v:
                out[(ds, bs)] = (
                    st.median(v),
                    sorted({cv(r) for r in recs if r["dataset"] == src and r["suite"] == suite}),
                    fks,
                )
    return out


def interp(pts, p):
    """Log-linear in p between curve points, clamped."""
    xs = [math.log10(max(x, 1e-7)) for x, _, _ in pts]
    ys = [y for _, y, _ in pts]
    return float(np.interp(math.log10(max(p, 1e-7)), xs, ys))


def table(lst, cur_ds, liv, bs):
    """Per-query arrays pooled over a dataset's query sets: p, IVF recall, exact recall and exact latency
    (the cheaper of V1 and V2 at the query's own p; recall of the arm chosen)."""
    P, RI, RE, LE, ARM = [], [], [], [], []
    for fk, sw, d in lst:
        c1 = cur_ds.get(("v1", fk, bs)) or cur_ds.get(("v1", "clause", bs))
        c2 = cur_ds.get(("v2", fk, bs)) or cur_ds.get(("v2", "clause", bs))
        ok = ~np.isnan(d["ivf"])
        p = d["p"][ok]
        l1, l2 = np.array([interp(c1, x) for x in p]), np.array([interp(c2, x) for x in p])
        v1 = d.get("v1", np.ones_like(d["ivf"]))[ok]
        v2 = d.get("v2", np.ones_like(d["ivf"]))[ok]
        use2 = l2 <= l1
        P.append(p)
        RI.append(d["ivf"][ok])
        RE.append(np.where(use2, np.nan_to_num(v2, nan=1.0), np.nan_to_num(v1, nan=1.0)))
        LE.append(np.minimum(l1, l2))
        ARM.append(use2)
    t = {k: np.concatenate(v) for k, v in zip(("p", "ri", "re", "le", "v2"), (P, RI, RE, LE, ARM))}
    t["li"] = np.full_like(t["p"], liv)
    return t


def route(t, thr):
    """Exact (cheaper of V1 / V2) if p_q < thr, else IVF: (mean recall, mean latency, share exact)."""
    ex = t["p"] < thr
    return (
        float(np.where(ex, t["re"], t["ri"]).mean()),
        float(np.where(ex, t["le"], t["li"]).mean()),
        float(ex.mean()),
    )


def oracle(t, target):
    """Cheapest per-query assignment reaching mean recall >= target: start all-IVF, switch the queries
    with the best recall gain per extra ms to exact first (greedy, exact for this knapsack's LP bound)."""
    gain, cost = t["re"] - t["ri"], np.maximum(t["le"] - t["li"], 1e-9)
    order = np.argsort(-gain / cost)
    need = target * len(gain) - t["ri"].sum()
    take = np.zeros(len(gain), bool)
    acc = 0.0
    for i in order:
        if acc >= need or gain[i] <= 0:
            break
        take[i] = True
        acc += gain[i]
    return (
        float(np.where(take, t["re"], t["ri"]).mean()),
        float(np.where(take, t["le"], t["li"]).mean()),
        float(take.mean()),
    )


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    trees = [Path(t) for t in sys.argv[2:]]
    recs = load(trees)
    pq, cur, ivl = per_query(recs, trees), curves(recs), ivf_latency(recs)
    # goodreads has both filter kinds over the same queries: keep clause, so no query counts twice
    pq = {k: v for k, v in pq.items() if not (k[0] == "goodreads" and k[1] == "bloom")}
    thrs = [0.0] + list(10 ** np.linspace(-6, 0, 61)) + [1.01]
    sets = collections.defaultdict(list)
    for (ds, fk, sw), d in pq.items():
        sets[ds].append((fk, sw, d))
    tabs, best, rows = {}, {}, []
    figs = {bs: plt.subplots(figsize=(6.4, 4.4)) for bs in (1, 16)}
    for ds, lst in sorted(sets.items()):
        for bs in (1, 16):
            if (ds, bs) not in ivl:
                continue
            liv, ivcv, _ = ivl[(ds, bs)]
            t = tabs[(ds, bs)] = table(lst, cur[ds], liv, bs)
            curve = [(x, *route(t, x)) for x in thrs]
            feas = [c for c in curve if c[1] >= TARGET]
            pick = min(feas, key=lambda c: c[2]) if feas else max(curve, key=lambda c: c[1])
            best[(ds, bs)] = pick[0]
            cvs = sorted(
                {d["cv"] for _, _, d in lst}
                | set(ivcv)
                | {c for k, pts in cur[ds].items() if k[2] == bs for *_, c in pts}
            )
            pol = {
                "all-IVF": route(t, 0.0),
                "all-exact (cheaper of V1 / V2 per query)": route(t, 1.01),
                "router (own fit)": pick[1:],
                "oracle (cheapest reaching the target)": oracle(t, TARGET),
            }
            for name, v in pol.items():
                rows.append(
                    (
                        ds,
                        bs,
                        name,
                        f"{pick[0]:.2e}" if name.startswith("router") else "",
                        round(v[0], 4),
                        round(v[1], 4),
                        round(v[2], 3),
                        round(float(t["v2"].mean()), 3),
                        " ".join(cvs),
                    )
                )
            ax = figs[bs][1]
            ax.plot(
                [c[2] for c in curve], [c[1] for c in curve], "-", label=f"{ds}: threshold sweep"
            )
            for name, v in pol.items():
                ax.scatter(
                    [v[1]],
                    [v[0]],
                    s=30,
                    marker={"all-IVF": "o", "router (own fit)": "D"}.get(
                        name, "*" if name.startswith("oracle") else "s"
                    ),
                )
                ax.annotate(f"{ds} {name.split(' (')[0]}", (v[1], v[0]), fontsize=5)
    for (src, bs), thr in sorted(best.items()):
        for (ds, bs2), t in sorted(tabs.items()):
            if bs2 == bs and ds != src:
                v = route(t, thr)
                rows.append(
                    (
                        ds,
                        bs,
                        f"router (t fitted on {src})",
                        f"{thr:.2e}",
                        round(v[0], 4),
                        round(v[1], 4),
                        round(v[2], 3),
                        round(float(t["v2"].mean()), 3),
                        "",
                    )
                )
    for bs, (fig, ax) in figs.items():
        ax.axhline(TARGET, color="k", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel(
            f"mean per-query p50 (ms, graph, per-cell batch medians, B={bs}; routing assumed free)"
        )
        ax.set_ylabel("mean recall_oracle@100")
        ax.set_title(f"Pass-rate router counterfactual, B={bs} (NOT CITABLE)", fontsize=9)
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / f"router-bs{bs}.png", dpi=130)
    with open(out / "router.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "dataset",
                "bs",
                "policy",
                "t",
                "recall@100",
                "latency_ms",
                "share_exact",
                "share_exact_on_V2",
                "code_versions",
            ]
        )
        w.writerows(rows)
    qs = [
        (ds, fk, sw, len(d["p"]), float(np.median(d["p"])), float(np.nanmean(d["ivf"])), d["cv"])
        for (ds, fk, sw), d in sorted(pq.items())
    ]
    with open(out / "router-querysets.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "dataset",
                "filter_kind",
                "sweep",
                "queries",
                "median_pass_rate",
                "ivf_recall",
                "code_version",
            ]
        )
        w.writerows(qs)
    print(f"{len(rows)} policy rows over {len(pq)} query sets")


if __name__ == "__main__":
    main()
