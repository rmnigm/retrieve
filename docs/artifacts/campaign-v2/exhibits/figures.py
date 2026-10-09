"""The exhibits `bench report` lacks or draws unreadably (README.md § figures): F2x, F3x, G1 and the
ratio tables behind T1. PNG + CSV per figure, every figure watermarked NOT CITABLE.

usage: figures.py OUT TREE [TREE ...]
"""

import collections
import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from load import ALGO, box, cv, load, pass_p, perf, pre_clause_skip, recall  # noqa: E402

from bench import stats  # noqa: E402  (load.py put evaluation/ on sys.path)

OUT = Path(sys.argv[1])
TREES = [Path(t) for t in sys.argv[2:]]
REAL = {
    "goodreads-synth": "goodreads",
    "arxiv-synth": "arxiv",
    "yfcc10m-synth": "yfcc10m",
    "arxiv-corr-synth": "arxiv",
}


def mark(fig, recs):
    fig.text(
        0.5,
        0.5,
        "NOT CITABLE",
        fontsize=40,
        color="red",
        alpha=0.15,
        ha="center",
        va="center",
        rotation=30,
    )
    cvs = sorted({cv(r) for r in recs})
    fig.text(
        0.01,
        0.005,
        f"exhibits | code_version {', '.join(cvs)} | NOT CITABLE (gate not green)",
        fontsize=6,
    )


def write_csv(name, header, rows):
    with open(OUT / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def med(x):
    x = [v for v in x if v is not None]
    return st.median(x) if x else None


def lat(rs, bs, k=100):
    """Median over seeds of the graph p50, else eager (official)."""
    v = []
    for r in rs:
        e = perf(r, bs, k, "graph") or perf(r, bs, k, "eager")
        if e:
            v.append(e["median_ms"])
    return med(v)


def tree_of(r):
    for t in TREES:
        if t.name == r["_tree"]:
            return t


def buckets(rs, k=100):
    """Real-sweep per-query recall bucketed by the query's own pass rate (half-decade bins, >= 20 queries)."""
    pc, rc = [], []
    for r in rs:
        if not r.get("per_query"):
            continue
        z = np.load(tree_of(r) / r["per_query"])
        pc.append(z["pass_count"] / r["n_items"])
        rc.append(z[f"recall_oracle@{k}"])
    if not pc:
        return []
    pc, rc = np.concatenate(pc), np.concatenate(rc)
    ok = (pc > 0) & ~np.isnan(rc)
    pc, rc = pc[ok], rc[ok]
    edges = 10.0 ** np.arange(-7, 0.51, 0.5)
    out = []
    for a, b in zip(edges, edges[1:]):
        m = (pc >= a) & (pc < b)
        if m.sum() >= 20:
            out.append((float(np.sqrt(a * b)), float(rc[m].mean()), int(m.sum())))
    return out


def f2x(recs):
    """recall_oracle@100 vs pass rate: every synth arm (V1, V3 per pool, ST per n_probe, postfilter per alpha),
    real-filter sweeps of the same dataset as per-query buckets."""
    synth = [
        r
        for r in recs
        if r["suite"] == "synth"
        and r["filter_kind"] == "clause"
        # recall only: a quality-only record (`skip_perf`) is complete for this figure
        and (r["status"] == "ok" or set(r.get("partial_reasons") or []) <= {"skip_perf", "ks_bs"})
        and r["status"] != "failed"
    ]
    dss = sorted({(r["dataset"], r["_tree"]) for r in synth})
    if not dss:
        return
    fig, axes = plt.subplots(1, len(dss), figsize=(6.5 * len(dss), 4.8), squeeze=False)
    rows = []
    for ax, (ds, tree) in zip(axes[0], dss):
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in synth:
            if (
                r["dataset"] == ds
                and r["_tree"] == tree
                and r["backend"] in ("triton", "torch", "official")
                and not r["params"].get("compile")
                and not (
                    r["algo"] in ("linr_v1_filter_mask", "linr_v2") and r["backend"] == "torch"
                )
            ):
                lab = (
                    ALGO[r["algo"]]
                    + (f" n_probe={r['params']['n_probe']}" if "n_probe" in r["params"] else "")
                    + (
                        f" pool={r['params']['candidate_pool_frac']:g}"
                        if "candidate_pool_frac" in r["params"]
                        else ""
                    )
                    + (f" α={r['params']['alpha']}" if "alpha" in r["params"] else "")
                    + f" [{r['backend']}]"
                )
                g[lab][pass_p(r)].append(recall(r))
        for lab, by in sorted(g.items()):
            ps = sorted(by)
            ys = [med(by[p]) for p in ps]
            ls = "--" if lab.startswith("PF") else "-"
            ax.plot(ps, ys, ls, marker="o", ms=4, label=f"synth {lab}")
            rows += [(ds, cv_tree(tree), "synth", lab, p, y, len(by[p])) for p, y in zip(ps, ys)]
        real = [
            r
            for r in recs
            if r["dataset"] == REAL[ds]
            and r["suite"] == "filter"
            and r["status"] == "ok"
            and r["filter_kind"] == "clause"
        ]
        for algo, extra in (
            ("silvertorch", {"n_probe": 24}),
            ("postfilter", {"alpha": 1}),
            ("postfilter", {"alpha": 8}),
            ("linr_v3", {}),
        ):
            rs = [
                r
                for r in real
                if r["algo"] == algo
                and r["backend"] in ("triton", "torch")
                and all(r["params"].get(k) == v for k, v in extra.items())
                and not r["params"].get("compile")
            ]
            if not rs:
                continue
            b = buckets(rs)
            lab = ALGO[algo] + "".join(f" {k}={v}" for k, v in extra.items()) + f" ({cv(rs[0])})"
            if b:
                ax.scatter(
                    [x for x, _, _ in b],
                    [y for _, y, _ in b],
                    marker="x",
                    s=30,
                    label=f"real {lab}",
                )
                rows += [(REAL[ds], cv(rs[0]), "real-bucket", lab, x, y, n) for x, y, n in b]
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("pass rate (synth: target p; real: each query's own pass rate)")
        ax.set_ylabel("recall_oracle@100 (median over seeds)")
        ax.set_title(f"{ds} ({cv_tree(tree)}), clause")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6, loc="lower right")
    fig.suptitle(
        "F2x: recall vs pass rate, uniform synthetic filter + real per-query buckets (k 100)"
    )
    mark(fig, synth)
    fig.tight_layout()
    fig.savefig(OUT / "f2x-recall-vs-pass-rate.png", dpi=130)
    write_csv(
        "f2x-recall-vs-pass-rate.csv",
        ["dataset", "code_version", "kind", "arm", "pass_rate", "recall@100", "n"],
        rows,
    )


FIX_A = {"v2": "V2 pre-Fix A", "v2.1": "V2 post-Fix A"}


def f1x(recs):
    """C1: V1 and V2 [triton] p50 vs pass rate, one row per (synth dataset, code_version), with the
    paired V2/V1 ratio over interleaved rounds (seed x window) and its 95 % CI."""
    v = [
        r
        for r in recs
        if r["suite"] == "synth"
        and r["status"] == "ok"
        and r["backend"] == "triton"
        and r["algo"] in ("linr_v1_filter_mask", "linr_v2")
    ]
    rows_ = sorted({(r["n_items"], r["dataset"], cv(r), box(r)) for r in v})
    if not rows_:
        return
    fig, axes = plt.subplots(len(rows_), 3, figsize=(15, 3.8 * len(rows_)), squeeze=False)
    out = []
    for i, (n, ds, c, bx) in enumerate(rows_):
        rs = [r for r in v if (r["n_items"], r["dataset"], cv(r), box(r)) == (n, ds, c, bx)]
        width = ", 10-clause table, pre-CLAUSE-SKIP" if any(pre_clause_skip(r) for r in rs) else ""
        for fk, ls in (("clause", "-"), ("bloom", "--")):
            g = collections.defaultdict(lambda: collections.defaultdict(list))
            pair = collections.defaultdict(lambda: ([], []))
            for r in rs:
                if r["filter_kind"] != fk:
                    continue
                g[r["algo"]][pass_p(r)].append(r)
            for j, bs in enumerate((1, 16)):
                for algo, by in sorted(g.items()):
                    ps = sorted(by)
                    ys = [lat(by[p], bs) for p in ps]
                    axes[i][j].plot(ps, ys, ls, marker="o", ms=3, label=f"{ALGO[algo]} {fk}")
                ratio_pts = []
                for p in sorted(g.get("linr_v2", {})):
                    a, b = pair[(p, bs)]
                    for r2 in g["linr_v2"][p]:
                        r1 = [
                            x
                            for x in g.get("linr_v1_filter_mask", {}).get(p, [])
                            if x["seed"] == r2["seed"]
                        ]
                        e2, e1 = perf(r2, bs, 100, "graph"), r1 and perf(r1[0], bs, 100, "graph")
                        same = r1 and (r2.get("interleave") or {}).get("group") == (
                            r1[0].get("interleave") or {}
                        ).get("group")
                        if (
                            e2
                            and e1
                            and same
                            and len(e2["window_medians_ms"]) == len(e1["window_medians_ms"])
                        ):
                            a += e2["window_medians_ms"]
                            b += e1["window_medians_ms"]
                    ci = stats.paired_ratio_ci(a, b) if a else None
                    if ci:
                        ratio_pts.append((p, *ci))
                        out.append(
                            (ds, n, c, FIX_A.get(c, c), fk, bs, p, *ci, len(a), stats.differs(ci))
                        )
                if ratio_pts:
                    ax = axes[i][2]
                    x = [q[0] for q in ratio_pts]
                    ax.errorbar(
                        x,
                        [q[1] for q in ratio_pts],
                        yerr=[[q[1] - q[2] for q in ratio_pts], [q[3] - q[1] for q in ratio_pts]],
                        fmt=ls + ("o" if bs == 1 else "s"),
                        ms=3,
                        capsize=2,
                        label=f"{fk} B={bs}",
                    )
        for j, bs in enumerate((1, 16)):
            ax = axes[i][j]
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(
                f"{ds}, N {n / 1e6:.1f} M ({c}, {FIX_A.get(c, c)}, box {bx}{width}), B={bs}",
                fontsize=9,
            )
            ax.set_xlabel("pass rate p")
            ax.set_ylabel("p50 ms (graph), k 100")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=6)
        ax = axes[i][2]
        ax.axhline(1.0, color="k", lw=0.7)
        ax.set_xscale("log")
        ax.set_title(f"V2/V1 paired over interleaved rounds ({c}, {FIX_A.get(c, c)})", fontsize=9)
        ax.set_xlabel("pass rate p")
        ax.set_ylabel("V2 / V1 (< 1: V2 faster), 95 % CI")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6)
    fig.suptitle(
        "F1x / C1: LiNR V1 vs V2 by pass rate and scale (rows differ in code_version and box: compare ratios within a row only)"
    )
    mark(fig, v)
    fig.tight_layout()
    fig.savefig(OUT / "f1x-v1-v2-vs-pass-rate.png", dpi=120)
    write_csv(
        "f1x-v2-over-v1.csv",
        [
            "dataset",
            "n_items",
            "code_version",
            "v2_state",
            "filter_kind",
            "bs",
            "p",
            "v2_over_v1",
            "ci_lo",
            "ci_hi",
            "rounds",
            "differs",
        ],
        out,
    )


def cv_tree(t):
    return {
        "v2": "v2 408b1188",
        "v21": "v2.1 f01255f1",
        "v22": "v2.2 0d23c615",
        "v23": "v2.3 1258a63e",
        "v24": "v2.4 d67d6263",
        "v25": "v2.5 472f2fc6",
        "v26": "v2.6 20e83bfc",
        "v27": "v2.7 641ec3b8",
        "v28": "v2.8 78cfbc72",
        "d1": "d1 72e5a90",
    }.get(t, t.split("__")[0].replace("campaign-", "") if t.startswith("campaign-v") else t)


def f3x(recs):
    """Recall-latency Pareto from `deep`, one panel per (dataset, sweep, bs): triton / official × n_lists,
    clause solid / bloom dashed, V3 along its pool; medians over seeds."""
    deep = [r for r in recs if r["suite"] == "deep" and r["status"] == "ok"]
    if not deep:
        return
    panels = sorted({(r["dataset"], r["_tree"], r["sweep"]) for r in deep})
    bss = [1, 16]
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 3.4 * len(panels)), squeeze=False)
    rows = []
    for i, (ds, tree, sw) in enumerate(panels):
        for j, bs in enumerate(bss):
            ax = axes[i][j]
            g = collections.defaultdict(lambda: collections.defaultdict(list))
            for r in deep:
                if (r["dataset"], r["_tree"], r["sweep"]) != (ds, tree, sw):
                    continue
                if r["algo"] == "silvertorch":
                    c, x = (
                        (
                            f"ST n_lists={r['params']['n_lists']} [{r['backend']}]",
                            r["filter_kind"],
                        ),
                        r["params"]["n_probe"],
                    )
                else:
                    c, x = (
                        (f"V3 [{r['backend']}]", r["filter_kind"]),
                        r["params"].get("candidate_pool") or r["params"].get("candidate_pool_frac"),
                    )
                g[c][x].append(r)
            for (lab, fk), by in sorted(g.items()):
                pts = []
                for x in sorted(by):
                    y, t = med([recall(r) for r in by[x]]), lat(by[x], bs)
                    if y is not None and t is not None:
                        pts.append((t, y, x))
                        rows.append((ds, cv_tree(tree), sw, bs, lab, fk, x, t, y))
                if pts:
                    ax.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        "-" if fk == "clause" else "--",
                        marker="o",
                        ms=3,
                        label=f"{lab} {fk}",
                    )
            ax.axhline(0.95, color="k", lw=0.6, ls=":")
            ax.axhline(0.90, color="k", lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_title(f"{ds} {sw} ({cv_tree(tree)}), B={bs}", fontsize=9)
            ax.set_xlabel("p50 ms (graph; official eager)")
            ax.set_ylabel("recall_oracle@100")
            ax.grid(alpha=0.3, which="both")
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.suptitle("F3x: recall-latency Pareto (deep), medians over seeds, k 100")
    mark(fig, deep)
    fig.tight_layout()
    fig.savefig(OUT / "f3x-pareto.png", dpi=110)
    write_csv(
        "f3x-pareto.csv",
        [
            "dataset",
            "code_version",
            "sweep",
            "bs",
            "curve",
            "filter_kind",
            "param",
            "p50_ms",
            "recall@100",
        ],
        rows,
    )


def g1(recs):
    """V1 triton graph / eager p50 vs pass rate (synth p; real sweeps at their mean pass rate), bs 1 and 16."""
    v1 = [
        r
        for r in recs
        if r["algo"] == "linr_v1_filter_mask" and r["backend"] == "triton" and r["status"] == "ok"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    rows = []
    for ax, bs in zip(axes, (1, 16)):
        g = collections.defaultdict(list)
        for r in v1:
            e, x = perf(r, bs, 100, "eager"), perf(r, bs, 100, "graph")
            if e and x:
                g[(r["dataset"], r["_tree"], r["filter_kind"])].append(
                    (
                        r["pass_rate"],
                        x["median_ms"] / e["median_ms"],
                        e["median_ms"],
                        x["median_ms"],
                        e.get("sm_mhz"),
                        x.get("sm_mhz"),
                        bool(e.get("unstable") or x.get("unstable")),
                        r["sweep"],
                        r["seed"],
                    )
                )
        for (ds, tree, fk), v in sorted(g.items()):
            v.sort()
            ax.plot(
                [a[0] for a in v],
                [a[1] for a in v],
                "o-" if fk == "clause" else "s--",
                ms=3,
                label=f"{ds} {fk} ({cv_tree(tree)})",
            )
            rows += [(ds, cv_tree(tree), fk, bs) + a for a in v]
        ax.axhline(1.0, color="k", lw=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("pass rate (record)")
        ax.set_ylabel("graph p50 / eager p50")
        ax.set_title(f"V1 [triton], k 100, B={bs}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6)
    fig.suptitle("G1: CUDA-graph replay vs eager for LiNR V1 (> 1 = graph slower)")
    mark(fig, v1)
    fig.tight_layout()
    fig.savefig(OUT / "g1-v1-graph-vs-eager.png", dpi=130)
    write_csv(
        "g1-v1-graph-vs-eager.csv",
        [
            "dataset",
            "code_version",
            "filter_kind",
            "bs",
            "pass_rate",
            "graph_over_eager",
            "eager_ms",
            "graph_ms",
            "eager_sm_mhz",
            "graph_sm_mhz",
            "unstable",
            "sweep",
            "seed",
        ],
        rows,
    )


def ratios(recs):
    """T1 inputs: C1 V2/V1, C2 V3/V2, C3 torch/triton, per synth p and bs (graph; torch eager and compiled)."""
    synth = [r for r in recs if r["suite"] == "synth" and r["status"] == "ok"]
    g = collections.defaultdict(list)
    for r in synth:
        tag = (
            f"{r['algo']}/{r['backend']}"
            + (f"/{r['params']['compile']}" if r["params"].get("compile") else "")
            + (
                f"/pool={r['params']['candidate_pool_frac']}"
                if "candidate_pool_frac" in r["params"]
                else ""
            )
            + (f"/a={r['params']['alpha']}" if "alpha" in r["params"] else "")
            + (f"/np={r['params']['n_probe']}" if "n_probe" in r["params"] else "")
        )
        g[(r["dataset"], r["_tree"], r["filter_kind"], pass_p(r), tag)].append(r)
    rows = []
    for (ds, tree, fk, p, tag), rs in sorted(g.items()):
        for bs in (1, 16):
            for mode in ("eager", "graph"):
                v = [perf(r, bs, 100, mode) for r in rs]
                v = [e["median_ms"] for e in v if e]
                if v:
                    rows.append(
                        (
                            ds,
                            cv_tree(tree),
                            fk,
                            p,
                            tag,
                            bs,
                            mode,
                            st.median(v),
                            med([recall(r) for r in rs]),
                            len(v),
                        )
                    )
    write_csv(
        "synth-arms.csv",
        [
            "dataset",
            "code_version",
            "filter_kind",
            "p",
            "arm",
            "bs",
            "mode",
            "p50_ms",
            "recall@100",
            "seeds",
        ],
        rows,
    )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load(TREES)
    f1x(recs)
    f2x(recs)
    f3x(recs)
    g1(recs)
    ratios(recs)
    print("\n".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
