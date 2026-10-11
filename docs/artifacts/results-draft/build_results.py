"""Tables and figures for docs/paper/results.md (NOT CITABLE): adapted official SilverTorch against LiNR V1-V3.

Usage: build_results.py HUB OUT [RERUN_PARQUET ...]

HUB is a local copy of pinkmeme/eval-results holding campaign-final/merged-4016/results.parquet,
campaign-final/ceiling-int8-*/ and artifacts/meta-fork/<gate>/out/*.json. Every number is read from those files.

The adapted official SilverTorch arm comes from records with backend "official-fork" when a parquet holding them is
passed (the rerun) and is then measured. Until then it is bridged from the gate grid: the gate times the adapted code
and our Triton SilverTorch in one process, so adapted = campaign Triton latency x (adapted / Triton in the gate), and
its recall is the campaign Triton record's, because the gate shows ids equal up to ties (none / exact) and the bloom
path equal to Meta's code. Bridged rows carry source "bridged" and are provisional.
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HUB = Path(sys.argv[1])
OUT = Path(sys.argv[2])
RERUN = [Path(p) for p in sys.argv[3:]]
GATES = ["head-82fafb6", "laion30m-82fafb6"]
LINR = {"linr_v1_filter_mask": "V1", "linr_v2": "V2", "linr_v3": "V3"}
SUITE_PREF = [
    "filter",
    "synth",
    "deep",
    "laion30m",
    "laion30m-bs1",
    "nsweep",
    "nsweep-synth",
    "laion30m-synth",
]
TARGETS = [0.9, 0.95, 0.99]


def load_records():
    parts = [pd.read_parquet(HUB / "campaign-final/merged-4016/results.parquet")]
    parts += [pd.read_parquet(p) for p in RERUN]
    df = pd.concat(parts, ignore_index=True)
    p = df.params.map(json.loads)
    for key in [
        "n_lists",
        "n_probe",
        "bloom_path",
        "candidate_pool_frac",
        "k_bits",
        "m_bits",
        "k_hash",
        "compile",
    ]:
        df[key] = p.map(lambda d, key=key: d.get(key))
    df["recall"] = np.where(
        df.perf_k == 1000, df["oracle_recall@1000"], df["oracle_recall@100"]
    )
    df["suite_rank"] = df.suite.map({s: i for i, s in enumerate(SUITE_PREF)}).fillna(99)
    return df


def load_gate():
    rows = []
    for gate in GATES:
        for f in sorted(
            glob.glob(str(HUB / "artifacts/meta-fork" / gate / "out/*.json"))
        ):
            for c in json.load(open(f)):
                c["gate"] = gate
                rows.append(c)
    g = pd.DataFrame(rows)
    g["fork_graph"] = g.graph_ms.map(
        lambda d: d.get("fork") if isinstance(d, dict) else np.nan
    )
    g["triton_graph"] = g.graph_ms.map(
        lambda d: d.get("triton") if isinstance(d, dict) else np.nan
    )
    g["fork_eager"] = g.eager_ms.map(lambda d: d["fork"])
    g["upstream_eager"] = g.eager_ms.map(lambda d: d["upstream"])
    g["triton_eager"] = g.eager_ms.map(lambda d: d["triton"])
    g["x_equal"] = g.vs_upstream.map(lambda d: bool(d["equal"]))
    g["kind"] = g["mode"].map({"exact": "clause", "bloom": "bloom", "none": "none"})
    return g


def geo(x):
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.log(x).mean()))


def gate_cells(g):
    """One row per gate cell: both timing orders folded (geometric mean of the per-order ratios)."""
    g = g[(g.score_path == "int32") & ~g.aa]
    keys = ["dataset", "dim", "kind", "sweep", "bloom_path", "n_probe", "bs", "k"]
    agg = g.groupby(keys).agg(
        fork_graph=("fork_graph", "mean"),
        triton_graph=("triton_graph", "mean"),
        fork_eager=("fork_eager", "mean"),
        upstream_eager=("upstream_eager", "mean"),
        r_graph=("graph_fork_over_triton", geo),
        r_eager_up=("eager_fork_over_upstream", geo),
        x_equal=("x_equal", "all"),
        n_orders=("swap", "nunique"),
        gate=("gate", "first"),
    )
    return agg.reset_index()


def st_records(df):
    return df[(df.algo == "silvertorch") & df.n_probe.notna() & (df.seed == 0)]


def adapted_arm(df, gc):
    """Adapted official SilverTorch per (dataset, dim, kind, sweep, n_probe, bs, k), graph mode."""
    st = st_records(df)
    measured = st[
        (st.backend == "official-fork")
        & (st.perf_mode == "graph")
        & (st.bloom_path.isna() | (st.bloom_path == "partial"))
    ]
    if len(measured):
        out = measured.assign(
            kind=measured.filter_kind,
            bs=measured.perf_bs,
            k=measured.perf_k,
            ms=measured.perf_median_ms,
            source="measured",
        )
        return out[
            [
                "dataset",
                "dim",
                "kind",
                "sweep",
                "n_lists",
                "n_probe",
                "bs",
                "k",
                "ms",
                "recall",
                "pass_rate",
                "source",
            ]
        ]
    tri = st[
        (st.backend == "triton")
        & (st.perf_mode == "graph")
        & (st.bloom_path.isna() | (st.bloom_path == "partial"))
    ]
    tri = tri.sort_values(["suite_rank"]).drop_duplicates(
        ["dataset", "dim", "filter_kind", "sweep", "n_probe", "perf_bs", "perf_k"]
    )
    cells = gc[(gc.bloom_path == "partial") & (gc.kind != "none")]
    m = cells.merge(
        tri,
        left_on=["dataset", "dim", "kind", "sweep", "n_probe", "bs", "k"],
        right_on=[
            "dataset",
            "dim",
            "filter_kind",
            "sweep",
            "n_probe",
            "perf_bs",
            "perf_k",
        ],
    )
    m = m.assign(
        ms=m.perf_median_ms * m.r_graph,
        triton_campaign_ms=m.perf_median_ms,
        calib=m.triton_graph / m.perf_median_ms,
        source="bridged",
    )
    return m[
        [
            "dataset",
            "dim",
            "kind",
            "sweep",
            "n_lists",
            "n_probe",
            "bs",
            "k",
            "ms",
            "recall",
            "pass_rate",
            "source",
            "r_graph",
            "triton_campaign_ms",
            "triton_graph",
            "calib",
            "gate",
        ]
    ]


def linr_arm(df):
    x = df[
        df.algo.isin(LINR)
        & (df.perf_mode == "graph")
        & (df.seed == 0)
        & (df.backend == "triton")
    ]
    x = x[x.candidate_pool_frac.isna() | (x.suite != "v3bits")]
    x = x.sort_values("suite_rank").drop_duplicates(
        ["dataset", "dim", "filter_kind", "sweep", "algo", "perf_bs", "perf_k"]
    )
    x = x.assign(variant=x.algo.map(LINR))
    ms = x.pivot_table(
        index=["dataset", "dim", "filter_kind", "sweep", "perf_bs", "perf_k"],
        columns="variant",
        values="perf_median_ms",
    )
    rc = x.pivot_table(
        index=["dataset", "dim", "filter_kind", "sweep", "perf_bs", "perf_k"],
        columns="variant",
        values="recall",
    )
    ms.columns = [f"{c}_ms" for c in ms.columns]
    rc.columns = [f"{c}_recall" for c in rc.columns]
    out = (
        ms.join(rc)
        .reset_index()
        .rename(columns={"filter_kind": "kind", "perf_bs": "bs", "perf_k": "k"})
    )
    exact = [c for c in ["V1_ms", "V2_ms"] if c in out]
    out = out.dropna(subset=exact, how="all")
    out["linr_best_ms"] = out[exact].min(axis=1)
    out["linr_best"] = out[exact].idxmin(axis=1).str[:2]
    return out


def head_to_head(adapted, linr):
    h = adapted.merge(linr, on=["dataset", "dim", "kind", "sweep", "bs", "k"])
    h["st_over_linr"] = h.ms / h.linr_best_ms
    h["winner"] = np.where(h.st_over_linr < 1, "SilverTorch", "LiNR " + h.linr_best)
    return h.sort_values(["dataset", "dim", "kind", "sweep", "bs", "k", "n_probe"])


def matched(h):
    rows = []
    for key, grp in h.groupby(["dataset", "dim", "kind", "sweep", "bs", "k"]):
        base = grp.iloc[0]
        for t in TARGETS:
            ok = grp[grp.recall >= t]
            row = dict(
                zip(["dataset", "dim", "kind", "sweep", "bs", "k"], key),
                target=t,
                pass_rate=base.pass_rate,
                linr_best=base.linr_best,
                linr_ms=base.linr_best_ms,
                max_recall=grp.recall.max(),
            )
            if len(ok):
                best = ok.loc[ok.ms.idxmin()]
                row.update(
                    n_probe=best.n_probe,
                    st_ms=best.ms,
                    st_recall=best.recall,
                    st_over_linr=best.ms / base.linr_best_ms,
                )
                row["winner"] = (
                    "SilverTorch"
                    if best.ms < base.linr_best_ms
                    else f"LiNR {base.linr_best}"
                )
            else:
                row["winner"] = (
                    f"LiNR {base.linr_best} (SilverTorch below target at every gated n_probe)"
                )
            rows.append(row)
    return pd.DataFrame(rows)


def adaptation(gc):
    g = gc.groupby(["dataset", "dim", "kind"]).agg(
        cells=("x_equal", "size"),
        x_equal=("x_equal", "sum"),
        adapted_over_meta_median=("r_eager_up", "median"),
        adapted_over_meta_max=("r_eager_up", "max"),
        adapted_over_meta_min=("r_eager_up", "min"),
    )
    return g.reset_index()


def crossover(ps, ratios):
    """First pass rate where V2 / V1 reaches 1, log-interpolated; None if V2 never loses or always loses."""
    for (p0, r0), (p1, r1) in zip(zip(ps, ratios), zip(ps[1:], ratios[1:])):
        if r0 < 1 <= r1:
            f = (np.log(1) - np.log(r0)) / (np.log(r1) - np.log(r0))
            return float(np.exp(np.log(p0) + f * (np.log(p1) - np.log(p0))))
    return None


def c1(df):
    x = df[
        df.algo.isin(["linr_v1_filter_mask", "linr_v2"])
        & (df.backend == "triton")
        & (df.perf_mode == "graph")
        & (df.perf_k == 100)
        & (df.seed == 0)
        & df.suite.isin(["synth", "nsweep-synth", "laion30m-synth"])
        & (df.filter_kind == "clause")
    ]
    x = x.sort_values("suite_rank").drop_duplicates(
        ["dataset", "dim", "sweep", "algo", "perf_bs"]
    )
    rows, curves = [], []
    for (ds, dim, bs), grp in x.groupby(["dataset", "dim", "perf_bs"]):
        pv = (
            grp.pivot_table(index="pass_rate", columns="algo", values="perf_median_ms")
            .dropna()
            .sort_index()
        )
        if len(pv) < 3:
            continue
        r = (pv.linr_v2 / pv.linr_v1_filter_mask).tolist()
        ps = pv.index.tolist()
        n = int(grp.n_items.iloc[0])
        rows.append(
            dict(
                dataset=ds,
                dim=dim,
                n_items=n,
                bs=bs,
                crossover_p=crossover(ps, r),
                v2_over_v1_lowest_p=r[0],
                lowest_p=ps[0],
                v2_over_v1_p1=r[-1],
            )
        )
        curves += [
            dict(dataset=ds, dim=dim, n_items=n, bs=bs, pass_rate=p, v2_over_v1=v)
            for p, v in zip(ps, r)
        ]
    return pd.DataFrame(rows), pd.DataFrame(curves)


def c1_real(linr):
    x = linr[
        linr.dataset.isin(
            [
                "arxiv",
                "goodreads",
                "pubmed",
                "yfcc10m",
                "laion30m",
                "laion10m",
                "laion3m",
                "laion1m",
            ]
        )
        & (linr.k == 100)
        & (linr.kind == "clause")
    ].copy()
    x["v2_over_v1"] = x.V2_ms / x.V1_ms
    return x[["dataset", "dim", "sweep", "bs", "V1_ms", "V2_ms", "v2_over_v1"]]


def c2(df):
    x = df[
        (df.algo.isin(["linr_v2", "linr_v3"]))
        & (df.perf_mode == "graph")
        & (df.seed == 0)
        & (df.perf_k == 100)
        & (df.backend == "triton")
        & df.suite.isin(["filter", "synth", "v3bits"])
    ]
    v2 = (
        x[x.algo == "linr_v2"]
        .sort_values("suite_rank")
        .drop_duplicates(["dataset", "filter_kind", "sweep", "perf_bs"])
    )
    v3 = x[x.algo == "linr_v3"]
    m = v3.merge(
        v2[["dataset", "filter_kind", "sweep", "perf_bs", "perf_median_ms"]],
        on=["dataset", "filter_kind", "sweep", "perf_bs"],
        suffixes=("", "_v2"),
    )
    m["v3_over_v2"] = m.perf_median_ms / m.perf_median_ms_v2
    m["pool"] = m.candidate_pool_frac.fillna(0.01)
    return m[
        [
            "dataset",
            "dim",
            "filter_kind",
            "sweep",
            "pass_rate",
            "suite",
            "pool",
            "k_bits",
            "perf_bs",
            "recall",
            "v3_over_v2",
        ]
    ]


def c3(df):
    x = df[(df.suite.isin(["c3", "c3-real"])) & (df.perf_k == 100)]
    tri = x[x.backend == "triton"][
        [
            "dataset",
            "dim",
            "sweep",
            "pass_rate",
            "perf_bs",
            "perf_mode",
            "perf_median_ms",
        ]
    ]
    tor = x[x.backend == "torch"].assign(
        variant=lambda d: d.compile.fillna("eager torch")
    )
    m = tor.merge(
        tri,
        on=["dataset", "dim", "sweep", "pass_rate", "perf_bs", "perf_mode"],
        suffixes=("", "_triton"),
    )
    m["torch_over_triton"] = m.perf_median_ms / m.perf_median_ms_triton
    return m[
        [
            "suite",
            "dataset",
            "dim",
            "sweep",
            "pass_rate",
            "perf_bs",
            "perf_mode",
            "variant",
            "torch_over_triton",
        ]
    ]


def c4(df):
    x = df[df.suite == "bloomwidth"]
    return x[
        ["dataset", "sweep", "backend", "m_bits", "k_hash", "bloom_fp_rate", "recall"]
    ].sort_values(["dataset", "backend", "k_hash", "m_bits"])


def c5(df, gc):
    x = df[
        (df.algo == "silvertorch")
        & (df.backend == "official")
        & df.bloom_path.notna()
        & (df.perf_mode == "eager")
        & (df.seed == 0)
    ]
    keys = ["suite", "dataset", "n_items", "sweep", "n_lists", "n_probe", "perf_bs"]
    pv = (
        x.pivot_table(index=keys, columns="bloom_path", values="perf_median_ms")
        .dropna()
        .reset_index()
    )
    pv["full_over_partial"] = pv.full / pv.partial
    meta = pv.assign(arm="Meta as published (eager)")
    b = gc[gc.kind == "bloom"]
    fp = b.pivot_table(
        index=["dataset", "dim", "sweep", "n_probe", "bs", "k"],
        columns="bloom_path",
        values=["fork_graph", "fork_eager"],
    ).dropna()
    adapted = pd.DataFrame(
        {
            "graph_full_over_partial": fp[("fork_graph", "full")]
            / fp[("fork_graph", "partial")],
            "eager_full_over_partial": fp[("fork_eager", "full")]
            / fp[("fork_eager", "partial")],
        }
    ).reset_index()
    return meta, adapted


def c6(df):
    st = st_records(df)
    q = st[st.backend == "triton"].drop_duplicates(
        ["dataset", "dim", "filter_kind", "sweep", "n_lists", "n_probe"]
    )
    q = q[
        [
            "dataset",
            "dim",
            "filter_kind",
            "sweep",
            "pass_rate",
            "n_lists",
            "n_probe",
            "oracle_recall@100",
        ]
    ]
    ceil = []
    for f in sorted(
        glob.glob(str(HUB / "campaign-final/ceiling-int8-*/ceiling-*.json"))
    ):
        d = json.load(open(f))
        r = d["recall_oracle@100"]
        ceil.append(
            dict(
                dataset=d["dataset"],
                sweep=d["sweep"],
                pass_rate=d["pass_rate"],
                n_lists=d["params"]["n_lists"],
                n_probe=d["params"]["n_probe"],
                int8_shipped=r["shipped"],
                int8_per_row=r["per_row"],
                fp16=r["fp16"],
            )
        )
    return q.sort_values(
        ["dataset", "filter_kind", "sweep", "n_lists", "n_probe"]
    ), pd.DataFrame(ceil)


def linr_vs_n(df):
    x = df[
        df.suite.isin(["nsweep", "nsweep-synth"])
        & df.algo.isin(LINR)
        & (df.perf_mode == "graph")
        & (df.seed == 0)
    ]
    x = x.assign(variant=x.algo.map(LINR))
    pv = x.pivot_table(
        index=["suite", "sweep", "perf_bs", "variant"],
        columns="n_items",
        values="perf_median_ms",
    ).reset_index()
    mem = x.drop_duplicates(["dataset", "variant"])[
        ["dataset", "n_items", "variant", "index_mib", "build_s"]
    ]
    return pv, mem


def memory(df):
    x = df[(df.suite == "filter") & (df.seed == 0)]
    x = x[x.algo.isin(LINR) | ((x.algo == "silvertorch") & (x.backend == "official"))]
    x = x.assign(arm=x.algo.map(LINR).fillna("SilverTorch, Meta as published"))
    g = x.groupby(["dataset", "dim", "n_items", "filter_kind", "arm"]).agg(
        index_mib=("index_mib", "median"),
        build_s=("build_s", "median"),
        reserved_mib=("memory_reserved_mib", "median"),
    )
    return g.reset_index()


def md(t, floatfmt="{:.3g}"):
    cols = list(t.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in t.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fig_synth(h, path):
    sets = [d for d in ["goodreads-synth", "arxiv-synth"] if d in set(h.dataset)]
    fig, axes = plt.subplots(len(sets), 2, figsize=(10, 3.6 * len(sets)), squeeze=False)
    for i, ds in enumerate(sets):
        for j, bs in enumerate([1, 16]):
            ax = axes[i][j]
            x = h[
                (h.dataset == ds) & (h.bs == bs) & (h.k == 100) & (h.kind == "clause")
            ]
            base = x.drop_duplicates("sweep").sort_values("pass_rate")
            ax.plot(base.pass_rate, base.V1_ms, "o-", color="#444444", label="LiNR V1")
            ax.plot(base.pass_rate, base.V2_ms, "s--", color="#888888", label="LiNR V2")
            for npb, grp in x.groupby("n_probe"):
                grp = grp.sort_values("pass_rate")
                lab = f"SilverTorch n_probe {int(npb)} (recall {grp.recall.min():.2f}-{grp.recall.max():.2f})"
                ax.plot(grp.pass_rate, grp.ms, "^-", label=lab)
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="filter pass rate",
                ylabel="ms per batch (graph)",
                title=f"{ds}, bs {bs}, k 100",
            )
            ax.legend(fontsize=7)
    fig.suptitle(
        "PROVISIONAL, NOT CITABLE: adapted SilverTorch bridged from the gate grid",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def fig_c1(curves, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for j, bs in enumerate([1, 16]):
        ax = axes[j]
        for (ds, dim), grp in curves[curves.bs == bs].groupby(["dataset", "dim"]):
            grp = grp.sort_values("pass_rate")
            ax.plot(grp.pass_rate, grp.v2_over_v1, "o-", ms=3, label=f"{ds} d{dim}")
        ax.axhline(1, color="k", lw=0.8)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="filter pass rate",
            ylabel="V2 / V1 latency (graph)",
            title=f"LiNR V2 against V1, bs {bs}",
        )
        ax.legend(fontsize=7)
    fig.suptitle("NOT CITABLE until D1-G", fontsize=9)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_records()
    graw = load_gate()
    gc = gate_cells(graw)
    adapted = adapted_arm(df, gc)
    linr = linr_arm(df)
    h = head_to_head(adapted, linr)
    mt = matched(h)
    c1t, c1c = c1(df)
    meta5, adapted5 = c5(df, gc)
    c6q, c6c = c6(df)
    nsw, nmem = linr_vs_n(df)
    tables = {
        "h2h": h,
        "matched": mt,
        "adaptation": adaptation(gc),
        "c1_crossover": c1t,
        "c1_curves": c1c,
        "c1_real": c1_real(linr),
        "c2_v3": c2(df),
        "c3_torch": c3(df),
        "c4_bloom": c4(df),
        "c5_meta": meta5,
        "c5_adapted": adapted5,
        "c6_recall": c6q,
        "c6_ceiling": c6c,
        "linr_vs_n": nsw,
        "linr_mem_n": nmem,
        "memory": memory(df),
    }
    with open(OUT / "tables.md", "w") as f:
        src = (
            "measured (rerun)"
            if (adapted.source == "measured").any()
            else "bridged from the gate grid (PROVISIONAL)"
        )
        f.write(f"# results-draft tables (NOT CITABLE). Adapted SilverTorch: {src}\n\n")
        for name, t in tables.items():
            t.to_csv(OUT / f"{name}.csv", index=False)
            f.write(f"## {name}\n\n{md(t)}\n\n")
    fig_synth(h, OUT / "fig_h2h_synth")
    fig_c1(c1c, OUT / "fig_c1_v2_over_v1")
    summary = {
        "perf_rows": int(len(df)),
        "gate_cells": int(len(graw)),
        "gate_x_equal": int(graw.x_equal.sum()),
        "adapted_source": sorted(adapted.source.unique().tolist()),
        "h2h_rows": int(len(h)),
        "calib_median": float(h.calib.median()) if "calib" in h else None,
        "calib_range": [float(h.calib.min()), float(h.calib.max())]
        if "calib" in h
        else None,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
