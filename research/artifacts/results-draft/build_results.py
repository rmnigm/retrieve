"""Tables and figures for research/paper/results.md (NOT CITABLE): adapted official SilverTorch against LiNR V1-V3.

Usage: build_results.py HUB OUT

HUB is a local copy of pinkmeme/eval-results holding numbers/<leg>/results.parquet (the NUMBERS legs),
numbers/A1b-laion-small-lists/ceil/, campaign-final/merged-4016/results.parquet, campaign-final/ceiling-int8-*/ and
artifacts/meta-fork/<gate>/out/*.json. Every number is read from those files.

The head-to-head is the NUMBERS A2 / A3 legs only: SilverTorch (backend "official-fork") at the picked points, interleaved
with LiNR V1 / V2 in one process per cell group. A cell measured twice (A2-small and its rerun, A2b and its rerun, A2-small
and A2b at LAION 1 / 3 M) keeps its stable record, the later leg's when both are stable; a cell unstable in every run keeps
its record and the flag. yfcc10m-synth is read at k 100 only (its LiNR k 1000 cells miss the exact gate, HB-6). LiNR's
claims (C1-C3) read campaign-final plus the LiNR-only NUMBERS legs (L1-L4); C4 / C5 / C7 read the final pass and the gates.
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
GATES = ["head-82fafb6"]
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
TIMED = [  # later wins between two stable records of one cell
    "A2-small",
    "A2-small-rerun",
    "A2b-laion-small-lists",
    "A2b-laion-small-lists-rerun",
    "A2-pubmed",
    "A2-yfcc",
    "A2-laion10m",
    "A2-laion30m-real",
    "A2-laion30m-synth",
    "A3-st-k2048",
]
CURVES = [
    "A1-small",
    "A1-pubmed",
    "A1-yfcc",
    "A1-laion10m",
    "A1-laion30m",
    "A1-laion30m-synth",
    "A1b-laion-small-lists",
]
LINR_LEGS = [
    "L1-linr-bs64",
    "L1b-linr-bs64-laion",
    "L2-v3pool",
    "L2b-v3pool-laion",
    "L3-linr-corr",
    "L4-linr-k2048",
]
CELL = ["dataset", "dim", "filter_kind", "sweep", "perf_bs", "perf_k"]


def params(df):
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
    k = df.perf_k if "perf_k" in df else pd.Series(100, index=df.index)
    df["recall"] = np.select(
        [k == 1000, k == 2048],
        [df.get("oracle_recall@1000"), df.get("oracle_recall@2048")],
        df["oracle_recall@100"],
    )
    return df


def leg(name):
    return pd.read_parquet(HUB / "numbers" / name / "results.parquet").assign(leg=name)


def load_records():
    """campaign-final plus the LiNR-only NUMBERS legs (claims C1-C6)."""
    parts = [pd.read_parquet(HUB / "campaign-final/merged-4016/results.parquet")]
    parts += [leg(name) for name in LINR_LEGS]
    df = params(pd.concat(parts, ignore_index=True))
    df["suite_rank"] = df.suite.map({s: i for i, s in enumerate(SUITE_PREF)}).fillna(99)
    return df


def load_timed():
    """Every timed record of the A2 / A3 legs, and one record per cell (module docstring)."""
    raw = params(pd.concat([leg(n).assign(leg_rank=i) for i, n in enumerate(TIMED)]))
    df = raw[
        raw.perf_mode.notna() & ~((raw.dataset == "yfcc10m-synth") & (raw.perf_k > 100))
    ]
    df["unstable"] = df.unstable.fillna(False).astype(bool)
    key = CELL + ["algo", "backend", "n_lists", "n_probe", "perf_mode"]
    df = df.sort_values(["unstable", "leg_rank"], ascending=[True, False])
    return raw, df.drop_duplicates(key)


def load_curves():
    df = params(pd.concat([leg(n) for n in CURVES], ignore_index=True))
    return df.drop_duplicates(
        ["dataset", "dim", "filter_kind", "sweep", "n_lists", "n_probe"]
    )


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


def head_to_head(timed, mode="graph"):
    """Every SilverTorch point against the faster exact LiNR (V1 / V2) of its cell, same process, same mode."""
    x = timed[timed.perf_mode == mode]
    st = x[x.algo == "silvertorch"]
    li = x[x.algo.isin(["linr_v1_filter_mask", "linr_v2"])].assign(
        v=lambda d: d.algo.map(LINR)
    )
    ms = li.pivot_table(index=CELL, columns="v", values="perf_median_ms")
    ms["linr_unstable"] = li.groupby(CELL).unstable.any()
    ms["linr_ms"] = ms[["V1", "V2"]].min(axis=1)
    ms["linr_best"] = ms[["V1", "V2"]].idxmin(axis=1)
    h = st.merge(ms.reset_index(), on=CELL)
    h = h.assign(
        n_items=h.n_items,
        st_ms=h.perf_median_ms,
        st_over_linr=h.perf_median_ms / h.linr_ms,
        flagged=h.unstable | h.linr_unstable,
    )
    h["winner"] = np.where(h.st_over_linr < 1, "SilverTorch", "LiNR " + h.linr_best)
    cols = CELL + ["n_items", "pass_rate", "n_lists", "n_probe", "recall", "st_ms"]
    cols += [
        "V1",
        "V2",
        "linr_best",
        "linr_ms",
        "st_over_linr",
        "winner",
        "flagged",
        "leg",
    ]
    return h[cols].sort_values(CELL + ["n_lists", "n_probe"])


def tuned(h):
    """The tuned point per cell and recall target: the fastest SilverTorch (n_lists, n_probe) reaching the target."""
    rows = []
    for key, grp in h.groupby(CELL):
        base = grp.iloc[0]
        for t in TARGETS:
            ok = grp[grp.recall >= t]
            row = dict(
                zip(CELL, key), n_items=base.n_items, pass_rate=base.pass_rate, target=t
            )
            row.update(
                linr_best=base.linr_best,
                linr_ms=base.linr_ms,
                max_recall=grp.recall.max(),
            )
            if len(ok):
                b = ok.loc[ok.st_ms.idxmin()]
                row.update(
                    n_lists=b.n_lists, n_probe=b.n_probe, recall=b.recall, st_ms=b.st_ms
                )
                row.update(st_over_linr=b.st_over_linr, flagged=b.flagged)
                row["winner"] = (
                    "SilverTorch" if b.st_over_linr < 1 else f"LiNR {base.linr_best}"
                )
            else:
                row["winner"] = f"LiNR {base.linr_best} (SilverTorch below target)"
            rows.append(row)
    return pd.DataFrame(rows)


def at24(h):
    """The paper's production setting: n_probe 24 at the smallest n_lists timed for the dataset."""
    small = h.groupby("dataset").n_lists.transform("min")
    return h[(h.n_probe == 24) & (h.n_lists == small)]


def by_size(tn):
    """Real filters, clause, k 100: per dataset and target, how many cells SilverTorch wins and its ratio range."""
    x = tn[
        (tn.filter_kind == "clause")
        & (tn.perf_k == 100)
        & ~tn.dataset.str.contains("synth")
    ]
    g = x.groupby(["n_items", "dataset", "target", "perf_bs"])
    out = g.agg(
        filters=("sweep", "nunique"),
        reached=("st_ms", "count"),
        st_wins=("winner", lambda w: int((w == "SilverTorch").sum())),
        ratio_min=("st_over_linr", "min"),
        ratio_max=("st_over_linr", "max"),
    )
    return out.reset_index()


def eager_graph_ids(timed_all):
    """NUMBERS gate: eager and graph return the same ids on every adapted cell (canonical id hash)."""
    x = timed_all[(timed_all.backend == "official-fork") & timed_all.perf_mode.notna()]
    k = ["leg", "path"] + CELL + ["params"]
    pv = x.pivot_table(
        index=k, columns="perf_mode", values="perf_ids_sha256_canon", aggfunc="first"
    )
    pv = pv.dropna()
    return int(len(pv)), int((pv.eager == pv.graph).sum())


def linr_numbers(df):
    """LiNR-only NUMBERS legs: V2 / V1 at bs 64 / 256 (L1, L1b), correlated filters (L3), k 2048 (L4)."""
    x = df[
        df.leg.isin(
            ["L1-linr-bs64", "L1b-linr-bs64-laion", "L3-linr-corr", "L4-linr-k2048"]
        )
    ]
    x = x[x.perf_mode == "graph"].assign(v=lambda d: d.algo.map(LINR))
    pv = x.pivot_table(
        index=[
            "leg",
            "dataset",
            "dim",
            "filter_kind",
            "sweep",
            "pass_rate",
            "perf_bs",
            "perf_k",
        ],
        columns="v",
        values="perf_median_ms",
    ).reset_index()
    pv["v2_over_v1"] = pv.V2 / pv.V1
    return pv


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
        & df.suite.isin(["filter", "synth", "v3bits", "n-v3pool", "n-v3pool-laion"])
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


def c6(curves):
    """The adapted code's probe curves (A1 legs, quality only) and the int8 / fp16 ceiling checks."""
    cols = ["dataset", "dim", "filter_kind", "sweep", "pass_rate", "n_lists", "n_probe"]
    q = curves[cols + ["oracle_recall@100", "oracle_recall@1000"]]
    files = glob.glob(str(HUB / "campaign-final/ceiling-int8-*/ceiling-*.json"))
    files += glob.glob(str(HUB / "numbers/A1b-laion-small-lists/ceil/ceiling-*.json"))
    ceil = []
    for f in sorted(files):
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
                source=f.split("/")[-3],
            )
        )
    keys = ["dataset", "filter_kind", "sweep", "n_lists", "n_probe"]
    return q.sort_values(keys), pd.DataFrame(ceil)


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


def memory(df, timed):
    """Index size and build time: LiNR and Meta's code from the final pass, the adapted code from the NUMBERS legs."""
    x = df[(df.suite == "filter") & (df.seed == 0)]
    x = x[x.algo.isin(LINR) | ((x.algo == "silvertorch") & (x.backend == "official"))]
    x = x.assign(arm=x.algo.map(LINR).fillna("SilverTorch, Meta as published"))
    st = timed[(timed.algo == "silvertorch") & timed.index_mib.notna()]
    st = st.assign(arm="SilverTorch, adapted")
    keys = ["dataset", "dim", "n_items", "filter_kind", "arm", "n_lists"]
    g = pd.concat([x, st]).groupby(keys, dropna=False)
    g = g.agg(
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
    """Latency against pass rate on the synthetic sets: LiNR V1 / V2 and SilverTorch at its timed probe counts."""
    names = ["goodreads-synth", "arxiv-synth", "laion10m-synth", "laion30m-synth"]
    sets = [d for d in names if d in set(h.dataset)]
    fig, axes = plt.subplots(len(sets), 3, figsize=(14, 3.4 * len(sets)), squeeze=False)
    for i, ds in enumerate(sets):
        nl = h[h.dataset == ds].n_lists.min()
        for j, bs in enumerate([1, 16, 64]):
            ax = axes[i][j]
            x = h[
                (h.dataset == ds)
                & (h.perf_bs == bs)
                & (h.perf_k == 100)
                & (h.filter_kind == "clause")
            ]
            base = x.drop_duplicates("sweep").sort_values("pass_rate")
            ax.plot(base.pass_rate, base.V1, "o-", color="#444444", label="LiNR V1")
            ax.plot(base.pass_rate, base.V2, "s--", color="#888888", label="LiNR V2")
            for npb, grp in x[x.n_lists == nl].groupby("n_probe"):
                if grp.sweep.nunique() < 3:
                    continue
                grp = grp.sort_values("pass_rate")
                lab = f"SilverTorch {int(nl)} lists / {int(npb)} probes (recall {grp.recall.min():.2f}-{grp.recall.max():.2f})"
                ax.plot(grp.pass_rate, grp.st_ms, "^-", label=lab)
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="filter pass rate",
                ylabel="ms per batch (graph)",
                title=f"{ds}, bs {bs}, k 100",
            )
            ax.legend(fontsize=6)
    fig.suptitle(
        "NOT CITABLE: NUMBERS legs, SilverTorch adapted from Meta's code against LiNR",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def fig_tuned(tn, path):
    """SilverTorch / best exact LiNR at the tuned point against N: real filters, clause, k 100."""
    x = tn[
        (tn.filter_kind == "clause")
        & (tn.perf_k == 100)
        & ~tn.dataset.str.contains("synth")
    ]
    x = x[x.st_over_linr.notna()]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for i, t in enumerate([0.9, 0.95]):
        for j, bs in enumerate([1, 16, 64]):
            ax = axes[i][j]
            y = x[(x.target == t) & (x.perf_bs == bs)]
            for (ds, sw), grp in y.groupby(["dataset", "sweep"]):
                ax.scatter(grp.n_items, grp.st_over_linr, s=18)
                ax.annotate(
                    f"{ds} {sw}",
                    (grp.n_items.iloc[0], grp.st_over_linr.iloc[0]),
                    fontsize=5,
                )
            ax.axhline(1, color="k", lw=0.8)
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="items N",
                ylabel="SilverTorch / best LiNR (graph)",
                title=f"recall >= {t}, bs {bs}, k 100",
            )
    fig.suptitle(
        "NOT CITABLE: tuned SilverTorch point per cell; filters it cannot reach are not drawn",
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
    timed_all, timed = load_timed()
    graw = load_gate()
    gc = gate_cells(graw)
    h = head_to_head(timed)
    tn = tuned(h)
    c1t, c1c = c1(df)
    meta5, adapted5 = c5(df, gc)
    c6q, c6c = c6(load_curves())
    nsw, nmem = linr_vs_n(df)
    pairs, equal = eager_graph_ids(timed_all)
    tables = {
        "by_size": by_size(tn),
        "tuned": tn,
        "at24": at24(h),
        "h2h": h,
        "h2h_eager": head_to_head(timed, "eager"),
        "adaptation": adaptation(gc),
        "c1_crossover": c1t,
        "c1_curves": c1c,
        "c1_real": c1_real(linr_arm(df)),
        "linr_numbers": linr_numbers(df),
        "c2_v3": c2(df),
        "c3_torch": c3(df),
        "c4_bloom": c4(df),
        "c5_meta": meta5,
        "c5_adapted": adapted5,
        "c6_recall": c6q,
        "c6_ceiling": c6c,
        "linr_vs_n": nsw,
        "linr_mem_n": nmem,
        "memory": memory(df, timed),
    }
    with open(OUT / "tables.md", "w") as f:
        f.write(
            "# results-draft tables (NOT CITABLE). SilverTorch measured on the NUMBERS legs\n\n"
        )
        for name, t in tables.items():
            t.to_csv(OUT / f"{name}.csv", index=False)
            f.write(f"## {name}\n\n{md(t)}\n\n")
    fig_synth(h, OUT / "fig_h2h_synth")
    fig_tuned(tn, OUT / "fig_tuned_vs_n")
    fig_c1(c1c, OUT / "fig_c1_v2_over_v1")
    legs = TIMED + CURVES + LINR_LEGS
    summary = {
        "campaign_rows": int(len(df)),
        "timed_rows": int(len(timed)),
        "timed_flagged_rows": int(timed.unstable.sum()),
        "h2h_rows": int(len(h)),
        "h2h_flagged": int(h.flagged.sum()),
        "eager_graph_ids_pairs": pairs,
        "eager_graph_ids_equal": equal,
        "gate_cells": int(len(graw)),
        "gate_x_equal": int(graw.x_equal.sum()),
        "leg_records": {
            n: json.load(open(HUB / "numbers" / n / "MANIFEST.json"))["n_records"]
            for n in legs
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "leg_records"}))


if __name__ == "__main__":
    main()
