"""Tables for research/paper/results.md and the story page, from the FULL-RERUN (tag `final-freeze`) only.

Usage: build_results.py FR OUT

FR holds one directory per Hub subtree fr/<leg>-<pod>-s<seed>[-rerun], each fetched with `bench fetch` (results.parquet).
Every number is read from those records:

- A `-rerun` directory's records replace its leg's unstable cells; a cell unstable in every run keeps its record and the flag.
  OOM attempts superseded by a retry are `failed` lines and never reach results.parquet.
- The head-to-head is the timed bloom legs: SilverTorch (Meta's code, adapted) on its bloom filter with the recheck off,
  interleaved in one process with LiNR V1 / V2 (exact clause filter) and, at k 100, LiNR V3 at the smallest pool reaching
  0.90 / 0.95. LiNR has no bloom arm. Recall is exact filtered top-k recall (oracle); bloom false positives are counted
  untimed elsewhere.
- Seed 0 everywhere; the timed bloom legs also at seeds 1-2, folded into `variance` (median and range over seeds per cell).
- LiNR's claims read the final-pass suites and the exact LiNR legs, SilverTorch's the curves and the bloom width leg.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FR = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LINR = {"linr_v1_filter_mask": "V1", "linr_v2": "V2", "linr_v3": "V3"}
TARGETS = [0.9, 0.95]
CELL = ["dataset", "dim", "sweep", "perf_bs", "perf_k"]
RUN = re.compile(
    r"(?P<leg>.+)-(?P<pod>pod1|pode|podf)-s(?P<seed>\d)(?P<rerun>-rerun)?$"
)


def recall(df):
    k = df.perf_k.fillna(100)
    out = df["oracle_recall@100"].copy()
    for kk in (1000, 2048):
        col = f"oracle_recall@{kk}"
        if col in df:
            out = out.where(k != kk, df[col])
    return out


def load():
    """Every ok record of every fetched leg, with the rerun rule applied (module docstring)."""
    parts = []
    for d in sorted(FR.iterdir()):
        m = RUN.match(d.name)
        if not m or not (d / "results.parquet").is_file():
            continue
        x = pd.read_parquet(d / "results.parquet")
        owner = [re.search(r"-(pod1|pode)-", s.name) for s in d.iterdir() if s.is_dir()]
        owner = next((o[1] for o in owner if o), m["pod"])
        if m["pod"] == "podf" and m["seed"] == "2":
            owner = "pod1"  # the scope note: pod f runs pod 1's cells at seed 2
        parts.append(
            x.assign(
                leg=m["leg"],
                pod=owner,
                run_pod=m["pod"],
                run_seed=int(m["seed"]),
                rerun=bool(m["rerun"]),
            )
        )
    df = pd.concat(parts, ignore_index=True)
    df = df[(df.status == "ok") & ~df.dataset.str.startswith("yfcc")]
    df["dataset"] = df.dataset.str.replace(
        r"-s\d$", "", regex=True
    )  # a seed's synthetic draw
    p = df.params.map(json.loads)
    for key in [
        "n_lists",
        "n_probe",
        "candidate_pool_frac",
        "k_bits",
        "m_bits",
        "k_hash",
        "compile",
        "recheck",
        "bloom_path",
    ]:
        df[key] = p.map(lambda d, key=key: d.get(key))
    df["unstable"] = df.unstable.fillna(False).astype(bool)
    df["ext"] = df.leg.str.startswith(
        "ext-"
    )  # TUNE-EXT: points at 50 / 75 % of n_lists for cells short of 0.9
    df["recall"] = recall(df)
    df["pool"] = df.candidate_pool_frac
    key = [
        "leg",
        "pod",
        "suite",
        "dataset",
        "dim",
        "filter_kind",
        "sweep",
        "algo",
        "backend",
        "params",
        "seed",
        "perf_mode",
        "perf_bs",
        "perf_k",
    ]
    df = df.sort_values(["unstable", "rerun"], ascending=[True, False])
    return df.drop_duplicates(key, keep="first").reset_index(drop=True)


def timed_bloom(df):
    return df[df.leg.str.contains("timed-bloom") & df.perf_mode.notna()]


def head_to_head(tb, mode="graph"):
    """Every SilverTorch bloom point against the faster exact LiNR (V1 / V2) of its cell, same leg, seed and mode."""
    x = tb[tb.perf_mode == mode]
    on = ["leg", "pod", "seed"] + CELL
    st = x[(x.algo == "silvertorch") & (x.filter_kind == "bloom")]
    assert (st.recheck == False).all()  # noqa: E712  the study's SilverTorch bloom (recheck off)
    li = x[x.algo.isin(["linr_v1_filter_mask", "linr_v2"])].assign(
        v=lambda d: d.algo.map(LINR)
    )
    ms = li.pivot_table(index=on, columns="v", values="perf_median_ms")
    ms["linr_unstable"] = li.groupby(on).unstable.any()
    ms["linr_ms"] = ms[["V1", "V2"]].min(axis=1)
    ms["linr_best"] = ms[["V1", "V2"]].idxmin(axis=1)
    h = st.merge(ms.reset_index(), on=on)
    h = h.assign(
        st_ms=h.perf_median_ms,
        st_over_linr=h.perf_median_ms / h.linr_ms,
        flagged=h.unstable | h.linr_unstable,
    )
    cols = on + [
        "n_items",
        "pass_rate",
        "n_lists",
        "n_probe",
        "recall",
        "st_ms",
        "V1",
        "V2",
        "linr_best",
        "linr_ms",
        "st_over_linr",
        "flagged",
        "ext",
        "run_pod",
        "index_mib",
        "build_s",
    ]
    return h[cols].sort_values(on + ["n_lists", "n_probe"])


def v3_matched(tb, mode="graph"):
    """LiNR V3 in the k 100 timed legs: per cell and target, the fastest timed pool whose recall reaches the target."""
    x = tb[(tb.perf_mode == mode) & (tb.algo == "linr_v3")]
    rows = []
    for key, g in x.groupby(["pod", "seed"] + CELL):
        for t in TARGETS:
            ok = g[g.recall >= t]
            if len(ok):
                b = ok.loc[ok.perf_median_ms.idxmin()]
                rows.append(
                    dict(
                        zip(["pod", "seed"] + CELL, key),
                        target=t,
                        v3_pool=b.pool,
                        v3_recall=b.recall,
                        v3_ms=b.perf_median_ms,
                        v3_unstable=b.unstable,
                    )
                )
    return pd.DataFrame(rows)


def tuned(h, v3):
    """Per cell, seed and target: the fastest SilverTorch point reaching the target, against exact LiNR and against the
    best LiNR at that recall (V1 / V2 exact, or V3 at its smallest pool reaching the target). A cell's points may come from
    its timed leg and a TUNE-EXT leg; each point's ratio is to the LiNR timed in its own run, the cell's LiNR columns are the
    original leg's. A point timed in another run (TUNE-EXT, possibly on another pod) keeps its own run's ratio: its ratio to
    the best LiNR is its ratio to its run's exact LiNR, rescaled by the original leg's best-over-exact LiNR."""
    on = ["pod", "seed"] + CELL
    rows = []
    for key, g in h.groupby(on):
        base = g.sort_values("ext").iloc[0]
        for t in TARGETS:
            row = dict(
                zip(on, key),
                n_items=base.n_items,
                pass_rate=base.pass_rate,
                target=t,
                V1=base.V1,
                V2=base.V2,
            )
            row.update(
                linr_best=base.linr_best,
                linr_ms=base.linr_ms,
                max_recall=g.recall.max(),
            )
            ok = g[g.recall >= t]
            if len(ok):
                b = ok.loc[ok.st_ms.idxmin()]
                row.update(
                    leg=b.leg,
                    ext=b.ext,
                    probe_frac=b.n_probe / b.n_lists,
                    n_lists=b.n_lists,
                    n_probe=b.n_probe,
                    recall=b.recall,
                    st_ms=b.st_ms,
                    st_over_linr=b.st_over_linr,
                    run_pod=b.run_pod,
                    flagged=b.flagged,
                )
            rows.append(row)
    tn = pd.DataFrame(rows)
    if len(v3):
        tn = tn.merge(v3, on=on + ["target"], how="left")
    else:
        tn = tn.assign(
            v3_pool=np.nan, v3_recall=np.nan, v3_ms=np.nan, v3_unstable=np.nan
        )
    best = tn[["linr_ms", "v3_ms"]].min(axis=1)
    tn["matched_best"] = np.where(tn.v3_ms < tn.linr_ms, "V3", tn.linr_best)
    tn["matched_ms"] = best
    tn["st_over_matched"] = tn.st_over_linr * tn.linr_ms / best
    tn["winner_exact"] = np.select(
        [tn.st_ms.isna(), tn.st_over_linr < 1],
        ["not reached", "SilverTorch"],
        "LiNR " + tn.linr_best,
    )
    tn["winner_matched"] = np.select(
        [tn.st_ms.isna(), tn.st_over_matched < 1],
        ["not reached", "SilverTorch"],
        "LiNR " + tn.matched_best,
    )
    return tn


def at24(h):
    """The papers' production setting: 24 probes at the dataset's smallest timed n_lists."""
    small = h.groupby(["leg", "dataset"]).n_lists.transform("min")
    return h[(h.n_probe == 24) & (h.n_lists == small)]


def variance(h, tn):
    """Seed spread of the timed head-to-head: per point and per tuned cell, median and range over the seeds present."""
    pt = ["leg", "pod"] + CELL + ["n_lists", "n_probe"]
    a = (
        h.groupby(pt)
        .agg(
            seeds=("seed", "nunique"),
            ratio_med=("st_over_linr", "median"),
            ratio_min=("st_over_linr", "min"),
            ratio_max=("st_over_linr", "max"),
            st_ms_med=("st_ms", "median"),
            st_ms_min=("st_ms", "min"),
            st_ms_max=("st_ms", "max"),
            recall_med=("recall", "median"),
            recall_min=("recall", "min"),
            recall_max=("recall", "max"),
        )
        .reset_index()
    )
    tc = ["pod"] + CELL + ["target"]
    b = (
        tn.groupby(tc)
        .agg(
            seeds=("seed", "nunique"),
            reached=("st_ms", "count"),
            ratio_med=("st_over_linr", "median"),
            ratio_min=("st_over_linr", "min"),
            ratio_max=("st_over_linr", "max"),
            matched_med=("st_over_matched", "median"),
            matched_min=("st_over_matched", "min"),
            matched_max=("st_over_matched", "max"),
        )
        .reset_index()
    )
    return a, b


def clause_vs_bloom(df):
    """SilverTorch's exact clause filter (the clause reference row, built at tag `final-freeze-clause`; the bloom path is
    byte-identical) against its bloom filter (the timed bloom legs): separate runs, not interleaved, paired on pod, GPU,
    seed, point and mode. A pair with either side unstable is dropped. Bloom's time over the exact filter's, the recall each
    returns, and both runs' sampled SM clocks."""
    x = df[
        (df.algo == "silvertorch")
        & df.perf_mode.notna()
        & df.leg.str.contains("timed-clause|timed-bloom")
        & ~df.unstable
    ]
    on = ["pod", "run_pod", "env_gpu", "seed", "perf_mode", "n_lists", "n_probe"] + CELL
    val = ["perf_median_ms", "recall", "perf_sm_mhz", "leg"]
    cl = x[x.filter_kind == "clause"][on + val]
    bl = x[x.filter_kind == "bloom"][on + val + ["pass_rate", "n_items", "ext"]]
    m = bl.merge(cl, on=on, suffixes=("_bloom", "_clause"))
    m["bloom_over_clause"] = m.perf_median_ms_bloom / m.perf_median_ms_clause
    m["recall_cost"] = m.recall_clause - m.recall_bloom
    return m


def untimed():
    """The untimed step's outputs, per pod whose cells they grade: SilverTorch bloom false positives at the timed points
    (grade_bloom), the bloom width's false-positive rate (fp_adapted) and the C6 ceiling split (ceiling_adapted)."""
    fp, fw, ceil = [], [], []
    for d in sorted(FR.iterdir()):
        m = RUN.match(d.name)
        if not m:
            continue
        tag = dict(leg=m["leg"], run_pod=m["pod"])
        for f in d.glob("grade_bloom-d*.csv"):
            fp.append(pd.read_csv(f).assign(dim=int(f.stem.split("-d")[-1]), **tag))
        for f in d.glob("fp_adapted.csv"):
            fw.append(pd.read_csv(f).assign(**tag))
        for f in d.glob("ceiling/ceiling-*.json"):
            j = json.loads(f.read_text())
            ceil.append(
                dict(
                    dataset=j["dataset"],
                    sweep=j["sweep"],
                    pass_rate=j["pass_rate"],
                    n_lists=j["params"]["n_lists"],
                    n_probe=j["params"]["n_probe"],
                    **j["recall_oracle@100"],
                    **tag,
                )
            )
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    fp = cat(fp)
    if len(fp):
        fp = fp.drop(columns="code_version").sort_values(
            ["dataset", "sweep", "n_lists", "n_probe"]
        )
    fw = cat(fw)
    if len(fw):
        fw = fw.drop(columns="code_version")
    return fp, fw, pd.DataFrame(ceil)


def c1(df):
    """LiNR V2 / V1 against pass rate on the synthetic sets, k 100, graph, seed 0, one leg per dataset and batch (the first with
    three pass rates)."""
    x = df[
        df.algo.isin(["linr_v1_filter_mask", "linr_v2"])
        & (df.backend == "triton")
        & (df.perf_mode == "graph")
    ]
    x = x[
        (x.perf_k == 100)
        & (x.seed == 0)
        & (x.filter_kind == "clause")
        & x.dataset.str.contains("synth")
    ]
    rows, curves = [], []
    for (ds, dim, bs), grp in x.groupby(["dataset", "dim", "perf_bs"]):
        legs = sorted(
            grp.leg.unique(), key=lambda s: (not s.startswith(("final", "linr")), s)
        )
        for leg in legs:
            pv = (
                grp[grp.leg == leg]
                .pivot_table(index="pass_rate", columns="algo", values="perf_median_ms")
                .dropna()
                .sort_index()
            )
            if len(pv) >= 3 and pv.shape[1] == 2:
                break
        else:
            continue
        grp = grp[grp.leg == leg]
        r = (pv.linr_v2 / pv.linr_v1_filter_mask).tolist()
        ps = pv.index.tolist()
        n = int(grp.n_items.iloc[0])
        rows.append(
            dict(
                dataset=ds,
                dim=dim,
                n_items=n,
                bs=bs,
                leg=leg,
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


def crossover(ps, ratios):
    """First pass rate where V2 / V1 reaches 1, log-interpolated; None if V2 never loses or always loses."""
    for (p0, r0), (p1, r1) in zip(zip(ps, ratios), zip(ps[1:], ratios[1:])):
        if r0 < 1 <= r1:
            f = -np.log(r0) / (np.log(r1) - np.log(r0))
            return float(np.exp(np.log(p0) + f * (np.log(p1) - np.log(p0))))
    return None


def linr_real(df):
    """Exact LiNR V1 / V2 on the real filters, graph, seed 0: final pass, the exact LiNR legs and the timed bloom legs."""
    x = df[
        df.algo.isin(["linr_v1_filter_mask", "linr_v2"])
        & (df.backend == "triton")
        & (df.perf_mode == "graph")
    ]
    x = x[
        (x.seed == 0) & (x.filter_kind == "clause") & ~x.dataset.str.contains("synth")
    ]
    pv = x.pivot_table(
        index=[
            "leg",
            "pod",
            "dataset",
            "dim",
            "sweep",
            "pass_rate",
            "perf_bs",
            "perf_k",
        ],
        columns="algo",
        values="perf_median_ms",
    ).reset_index()
    pv = pv.rename(columns={"linr_v1_filter_mask": "V1", "linr_v2": "V2"})
    pv["v2_over_v1"] = pv.V2 / pv.V1
    return pv


def c2(tb):
    """LiNR V3 against V2 in the same process (the timed k 100 legs, seed 0): recall and time per pool."""
    x = tb[(tb.perf_mode == "graph") & (tb.seed == 0) & (tb.perf_k == 100)]
    on = ["leg", "pod"] + CELL
    v2 = x[x.algo == "linr_v2"][on + ["perf_median_ms"]].rename(
        columns={"perf_median_ms": "v2_ms"}
    )
    v3 = x[x.algo == "linr_v3"].merge(v2, on=on)
    v3["v3_over_v2"] = v3.perf_median_ms / v3.v2_ms
    return v3[
        on
        + [
            "n_items",
            "pass_rate",
            "pool",
            "recall",
            "perf_median_ms",
            "v2_ms",
            "v3_over_v2",
            "unstable",
        ]
    ]


def c2_pools(df):
    """V3's recall@100 per pool from the pool legs (quality, seed 0)."""
    x = df[df.leg.str.startswith("v3pool") & (df.algo == "linr_v3") & (df.seed == 0)]
    return x.drop_duplicates(["dataset", "sweep", "pool"])[
        ["dataset", "sweep", "pass_rate", "pool", "recall"]
    ].sort_values(["dataset", "sweep", "pool"])


def c3(df):
    x = df[df.suite.isin(["c3", "c3-real"]) & (df.perf_k == 100) & (df.seed == 0)]
    on = ["dataset", "dim", "sweep", "pass_rate", "perf_bs", "perf_mode"]
    tri = x[x.backend == "triton"][on + ["perf_median_ms"]]
    tor = x[x.backend == "torch"].assign(
        variant=lambda d: d.compile.fillna("eager torch")
    )
    m = tor.merge(tri, on=on, suffixes=("", "_triton"))
    m["torch_over_triton"] = m.perf_median_ms / m.perf_median_ms_triton
    return m[["suite"] + on + ["variant", "torch_over_triton"]]


def c4(df):
    """The bloom width leg (quality, seed 0): recall and index size per width; false positives come from the untimed step."""
    x = df[(df.leg == "bloomwidth") & (df.seed == 0)].drop_duplicates(
        ["dataset", "sweep", "m_bits", "k_hash", "n_lists", "n_probe"]
    )
    return x[
        [
            "dataset",
            "sweep",
            "pass_rate",
            "n_lists",
            "n_probe",
            "m_bits",
            "k_hash",
            "recall",
            "index_mib",
        ]
    ].sort_values(["dataset", "sweep", "k_hash", "m_bits"])


def c5(df):
    """Co-design (Meta's code as published): full-index bloom over probe-then-filter, eager, seed 0."""
    x = df[
        (df.algo == "silvertorch")
        & (df.backend == "official")
        & df.bloom_path.notna()
        & (df.perf_mode == "eager")
        & (df.seed == 0)
    ]
    if x.empty:
        return pd.DataFrame(
            columns=[
                "suite",
                "dataset",
                "n_items",
                "sweep",
                "n_lists",
                "n_probe",
                "perf_bs",
                "full",
                "partial",
                "full_over_partial",
            ]
        )
    keys = ["suite", "dataset", "n_items", "sweep", "n_lists", "n_probe", "perf_bs"]
    pv = (
        x.pivot_table(index=keys, columns="bloom_path", values="perf_median_ms")
        .dropna()
        .reset_index()
    )
    pv["full_over_partial"] = pv.full / pv.partial
    return pv


def c6(df):
    """The adapted code's probe curves (quality, seed 0), bloom (recheck off) and clause: the top recall per filter."""
    x = df[df.leg.str.contains("curve") & (df.seed == 0) & (df.algo == "silvertorch")]
    curves = x.drop_duplicates(
        ["dataset", "dim", "filter_kind", "sweep", "n_lists", "n_probe"]
    )
    curves = curves[
        [
            "dataset",
            "dim",
            "n_items",
            "filter_kind",
            "sweep",
            "pass_rate",
            "n_lists",
            "n_probe",
            "oracle_recall@100",
            "oracle_recall@1000",
        ]
    ]
    keys = ["dataset", "n_items", "filter_kind", "sweep", "pass_rate"]
    quarter = curves[curves.n_probe <= curves.n_lists / 4]
    top = quarter.groupby(keys).agg(top_recall100=("oracle_recall@100", "max"))
    top["top_recall100_any"] = curves.groupby(keys)["oracle_recall@100"].max()
    top = top.reset_index()
    return curves.sort_values(
        ["dataset", "filter_kind", "sweep", "n_lists", "n_probe"]
    ), top


def yambda(df):
    x = df[(df.leg == "yambda") & (df.seed == 0)]
    return x[
        [
            "algo",
            "filter_kind",
            "sweep",
            "pass_rate",
            "n_lists",
            "n_probe",
            "oracle_recall@100",
            "index_mib",
            "build_s",
        ]
    ].sort_values(["algo", "sweep", "n_lists", "n_probe"])


def memory(df):
    """Index size and build time per arm, from the timed bloom legs (seed 0)."""
    x = timed_bloom(df)
    x = x[(x.seed == 0) & x.index_mib.notna()].assign(
        arm=lambda d: d.algo.map(LINR).fillna("SilverTorch")
    )
    g = x.groupby(["dataset", "dim", "n_items", "arm", "n_lists"], dropna=False).agg(
        index_mib=("index_mib", "median"),
        build_s=("build_s", "median"),
        reserved_mib=("memory_reserved_mib", "median"),
    )
    return g.reset_index()


def md(t, floatfmt="{:.3g}"):
    cols = list(t.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in t.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                ""
                if isinstance(v, float) and np.isnan(v)
                else floatfmt.format(v)
                if isinstance(v, float)
                else str(v)
                for v in r
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    tb = timed_bloom(df)
    h = head_to_head(tb)
    tn = tuned(h, v3_matched(tb))
    var_pt, var_tn = variance(h, tn)
    c1t, c1c = c1(df)
    c6c, c6t = c6(df)
    fp, fw, ceil = untimed()
    tables = {
        "tuned": tn,
        "at24": at24(h),
        "h2h": h,
        "h2h_eager": head_to_head(tb, "eager"),
        "variance_points": var_pt,
        "variance_tuned": var_tn,
        "clause_vs_bloom": clause_vs_bloom(df),
        "c1_crossover": c1t,
        "c1_curves": c1c,
        "linr_real": linr_real(df),
        "c2_v3": c2(tb),
        "c2_pools": c2_pools(df),
        "c3_torch": c3(df),
        "c4_bloomwidth": c4(df),
        "c4_fp_points": fp,
        "c4_fp_width": fw,
        "c5_meta": c5(df),
        "c6_curves": c6c,
        "c6_top": c6t,
        "c6_ceiling": ceil,
        "yambda": yambda(df),
        "memory": memory(df),
    }
    with open(OUT / "tables.md", "w") as f:
        f.write(
            "# results-draft tables from the FULL-RERUN (tag final-freeze). NOT CITABLE until the FR gate is green.\n\n"
        )
        for name, t in tables.items():
            t.to_csv(OUT / f"{name}.csv", index=False)
            f.write(f"## {name}\n\n{md(t)}\n\n")
    df.drop(columns=["per_query"], errors="ignore").to_parquet(OUT / "results.parquet")
    runs = df.groupby(["leg", "pod", "run_seed", "rerun"]).size()
    summary = {
        "records": int(len(df)),
        "runs": {
            f"{lg}-{p}-s{s}{'-rerun' if r else ''}": int(n)
            for (lg, p, s, r), n in runs.items()
        },
        "seeds_timed": sorted(int(s) for s in tb.seed.unique()),
        "h2h_rows": int(len(h)),
        "h2h_flagged": int(h.flagged.sum()),
        "tuned_rows": int(len(tn)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}))


if __name__ == "__main__":
    main()
