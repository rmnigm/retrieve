"""Rebuild the paper's table and figure data from the final run's records.

usage: python build_tables.py [RECORDS] [OUT]   (defaults: records/records.parquet, tables/)

RECORDS is one Parquet file of `bench` records (one row per timed entry, the harness's own column
names) with three columns added: `experiment` (which table a record feeds), `run` (records with
the same run were timed in one process, interleaved) and `followup` (points added later at deeper
probes, timed in their own run). The two untimed measurements sit next to it as CSVs. README.md
states the selection rules; they are the constants below.
"""

import json
import math
import sys
from pathlib import Path

import polars as pl

RECORDS = Path(sys.argv[1] if len(sys.argv) > 1 else "records/records.parquet")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "tables")
SCORE_PATH = RECORDS.parent / "score_path_diagnostic.csv"
FALSE_POSITIVES = RECORDS.parent / "bloom_false_positives.csv"

TARGETS = [0.90, 0.95]
PROBE_BUDGET = 0.25  # SilverTorch may probe at most a quarter of its lists
SEED = 0  # seeds 1-2 of the head-to-head runs stay in the records; no table reads them
V1, V2, V3 = "linr_v1_filter_mask", "linr_v2", "linr_v3"
ARM = {V1: "V1", V2: "V2", V3: "V3", "silvertorch": "SilverTorch"}
CELL = ["dataset", "dim", "sweep", "perf_bs", "perf_k"]
SELECTED = [  # the representative real-filter selections (k 100): dataset, filter, target, batch
    ("goodreads", "c0_genre", 0.95, 1),
    ("arxiv", "c0_maincat", 0.95, 1),
    ("pubmed", "c0_mesh", 0.95, 1),
    ("laion10m", "tags4", 0.95, 1),
    ("laion10m", "tags4", 0.95, 64),
    ("laion30m", "c0_domain", 0.90, 1),
]
NEAR_TEN = (0.09, 0.11)  # pass rates read as "near 10 %"
BROAD = (0.19, 0.51)  # the 20 / 30 / 50 % synthetic rates
FIXED = (4096, 1024)  # the fixed probe setting: n_lists, n_probe
LAION = ["laion1m", "laion3m", "laion10m", "laion30m"]


def param(key, dtype):
    return (
        pl.col("params")
        .map_elements(lambda s: json.loads(s).get(key), return_dtype=dtype)
        .alias(key)
    )


def load():
    r = pl.read_parquet(RECORDS).with_columns(
        param("n_lists", pl.Int64),
        param("n_probe", pl.Int64),
        param("candidate_pool_frac", pl.Float64),
        param("recheck", pl.Boolean),
        param("bloom_path", pl.String),
        param("m_bits", pl.Int64),
        param("k_hash", pl.Int64),
    )
    recall = (
        pl.when(pl.col("perf_k") == 1000)
        .then(pl.col("oracle_recall@1000"))
        .when(pl.col("perf_k") == 2048)
        .then(pl.col("oracle_recall@2048"))
        .otherwise(pl.col("oracle_recall@100"))
    )
    return r.with_columns(
        recall=recall,
        synthetic=pl.col("dataset").str.ends_with("-synth"),
        perf_bs=pl.col("perf_bs").cast(pl.Int64),
        perf_k=pl.col("perf_k").cast(pl.Int64),
    )


def exact_linr(x, on):
    """The faster exact LiNR (V1 / V2) per group, with both times."""
    li = x.filter(pl.col("algo").is_in([V1, V2]))
    return (
        li.group_by(on)
        .agg(
            v1_ms=pl.col("perf_median_ms").filter(pl.col("algo") == V1).first(),
            v2_ms=pl.col("perf_median_ms").filter(pl.col("algo") == V2).first(),
            exact_unstable=pl.col("unstable").any(),
        )
        .with_columns(
            exact_ms=pl.min_horizontal("v1_ms", "v2_ms"),
            exact_variant=pl.when(pl.col("v2_ms") < pl.col("v1_ms"))
            .then(pl.lit("V2"))
            .otherwise(pl.lit("V1")),
        )
    )


def fastest_reaching(x, key, cols):
    """Per key and recall target, the fastest row whose recall reaches the target."""
    return (
        x.join(pl.DataFrame({"target": TARGETS}), how="cross")
        .filter(pl.col("recall") >= pl.col("target"))
        .sort("perf_median_ms", maintain_order=True)
        .group_by([*key, "target"], maintain_order=True)
        .first()
        .select(*key, "target", **cols)
    )


def head_to_head(r):
    """Per workload (dataset, dim, filter, batch, k) and recall target, seed 0: the fastest SilverTorch point
    within the probe budget reaching the target, against the fastest LiNR reaching it (exact V1 / V2, or V3
    at its fastest pool reaching the target). A follow-up point's ratio is to the exact LiNR of its own run,
    rescaled by the workload's original run's best LiNR over exact LiNR."""
    x = r.filter(
        (pl.col("experiment") == "head_to_head") & (pl.col("perf_mode") == "graph")
    )
    st = x.filter(
        (pl.col("algo") == "silvertorch") & (pl.col("filter_kind") == "bloom")
    )
    assert not st["recheck"].any(), "the study's bloom runs with the recheck off"
    on = ["run", "seed", *CELL]
    h = (
        st.join(exact_linr(x, on), on=on)
        .with_columns(
            st_over_exact=pl.col("perf_median_ms") / pl.col("exact_ms"),
            flagged=pl.col("unstable") | pl.col("exact_unstable"),
        )
        .filter(pl.col("n_probe") <= PROBE_BUDGET * pl.col("n_lists"))
        .sort([*on, "n_lists", "n_probe"])
    )
    key = ["seed", *CELL]
    base = (
        h.sort("followup", maintain_order=True)
        .group_by(key, maintain_order=True)
        .first()
        .select(*key, "pass_rate", "synthetic", "exact_ms", "exact_variant")
    )
    picks = fastest_reaching(
        h,
        key,
        dict(
            st_n_lists="n_lists",
            st_n_probe="n_probe",
            st_recall="recall",
            st_ms="perf_median_ms",
            st_over_exact="st_over_exact",
            flagged="flagged",
        ),
    )
    v3 = fastest_reaching(
        x.filter(pl.col("algo") == V3),
        key,
        dict(
            v3_pool="candidate_pool_frac",
            v3_ms="perf_median_ms",
            v3_unstable="unstable",
        ),
    )
    t = (
        base.join(pl.DataFrame({"target": TARGETS}), how="cross")
        .join(picks, on=[*key, "target"], how="left")
        .join(v3, on=[*key, "target"], how="left")
        .with_columns(linr_ms=pl.min_horizontal("exact_ms", "v3_ms"))
        .with_columns(
            linr_variant=pl.when(pl.col("v3_ms") < pl.col("exact_ms"))
            .then(pl.lit("V3"))
            .otherwise(pl.col("exact_variant")),
            st_over_linr=pl.col("st_over_exact")
            * pl.col("exact_ms")
            / pl.col("linr_ms"),
            reached=pl.col("st_ms").is_not_null(),
            unstable=pl.col("flagged").fill_null(False)
            | pl.col("v3_unstable").fill_null(False),
        )
    )
    return (
        t.filter(pl.col("seed") == SEED)
        .sort(["synthetic", "dataset", "dim", "sweep", "perf_bs", "perf_k", "target"])
        .select(
            "dataset",
            "dim",
            pl.col("sweep").alias("filter"),
            pl.when("synthetic")
            .then(pl.lit("synthetic"))
            .otherwise(pl.lit("real"))
            .alias("filter_type"),
            "pass_rate",
            pl.col("perf_bs").alias("batch"),
            pl.col("perf_k").alias("k"),
            pl.col("target").alias("recall_target"),
            pl.col("reached").alias("silvertorch_reached_target"),
            pl.col("st_n_lists").alias("silvertorch_n_lists"),
            pl.col("st_n_probe").alias("silvertorch_n_probe"),
            pl.col("st_recall").alias("silvertorch_recall"),
            pl.col("st_ms").alias("silvertorch_ms"),
            "linr_variant",
            pl.when(pl.col("linr_variant") == "V3")
            .then("v3_pool")
            .alias("linr_v3_pool"),
            "linr_ms",
            pl.col("st_over_linr").alias("silvertorch_over_linr"),
            pl.col("unstable").alias("timing_unstable"),
        )
    )


def head_to_head_summary(sel):
    return (
        sel.group_by("filter_type", "recall_target")
        .agg(
            timed_combinations=pl.len(),
            target_reached=pl.col("silvertorch_reached_target").sum(),
            silvertorch_faster=(pl.col("silvertorch_over_linr") < 1).sum(),
            median_silvertorch_over_linr=pl.col("silvertorch_over_linr").median(),
        )
        .sort("filter_type", "recall_target")
    )


def selected_points(sel):
    rows = [
        sel.filter(
            (pl.col("dataset") == ds)
            & (pl.col("filter") == sw)
            & (pl.col("recall_target") == t)
            & (pl.col("batch") == bs)
            & (pl.col("k") == 100)
        )
        for ds, sw, t, bs in SELECTED
    ]
    assert all(len(x) == 1 for x in rows)
    return pl.concat(rows).select(
        "dataset",
        "filter",
        "batch",
        "recall_target",
        "silvertorch_n_lists",
        "silvertorch_n_probe",
        "silvertorch_recall",
        "silvertorch_ms",
        "linr_variant",
        "linr_ms",
    )


def crossover(rates, ratios):
    """First pass rate where V2 / V1 reaches 1, interpolated in log-log; None if it never crosses."""
    for p0, r0, p1, r1 in zip(rates, ratios, rates[1:], ratios[1:]):
        if r0 < 1 <= r1:
            f = -math.log(r0) / (math.log(r1) - math.log(r0))
            return math.exp(math.log(p0) + f * (math.log(p1) - math.log(p0)))
    return None


def linr_v2_over_v1(r):
    """LiNR V2 / V1 against the synthetic pass rate (exact filter, k 100, seed 0), both timed in one run."""
    x = r.filter(pl.col("experiment") == "linr_pass_rate")
    on = ["run", "dataset", "dim", "n_items", "sweep", "pass_rate", "perf_bs"]
    return (
        exact_linr(x, on)
        .with_columns(v2_over_v1=pl.col("v2_ms") / pl.col("v1_ms"))
        .sort("dataset", "dim", "perf_bs", "pass_rate")
        .select(
            "dataset",
            "dim",
            "n_items",
            pl.col("perf_bs").alias("batch"),
            pl.col("sweep").alias("filter"),
            "pass_rate",
            "v1_ms",
            "v2_ms",
            "v2_over_v1",
            pl.col("exact_unstable").alias("timing_unstable"),
        )
    )


def linr_v3_vs_exact(r):
    """LiNR V3 at every timed pool against the faster exact variant of the same run (seed 0, k 100)."""
    x = r.filter(
        (pl.col("experiment") == "head_to_head")
        & (pl.col("perf_mode") == "graph")
        & (pl.col("seed") == SEED)
        & (pl.col("perf_k") == 100)
    )
    on = ["run", *CELL]
    return (
        x.filter(pl.col("algo") == V3)
        .join(exact_linr(x, on), on=on)
        .with_columns(v3_over_exact=pl.col("perf_median_ms") / pl.col("exact_ms"))
        .sort("dataset", "dim", "sweep", "perf_bs", "candidate_pool_frac", "followup")
        .select(
            "dataset",
            "dim",
            pl.col("sweep").alias("filter"),
            "pass_rate",
            pl.col("perf_bs").alias("batch"),
            pl.col("candidate_pool_frac").alias("v3_pool"),
            pl.col("recall").alias("v3_recall"),
            pl.col("perf_median_ms").alias("v3_ms"),
            "exact_variant",
            "exact_ms",
            "v3_over_exact",
            (pl.col("unstable") | pl.col("exact_unstable")).alias("timing_unstable"),
            pl.col("followup").alias("followup_run"),
        )
    )


def probe_sweep(r):
    """SilverTorch's whole n_probe range on LAION 30M with synthetic filters (batch 16, k 100), and LiNR V1 timed
    once per filter in the same run."""
    x = r.filter(pl.col("experiment") == "probe_sweep")
    v1 = x.filter(pl.col("algo") == V1).select("sweep", v1_ms="perf_median_ms")
    return (
        x.filter(pl.col("algo") == "silvertorch")
        .join(v1, on="sweep")
        .sort("pass_rate", "n_lists", "n_probe")
        .select(
            "dataset",
            pl.col("sweep").alias("filter"),
            "pass_rate",
            pl.col("perf_bs").alias("batch"),
            "n_lists",
            "n_probe",
            pl.col("recall").alias("silvertorch_recall"),
            pl.col("perf_median_ms").alias("silvertorch_ms"),
            "v1_ms",
            pl.col("unstable").alias("timing_unstable"),
        )
    )


def fixed_vs_tuned(sweep, target=0.90):
    """Per filter at FIXED's n_lists: the fastest swept n_probe reaching the target against the fixed n_probe."""
    x = sweep.filter(pl.col("n_lists") == FIXED[0])
    best = (
        x.filter(pl.col("silvertorch_recall") >= target)
        .sort("silvertorch_ms")
        .group_by("filter")
        .first()
        .select("filter", tuned_n_probe="n_probe", tuned_ms="silvertorch_ms")
    )
    fixed = x.filter(pl.col("n_probe") == FIXED[1]).select(
        "filter", fixed_ms="silvertorch_ms"
    )
    return best.join(fixed, on="filter").with_columns(
        fixed_over_tuned=pl.col("fixed_ms") / pl.col("tuned_ms")
    )


def codesign_pairs(r):
    """Meta's published code, eager, seed 0: the bloom filter over the whole index against filtering only the
    searched clusters, same run and point. LAION 30M was timed in two runs: `repeat` 1 and 2."""
    x = r.filter(pl.col("experiment") == "codesign")
    on = ["run", "dataset", "dim", "n_items", "sweep", "pass_rate"]
    on += ["perf_bs", "perf_k", "n_lists", "n_probe"]
    keep = ["perf_median_ms", "unstable"]
    full = x.filter(pl.col("bloom_path") == "full").select(*on, *keep)
    part = x.filter(pl.col("bloom_path") == "partial").select(*on, *keep)
    return (
        full.join(part, on=on, suffix="_partial", validate="1:1")
        .with_columns(
            repeat=pl.col("run")
            .rank("dense")
            .over("dataset", "sweep", "perf_bs", "n_lists", "n_probe")
        )
        .sort("n_items", "dataset", "sweep", "perf_bs", "n_lists", "n_probe", "repeat")
        .select(
            "dataset",
            "dim",
            "n_items",
            pl.col("sweep").alias("filter"),
            "pass_rate",
            pl.col("perf_bs").alias("batch"),
            pl.col("perf_k").alias("k"),
            "n_lists",
            "n_probe",
            "repeat",
            pl.col("perf_median_ms").alias("whole_index_ms"),
            pl.col("perf_median_ms_partial").alias("searched_clusters_ms"),
            (pl.col("perf_median_ms") / pl.col("perf_median_ms_partial")).alias(
                "whole_index_over_searched_clusters"
            ),
            (pl.col("unstable") | pl.col("unstable_partial")).alias("timing_unstable"),
        )
    )


def bloom_width(r, fp):
    """SilverTorch's bloom filter per width and hash count (seed 0): recall@100 and index size from the
    quality-only run, false positives over the whole catalogue from the untimed measurement."""
    x = r.filter(pl.col("experiment") == "bloom_width").select(
        "dataset",
        pl.col("sweep").alias("filter"),
        "pass_rate",
        "m_bits",
        "k_hash",
        pl.col("recall").alias("recall_at_100"),
        "index_mib",
    )
    return x.join(
        fp, on=["dataset", "filter", "m_bits", "k_hash"], validate="1:1"
    ).sort("dataset", "filter", "k_hash", "m_bits")


def index_memory(r):
    """Resident index size per arm from the head-to-head runs (seed 0): the median over their records."""
    x = r.filter(
        (pl.col("experiment") == "head_to_head")
        & (pl.col("seed") == SEED)
        & pl.col("index_mib").is_not_null()
    )
    return (
        x.with_columns(arm=pl.col("algo").replace_strict(ARM))
        .group_by("dataset", "dim", "n_items", "arm", "n_lists")
        .agg(index_mib=pl.col("index_mib").median())
        .sort("dataset", "dim", "arm", "n_lists", nulls_last=True)
    )


def row(claim, quantity, condition, values, kind="ratio"):
    v = pl.Series(values, dtype=pl.Float64)
    return dict(
        claim=claim,
        quantity=quantity,
        condition=condition,
        n=len(v),
        min=rounded(v.min(), kind),
        max=rounded(v.max(), kind),
    )


def rounded(x, kind):
    """`min` / `max` hold several kinds of quantity, so they are written as text at each kind's precision."""
    if kind == "count":
        return str(round(x))
    if kind == "pass_rate":
        return f"{x:.4g}"
    return str(round(x, 3))


def claims(v2v1, v3, sweep, codesign, diag):
    rows = []
    c = "LiNR V1/V2 ordering"
    near = v2v1.filter(
        pl.col("pass_rate").is_between(*NEAR_TEN)
        & ~pl.col("timing_unstable")
        & pl.col("dataset").is_in([f"{d}-synth" for d in LAION])
    )
    for bs in (1, 16):
        for d in LAION:
            v = near.filter(
                (pl.col("batch") == bs) & (pl.col("dataset") == f"{d}-synth")
            )["v2_over_v1"]
            rows.append(
                row(
                    c,
                    "V2 / V1 latency",
                    f"{d}-synth, pass rate near 10 %, batch {bs}",
                    v,
                )
            )
    cross = []
    for (ds, _dim), g in v2v1.filter(pl.col("batch") == 1).group_by(
        "dataset", "dim", maintain_order=True
    ):
        if "corr" in ds:
            continue
        g = g.sort("pass_rate")
        p = crossover(g["pass_rate"].to_list(), g["v2_over_v1"].to_list())
        if p is not None:
            cross.append(p)
    rows.append(
        row(
            c,
            "pass rate where V2 / V1 reaches 1",
            "uniform synthetic filters, batch 1",
            cross,
            "pass_rate",
        )
    )

    c = "LiNR V3 shortlist"
    broad = v3.filter(
        pl.col("dataset").str.ends_with("-synth")
        & pl.col("pass_rate").is_between(*BROAD, closed="none")
        & ~pl.col("timing_unstable")
    )
    for bs in (1, 16, 64):
        v = broad.filter(pl.col("batch") == bs)["v3_over_exact"]
        rows.append(
            row(
                c,
                "V3 / faster exact latency",
                f"20-50 % synthetic, LAION 10M/30M, batch {bs}",
                v,
            )
        )
    v = broad.filter(pl.col("batch") == 1)["v3_recall"]
    rows.append(row(c, "V3 recall@100", "same configurations", v, "recall"))

    c = "SilverTorch probes and precision"
    ft = fixed_vs_tuned(sweep)
    rows.append(
        row(
            c,
            "fastest n_probe reaching recall 0.90",
            "LAION 30M synthetic, 4,096 lists, batch 16",
            ft["tuned_n_probe"],
            "count",
        )
    )
    rows.append(
        row(c, "fixed 1,024 probes / tuned latency", "same", ft["fixed_over_tuned"])
    )
    for ds, sw in (("pubmed", "c3_journal_reverse"), ("laion1m", "c0_domain")):
        d = diag.filter((pl.col("dataset") == ds) & (pl.col("filter") == sw))
        where = f"{ds} {sw}, same probed clusters"
        rows.append(
            row(
                c,
                "recall@100, int8 scores, fp16 output",
                where,
                d["recall_at_100_int8_fp16_out"],
                "recall",
            )
        )
        rows.append(
            row(
                c,
                "recall@100, fp16 inputs, fp32 accumulation",
                where,
                d["recall_at_100_fp16_fp32_acc"],
                "recall",
            )
        )

    c = "SilverTorch co-design"
    stable = codesign.filter(~pl.col("timing_unstable"))
    for d in LAION:
        v = stable.filter(pl.col("dataset") == d)["whole_index_over_searched_clusters"]
        rows.append(
            row(
                c,
                "whole-index / searched-cluster filtering latency",
                f"{d}, stable pairs",
                v,
            )
        )
    return pl.DataFrame(rows)


def tidy(t):
    """The precision the paper reports: latency and ratios to 3 decimals, recall to 3 decimals, pass and
    false-positive rates to 4 significant digits, index size to whole MiB."""
    out = []
    for c, dtype in t.schema.items():
        col = pl.col(c)
        if dtype != pl.Float64 or c in ("recall_target", "v3_pool", "linr_v3_pool"):
            out.append(col)
        elif c == "pass_rate" or c.startswith("fp_rate"):
            out.append(col.round_sig_figs(4))
        elif c == "index_mib":
            out.append(col.round(0).cast(pl.Int64))
        else:
            out.append(col.round(3))
    return t.select(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = load()
    sel = head_to_head(r)
    v2v1 = linr_v2_over_v1(r)
    v3 = linr_v3_vs_exact(r)
    sweep = probe_sweep(r)
    codesign = codesign_pairs(r)
    tables = {
        "claims_summary": claims(v2v1, v3, sweep, codesign, pl.read_csv(SCORE_PATH)),
        "head_to_head_summary": head_to_head_summary(sel),
        "head_to_head_selections": sel,
        "selected_points": selected_points(sel),
        "linr_v2_over_v1": v2v1,
        "linr_v3_vs_exact": v3,
        "probe_sweep": sweep,
        "codesign_pairs": codesign,
        "bloom_width": bloom_width(r, pl.read_csv(FALSE_POSITIVES)),
        "index_memory": index_memory(r),
    }
    for name, t in tables.items():
        tidy(t).write_csv(OUT / f"{name}.csv")
        print(f"{name}.csv: {t.height} rows")


if __name__ == "__main__":
    main()
