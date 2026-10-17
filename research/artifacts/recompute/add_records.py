"""Extend the export's results/records/records.parquet from the Hub build's results.parquet.

usage: add_records.py BUILD_PARQUET EXPORT_RECORDS OUT_RECORDS

The existing records keep their rows, order and values; each gains n_queries_oracle and heldout_recall@{100,1000}
from its build row. New experiments are appended with new opaque run ids (one per build (leg, pod)):

- native_pytorch: the c3 / c3-real legs (Triton V1 and native PyTorch V1, eager and compiled, both modes) and the
  synth leg's native V2 rows with their Triton V1 / V2 eager twins (the graph twins are linr_pass_rate already).
- v3_code_length: the V3 bits sweep (graph).
- recall_bridge: Yambda (quality only) and the Goodreads c0_genre Bloom quality-curve points.
- reverse_clause: SilverTorch's exact clause filter on the four reverse-clause filters and the exact LiNR legs on the
  same filters (graph, seed 0, k 100).
"""

import sys

import polars as pl

BUILD, OLD, NEW = sys.argv[1:4]
BACKEND = {"official-fork": "cuda", "official": "published", "triton": "triton", "torch": "torch"}
ADDED = ["n_queries_oracle", "heldout_recall@100", "heldout_recall@1000"]
REVERSE = {"goodreads": ["c1_lang_reverse", "all4"], "pubmed": ["c3_journal_reverse", "all5"]}
KEY = ["dataset", "dim", "sweep", "algo", "backend", "params", "seed", "perf_mode", "perf_bs", "perf_k", "filter_kind"]

b = pl.read_parquet(BUILD).with_columns(pl.col("backend").replace_strict(BACKEND))
old = pl.read_parquet(OLD)
cols = old.columns

# run id <- build (leg, pod): the majority leg among each run's rows matched on the timed median
timed = old.filter(pl.col("perf_median_ms").is_not_null())
m = timed.join(b.select("leg", "pod", *KEY, "perf_median_ms"), on=[*KEY, "perf_median_ms"])
runs = (
    m.group_by("run", "leg", "pod")
    .len()
    .sort("len", descending=True)
    .group_by("run", maintain_order=True)
    .first()
    .select("run", "leg", "pod")
)
bw = old.filter(pl.col("experiment") == "bloom_width")["run"].unique().to_list()
runs = pl.concat([runs, pl.DataFrame({"run": bw, "leg": ["bloomwidth"], "pod": ["pode"]}, schema=runs.schema)])
assert runs["run"].n_unique() == runs.height == old["run"].n_unique()

src = b.join(runs, on=["leg", "pod"], how="left")
on = ["run", *KEY]
ext = old.join(src.filter(pl.col("run").is_not_null()).select(*on, *ADDED), on=on, how="left", validate="m:1", nulls_equal=True)
assert ext.height == old.height and ext["n_queries_oracle"].null_count() == 0, "every record has its build row"
assert ext.select(cols).equals(old)

next_run = [old["run"].max() + 1]


def tag(x, experiment):
    """Give unmapped (leg, pod) a fresh run id, in sorted order."""
    new = x.filter(pl.col("run").is_null()).select("leg", "pod").unique().sort("leg", "pod")
    ids = pl.DataFrame(
        {"leg": new["leg"], "pod": new["pod"], "new_run": range(next_run[0], next_run[0] + new.height)},
        schema={"leg": pl.String, "pod": pl.String, "new_run": pl.Int32},
    )
    next_run[0] += new.height
    global src
    src = src.join(ids, on=["leg", "pod"], how="left").with_columns(run=pl.coalesce("run", "new_run")).drop("new_run")
    x = x.join(ids, on=["leg", "pod"], how="left").with_columns(run=pl.coalesce("run", "new_run")).drop("new_run")
    return x.with_columns(experiment=pl.lit(experiment), followup=pl.lit(False))


have = ext.select(on)
fresh = lambda x: x.join(have, on=on, how="anti", nulls_equal=True)  # noqa: E731
leg = pl.col("leg")
timed_ok = pl.col("perf_median_ms").is_not_null()

native = fresh(
    src.filter(
        timed_ok
        & (
            leg.is_in(["final-c3", "final-c3-real"])
            | (
                (leg == "final-synth")
                & (pl.col("dataset") == "goodreads-synth")
                & pl.col("sweep").is_in(["p001", "p01", "p1"])
                & pl.col("algo").is_in(["linr_v1_filter_mask", "linr_v2"])
            )
        )
        & (pl.col("seed") == 0)
        & (pl.col("perf_k") == 100)
    )
)
native = tag(native, "native_pytorch")

bits = tag(fresh(src.filter((leg == "final-v3bits") & timed_ok & (pl.col("perf_mode") == "graph"))), "v3_code_length")

bridge = src.filter(
    (leg == "yambda")
    | (
        (leg == "curve-d128")
        & (pl.col("dataset") == "goodreads")
        & (pl.col("sweep") == "c0_genre")
        & (pl.col("filter_kind") == "bloom")
        & (pl.col("algo") == "silvertorch")
    )
).filter(pl.col("seed") == 0)
bridge = tag(fresh(bridge), "recall_bridge")

rev = pl.lit(False)
for ds, sws in REVERSE.items():
    rev = rev | ((pl.col("dataset") == ds) & pl.col("sweep").is_in(sws))
clause = leg.str.starts_with("timed-clause-lite") & (pl.col("algo") == "silvertorch")
exact = leg.is_in(["final-filter", "linr"]) & pl.col("algo").is_in(["linr_v1_filter_mask", "linr_v2"])
reverse = src.filter(
    rev & (clause | exact) & timed_ok & (pl.col("perf_mode") == "graph") & (pl.col("seed") == 0) & (pl.col("perf_k") == 100)
)
reverse = tag(fresh(reverse), "reverse_clause")

new = pl.concat([x.select(*cols, *ADDED) for x in (native, bits, bridge, reverse)])
assert new.select(on).is_unique().all()
out = pl.concat([ext.select(*cols, *ADDED), new])
out.write_parquet(NEW)
print(out.group_by("experiment").agg(pl.len(), pl.col("run").unique().sort()).sort("experiment"))
print("records", old.height, "->", out.height)
