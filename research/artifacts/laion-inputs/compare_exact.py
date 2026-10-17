"""exact_ids.sh's records against the final run's (records.py): per (dataset, filter), the counts, recall and the
result-id sha256 of exact LiNR V1, bs 16, k 100, graph.

usage: compare_exact.py NEW_RESULTS_DIR RECORDS.parquet"""

import sys

import polars as pl

sys.path.insert(0, __import__("os").path.dirname(__file__))
from records import rows_of  # noqa: E402

new = rows_of(sys.argv[1], statuses=("ok", "partial"))  # partial: graph mode only
old = pl.read_parquet(sys.argv[2])
key = ["dataset", "sweep", "algo", "k", "bs", "mode"]
cols = ["n_kept", "n_queries_heldout", "n_targets_in_filter", "pass_rate", "oracle_recall", "heldout_recall",
        "ids_sha256"]  # fmt: skip
sel = (
    (pl.col("algo") == "linr_v1_filter_mask")
    & (pl.col("k") == 100)
    & (pl.col("bs") == 16)
    & (pl.col("mode") == "graph")
)
o = (
    old.filter(sel & (pl.col("filter_kind") == "clause"))
    .group_by(key)
    .agg(*[pl.col(c).unique().alias(f"old_{c}") for c in cols])
)
j = new.filter(sel).select(key + cols).join(o, on=key, how="left")
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_fmt_str_lengths(20)
for c in cols:
    j = j.with_columns(
        pl.struct(c, f"old_{c}")
        .map_elements(
            lambda r, c=c: r[c] in (r[f"old_{c}"] or []), return_dtype=pl.Boolean
        )
        .alias(f"eq_{c}")
    )
print(j.select("dataset", "sweep", *[f"eq_{c}" for c in cols]))
print(
    j.select(
        "dataset",
        "sweep",
        "oracle_recall",
        "old_oracle_recall",
        "heldout_recall",
        "old_heldout_recall",
        "n_targets_in_filter",
        "old_n_targets_in_filter",
    )
)
j.write_parquet(sys.argv[1].rstrip("/") + "-compare.parquet")
