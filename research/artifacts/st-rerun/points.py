"""SilverTorch point by point: the final run's recall and graph median against the rerun's, same
(dataset, sweep, bs, k, n_lists, n_probe), seed 0. LiNR likewise per variant, as the cross-run sanity check.

usage: python points.py RECORDS_PARQUET RESULTS_DIR OUT_CSV
RECORDS_PARQUET = results/records/records.parquet on origin/export (head-to-head, graph, seed 0).
"""

import json
import sys
from pathlib import Path

import polars as pl

old = (
    pl.read_parquet(sys.argv[1])
    .filter((pl.col("experiment") == "head_to_head") & (pl.col("perf_mode") == "graph") & (pl.col("seed") == 0))
    .with_columns(p=pl.col("params").map_elements(lambda s: json.dumps(json.loads(s), sort_keys=True), return_dtype=pl.String))
    .with_columns(
        old_recall=pl.when(pl.col("perf_k") == 1000)
        .then(pl.col("oracle_recall@1000"))
        .when(pl.col("perf_k") == 2048)
        .then(pl.col("oracle_recall@2048"))
        .otherwise(pl.col("oracle_recall@100"))
    )
    .group_by("dataset", "sweep", "algo", "p", "perf_bs", "perf_k")
    .agg(old_ms=pl.col("perf_median_ms").min(), old_recall=pl.col("old_recall").first(), runs=pl.len())
)
rows = []
for f in Path(sys.argv[2]).glob("*/*.jsonl"):
    if f.name.endswith(".samples.jsonl"):
        continue
    for line in open(f):
        r = json.loads(line)
        if r["status"] == "failed":
            continue
        for e in r["perf"] or []:
            if e["mode"] == "graph" and e.get("median_ms") is not None:
                rows.append(
                    {
                        "dataset": r["dataset"],
                        "sweep": r["sweep"],
                        "algo": r["algo"],
                        "p": json.dumps(r["params"], sort_keys=True),
                        "perf_bs": e["bs"],
                        "perf_k": e["k"],
                        "new_ms": e["median_ms"],
                        "new_recall": r["quality"]["oracle"][f"recall@{e['k']}"],
                        "new_unstable": bool(e["unstable"]),
                    }
                )
new = pl.DataFrame(rows).with_columns(pl.col("perf_bs").cast(pl.Int64), pl.col("perf_k").cast(pl.Int64))
t = (
    new.join(old.with_columns(pl.col("perf_bs").cast(pl.Int64), pl.col("perf_k").cast(pl.Int64)), on=["dataset", "sweep", "algo", "p", "perf_bs", "perf_k"], how="left")
    .with_columns(ms_new_over_old=pl.col("new_ms") / pl.col("old_ms"), recall_delta=pl.col("new_recall") - pl.col("old_recall"))
    .sort("dataset", "sweep", "algo", "p", "perf_bs", "perf_k")
)
t.write_csv(sys.argv[3])
print(
    t.group_by("dataset", "algo")
    .agg(
        n=pl.len(),
        matched=pl.col("old_ms").is_not_null().sum(),
        ms_ratio_median=pl.col("ms_new_over_old").median(),
        ms_ratio_min=pl.col("ms_new_over_old").min(),
        ms_ratio_max=pl.col("ms_new_over_old").max(),
        drecall_min=pl.col("recall_delta").min(),
        drecall_max=pl.col("recall_delta").max(),
    )
    .sort("dataset", "algo")
)
