"""At-risk head-to-head settings and the SilverTorch grid each group reruns.

usage: python select.py H2H_CSV RECORDS_PARQUET OUT_DIR
H2H_CSV = results/tables/head_to_head_probe_budgets.csv, RECORDS_PARQUET = results/records/records.parquet
(both on origin/export). Writes at_risk.csv (one row per setting) and plan.json (one entry per group).
"""

import csv
import json
import sys
from pathlib import Path

import polars as pl

h2h, recs, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
rows = [
    x
    for x in csv.DictReader(open(h2h))
    if x["dataset"] != "yambda"
    and (x["silvertorch_over_linr"] == "" or float(x["silvertorch_over_linr"]) > 0.7)
]
with open(out / "at_risk.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

r = pl.read_parquet(recs).filter(
    (pl.col("experiment") == "head_to_head")
    & (pl.col("perf_mode") == "graph")
    & (pl.col("seed") == 0)
    & (pl.col("algo") == "silvertorch")
    & (pl.col("filter_kind") == "bloom")
)
p = r.with_columns(
    n_lists=pl.col("params").map_elements(lambda s: json.loads(s)["n_lists"], return_dtype=pl.Int64),
    n_probe=pl.col("params").map_elements(lambda s: json.loads(s)["n_probe"], return_dtype=pl.Int64),
)
groups = {}
for x in rows:
    g = (x["dataset"], int(x["dim"]), x["filter"], int(x["batch"]), int(x["k"]))
    e = groups.setdefault(g, {"dataset": g[0], "dim": g[1], "sweep": g[2], "bs": g[3], "k": g[4], "linr": [], "targets": []})
    e["targets"].append(float(x["recall_target"]))
    li = {"variant": x["linr_variant"], "pool": float(x["linr_v3_pool"]) if x["linr_v3_pool"] else None}
    if li not in e["linr"]:
        e["linr"].append(li)
for g, e in groups.items():
    pts = p.filter(
        (pl.col("dataset") == g[0]) & (pl.col("sweep") == g[2]) & (pl.col("perf_bs") == g[3]) & (pl.col("perf_k") == g[4])
    )
    e["points"] = sorted({(a, b) for a, b in zip(pts["n_lists"], pts["n_probe"])})
    e["pass_rate"] = pts["pass_rate"][0]
json.dump(list(groups.values()), open(out / "plan.json", "w"), indent=1)
print(len(rows), "settings,", len(groups), "groups")
