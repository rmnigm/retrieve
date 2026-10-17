"""Run b: LiNR's seed spread next to SilverTorch's, and the head-to-head ratios inside the combined band.

Spread of a timed cell = (max - min) / median of its graph medians over seeds, as validation's SilverTorch seed
spread; pooled per dataset as the mean over cells. LiNR "new": seeds 1 and 2 timed here (one box); "+s0": with the
recorded seed 0 added (another box). Recorded: the final run's head-to-head seeds 0-2 of both arms (seeds ran on
different pods). A head-to-head ratio r = SilverTorch / best LiNR is "inside the band" when
|r - 1| <= (s_ST + s_LiNR) / 2, the half ranges of the two pooled spreads of its dataset.

usage: compare_b.py NEW_RESULTS_DIR RECORDS.parquet SELECTIONS.csv OUT_PREFIX"""

import json
import sys
from pathlib import Path

import polars as pl

ALGO = {"V1": "linr_v1_filter_mask", "V2": "linr_v2", "V3": "linr_v3"}
rows = []
for p in Path(sys.argv[1]).rglob("*.jsonl"):
    if p.name.endswith(".samples.jsonl"):
        continue
    for line in p.open():
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        for v in r["perf"] or []:
            if v["mode"] == "graph" and v["k"] == 100:
                rows.append({"dataset": r["dataset"], "sweep": r["sweep"], "algo": r["algo"],
                             "params": json.dumps(r["params"], sort_keys=True), "seed": r["seed"], "bs": v["bs"],
                             "ms": v["median_ms"], "unstable": v["unstable"], "sm_mhz": v["sm_mhz"]})  # fmt: skip
new = pl.DataFrame(rows)
rec = (
    pl.read_parquet(sys.argv[2])
    .filter(
        (pl.col("experiment") == "head_to_head")
        & (pl.col("perf_k") == 100)
        & (pl.col("perf_mode") == "graph")
        & (pl.col("filter_kind") != "none")
    )
    .select(
        "dataset",
        "sweep",
        "algo",
        "params",
        "seed",
        bs=pl.col("perf_bs").cast(pl.Int64),
        ms="perf_median_ms",
    )
)
rec = rec.with_columns(
    params=pl.col("params").map_elements(
        lambda s: json.dumps(json.loads(s), sort_keys=True), return_dtype=pl.String
    )
)
key = ["dataset", "sweep", "algo", "params", "bs"]


def spread(df: pl.DataFrame, label: str) -> pl.DataFrame:
    return df.group_by(key).agg(
        pl.len().alias(f"n_{label}"),
        ((pl.col("ms").max() - pl.col("ms").min()) / pl.col("ms").median()).alias(
            label
        ),
    )


linr_rec = rec.filter(pl.col("algo") != "silvertorch")
cells = (
    spread(new, "new_s12")
    .join(spread(pl.concat([new.select(rec.columns), linr_rec.filter(pl.col("seed") == 0)]), "new_s12_rec_s0"),
          on=key, how="left")
    .join(spread(linr_rec, "rec_s012"), on=key, how="left")
    .join(new.group_by(key).agg(pl.col("unstable").any()), on=key)
)  # fmt: skip
cells.sort(key).write_csv(sys.argv[4] + "linr_cells.csv")
st = spread(rec.filter(pl.col("algo") == "silvertorch"), "st_s012").filter(
    pl.col("n_st_s012") >= 2
)
pool = (
    cells.group_by("dataset").agg(pl.col("new_s12").mean().alias("linr_new_s12"), pl.col("new_s12").max().alias("linr_new_s12_max"),
                                  pl.col("new_s12_rec_s0").mean().alias("linr_with_rec_s0"),
                                  pl.col("rec_s012").mean().alias("linr_rec_s012"), pl.len().alias("linr_cells"))
    .join(st.group_by("dataset").agg(pl.col("st_s012").mean().alias("st_rec_s012"), pl.len().alias("st_cells")),
          on="dataset", how="left")
)  # fmt: skip
pool.sort("dataset").write_csv(sys.argv[4] + "spread.csv")

sel = pl.read_csv(sys.argv[3]).filter(
    (pl.col("filter_type") == "real")
    & (pl.col("k") == 100)
    & pl.col("silvertorch_reached_target")
)
band = (
    sel.join(pool, on="dataset", how="left")
    .with_columns(
        half_band=(pl.col("st_rec_s012") + pl.col("linr_new_s12")) / 2,
        half_band_rec=(pl.col("st_rec_s012") + pl.col("linr_rec_s012")) / 2,
    )
    .with_columns(
        inside=(pl.col("silvertorch_over_linr") - 1).abs() <= pl.col("half_band"),
        inside_rec=(pl.col("silvertorch_over_linr") - 1).abs()
        <= pl.col("half_band_rec"),
    )
)
band.write_csv(sys.argv[4] + "band.csv")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(250)
print(pool.sort("dataset"))
print(band.select("dataset", "filter", "batch", "recall_target", "linr_variant", "silvertorch_over_linr", "half_band",
                  "inside", "half_band_rec", "inside_rec").sort("silvertorch_over_linr", descending=True).head(15))  # fmt: skip
print(
    "inside the band:",
    band["inside"].sum(),
    "/",
    band.height,
    "(recorded LiNR spread:",
    band["inside_rec"].sum(),
    ")",
)
