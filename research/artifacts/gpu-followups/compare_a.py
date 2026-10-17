"""Run a against the final run: per re-run point, recall@100 and the graph median (ms) at bs 1 / 16, recorded
(seed 0) vs now; per head-to-head workload and target, the SilverTorch point the final run's rule selects among
the re-run points (fastest reaching the target with n_probe <= n_lists / 4) over the recorded best LiNR.

usage: compare_a.py NEW_RESULTS_DIR RECORDS.parquet SELECTIONS.csv OUT_PREFIX"""

import json
import os
import sys
from pathlib import Path

import polars as pl

rows = []
for p in Path(sys.argv[1]).rglob("*.jsonl"):
    if p.name.endswith(".samples.jsonl"):
        continue
    for line in p.open():
        r = json.loads(line)
        if r.get("status") != "ok" or r["algo"] != "silvertorch":
            continue
        for v in r["perf"] or []:
            if v["mode"] == "graph":
                rows.append(
                    {
                        "dataset": r["dataset"],
                        "sweep": r["sweep"],
                        "n_lists": r["params"]["n_lists"],
                        "n_probe": r["params"]["n_probe"],
                        "bs": v["bs"],
                        "k": v["k"],
                        "new_recall": r["quality"]["oracle"]["recall@100"],
                        "new_ms": v["median_ms"],
                        "new_unstable": v["unstable"],
                        "new_sm_mhz": v["sm_mhz"],
                    }
                )
new = pl.DataFrame(rows).filter(pl.col("k") == 100)
old = (
    pl.read_parquet(sys.argv[2])
    .filter(
        (pl.col("experiment") == "head_to_head")
        & (pl.col("algo") == "silvertorch")
        & (pl.col("seed") == 0)
        & (pl.col("perf_k") == 100)
    )
    .with_columns(
        n_lists=pl.col("params").map_elements(
            lambda p: json.loads(p)["n_lists"], return_dtype=pl.Int64
        ),
        n_probe=pl.col("params").map_elements(
            lambda p: json.loads(p)["n_probe"], return_dtype=pl.Int64
        ),
        bs=pl.col("perf_bs").cast(pl.Int64),
    )
    .group_by("dataset", "sweep", "n_lists", "n_probe", "bs")
    .agg(
        old_recall=pl.col("oracle_recall@100").first(),
        old_ms=pl.col("perf_median_ms").min(),
        old_unstable=pl.col("unstable").any(),
    )
)
pts = (
    new.join(old, on=["dataset", "sweep", "n_lists", "n_probe", "bs"], how="left")
    .with_columns(
        d_recall=pl.col("new_recall") - pl.col("old_recall"),
        ms_ratio=pl.col("new_ms") / pl.col("old_ms"),
    )
    .sort("dataset", "sweep", "bs", "n_lists", "n_probe")
)
pts.write_csv(sys.argv[4] + "points.csv")

sel = pl.read_csv(sys.argv[3]).filter(
    (pl.col("k") == 100) & pl.col("batch").is_in([1, 16])
)
out = []
for r in sel.iter_rows(named=True):
    c = pts.filter(
        (pl.col("dataset") == r["dataset"])
        & (pl.col("sweep") == r["filter"])
        & (pl.col("bs") == r["batch"])
    )
    if c.is_empty():
        continue
    ok = c.filter(
        (pl.col("new_recall") >= r["recall_target"])
        & (pl.col("n_probe") <= pl.col("n_lists") // 4)
    )
    best = ok.sort("new_ms").row(0, named=True) if ok.height else None
    ratio = best["new_ms"] / r["linr_ms"] if best else None
    old_ratio = r["silvertorch_over_linr"]
    out.append(
        {
            "dataset": r["dataset"],
            "filter": r["filter"],
            "batch": r["batch"],
            "target": r["recall_target"],
            "old_point": f"{r['silvertorch_n_lists']}/{r['silvertorch_n_probe']}"
            if r["silvertorch_reached_target"]
            else "not reached",
            "old_recall": r["silvertorch_recall"],
            "old_ratio": old_ratio,
            "new_point": f"{best['n_lists']}/{best['n_probe']}"
            if best
            else "not reached",
            "new_recall": best["new_recall"] if best else c["new_recall"].max(),
            "new_ratio": ratio,
            "linr": r["linr_variant"],
            "linr_ms": r["linr_ms"],
            "flip": (old_ratio is None) != (ratio is None)
            or (ratio is not None and (old_ratio < 1) != (ratio < 1)),
        }
    )
v = pl.DataFrame(out).sort("dataset", "filter", "batch", "target")
# points dropped from this run (controller, 2026-10-17): the other pod's rerun measures them
covered = {
    (d, f, int(n))
    for d, f, n in (x.split("/") for x in os.environ.get("COVERED", "").split(",") if x)
}
v = v.with_columns(
    note=pl.struct("dataset", "filter", "old_point", "new_point").map_elements(
        lambda r: (
            "covered by the pod-h rerun"
            if r["new_point"] == "not reached"
            and r["old_point"] != "not reached"
            and (r["dataset"], r["filter"], int(r["old_point"].split("/")[0]))
            in covered
            else ""
        ),
        return_dtype=pl.String,
    )
).with_columns(flip=pl.col("flip") & (pl.col("note") == ""))
v.write_csv(sys.argv[4] + "verdicts.csv")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_width_chars(250)
print(
    pts.select(
        "dataset",
        "sweep",
        "bs",
        "n_lists",
        "n_probe",
        "old_recall",
        "new_recall",
        "d_recall",
        "old_ms",
        "new_ms",
        "ms_ratio",
        "new_unstable",
    )
)
print(v)
print("flips:", v["flip"].sum(), "of", v.height)
