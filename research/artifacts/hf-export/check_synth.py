"""Records' pass_rate on uniform synthetic sweeps against the regenerated synth_filter*.json."""
import json
from pathlib import Path

import polars as pl

A = Path("/data/anon-export")
DIRS = {"goodreads": "eval-goodreads-work-id", "arxiv": "eval-arxiv-papers", "pubmed": "eval-pubmed"}
NAMES = ["p0001", "p0003", "p001", "p003", "p005", "p01", "p02", "p03", "p05", "p1"]
df = pl.read_parquet("/data/work/all_records.parquet").filter(
    pl.col("dataset").str.contains("-synth") & ~pl.col("dataset").str.contains("corr|laion")
    & pl.col("pass_rate").is_not_null()
)
bad = n = 0
for r in df.select("dataset", "sweep", "pass_rate", "n_items").unique().iter_rows(named=True):
    base, _, draw = r["dataset"].partition("-synth")
    draw = int(draw[2:]) if draw else 0
    side = A / DIRS[base] / ("synth_filter.json" if draw == 0 else f"synth_filter_s{draw}.json")
    meta = json.loads(side.read_text())
    want = meta["pass_counts"][NAMES.index(r["sweep"])] / meta["n_items"]
    n += 1
    got = round(r["pass_rate"] * r["n_items"])
    if got != meta["pass_counts"][NAMES.index(r["sweep"])] or abs(want - r["pass_rate"]) > 1e-12 or meta["n_items"] != r["n_items"]:
        bad += 1
        print("MISMATCH", r, want)
print(f"{n} (dataset, sweep) pass counts compared (count = round(pass_rate * n_items), rate within 1e-12), {bad} mismatches")
