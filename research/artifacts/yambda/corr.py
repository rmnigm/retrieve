"""Does recall against the exact top-k track end-task recall on Yambda-500m (unfiltered, A5-yambda records)?

Per configuration (dim, algo, params) the records carry ``oracle_recall@k`` (share of the exact top-k returned) and
``heldout_recall@k`` (share of the user's held-out Listen+ events in the returned top-k). Writes ``yambda_recall.csv``
(one row per configuration and k, with held-out recall over the exact scan's at the same dim) and
``yambda_recall.json`` (Pearson / Spearman per dim and k, and pooled over dims on the ratio to the exact scan).

usage: corr.py RESULTS_PARQUET OUT
"""

import json
import sys
from pathlib import Path

import polars as pl
from scipy.stats import pearsonr, spearmanr

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
EXACT = "linr_v1_filter_mask"

df = pl.read_parquet(SRC).filter(
    (pl.col("dataset") == "yambda-500m") & (pl.col("status") == "ok")
)
df = df.with_columns(config=pl.col("algo") + " " + pl.col("params")).unique(
    ["dim", "config"], keep="last", maintain_order=True
)
rows, stats = [], {}
for k in [100, 1000]:
    x = df.drop_nulls([f"oracle_recall@{k}", f"heldout_recall@{k}"]).select(
        "dim",
        "algo",
        "config",
        pl.lit(k).alias("k"),
        pl.col(f"oracle_recall@{k}").alias("oracle"),
        pl.col(f"heldout_recall@{k}").alias("heldout"),
    )
    exact = dict(x.filter(pl.col("algo") == EXACT).select("dim", "heldout").iter_rows())
    x = x.with_columns(
        heldout_over_exact=pl.col("heldout") / pl.col("dim").replace_strict(exact)
    )
    rows.append(x)
    approx = x.filter(pl.col("algo") != EXACT)
    for (d,), g in approx.group_by(["dim"], maintain_order=True):
        stats[f"k{k} d{d}"] = dict(
            n=g.height,
            pearson=pearsonr(g["oracle"], g["heldout"])[0],
            spearman=spearmanr(g["oracle"], g["heldout"])[0],
            exact_heldout=exact[d],
        )
    stats[f"k{k} pooled"] = dict(
        n=approx.height,
        pearson=pearsonr(approx["oracle"], approx["heldout_over_exact"])[0],
        spearman=spearmanr(approx["oracle"], approx["heldout_over_exact"])[0],
    )
    for lo, hi in [(0.9, 1.01), (0.0, 0.7)]:
        b = approx.filter(pl.col("oracle").is_between(lo, hi, closed="left"))[
            "heldout_over_exact"
        ]
        if b.len():
            stats[f"k{k} oracle in [{lo}, {hi})"] = dict(
                n=b.len(), min=b.min(), median=b.median(), max=b.max()
            )
OUT.mkdir(parents=True, exist_ok=True)
pl.concat(rows).sort("k", "dim", "config").write_csv(OUT / "yambda_recall.csv")
(OUT / "yambda_recall.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
