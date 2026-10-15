"""Does recall against the exact filtered top-k track end-task recall? goodreads d128, from the FULL-RERUN records.

Every distinct configuration on goodreads (SilverTorch probe-curve and timed points, bloom with the recheck off and clause, LiNR V1 / V2, LiNR V3 at candidate
pools 0.5-20 %) carries both metrics per record: ``oracle_recall@k`` (share of the exact filtered top-k returned) and
``heldout_recall@k`` (share of held-out interactions in the returned top-k, the end-task metric; rows whose target
passes the filter). Writes ``product_recall.csv`` (one row per configuration and k) and ``product_recall.json``
(Pearson / Spearman per filter and k, and pooled over filters with held-out recall divided by the exact scan's), for
SilverTorch's bloom points (the study arm) and for every configuration (robustness).

usage: product_recall.py FR OUT   (FR: one bench-fetched directory per fr/ subtree)
"""

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

HUB, OUT = Path(sys.argv[1]), Path(sys.argv[2])
LEGS = [
    "curve-d128-pode-s0",
    "timed-bloom-d128-pode-s0",
    "v3pool-pode-s0",
    "v3pool-pode-s0-rerun",
    "final-filter-pode-s0",
]
EXACT = ["linr_v1_filter_mask", "linr_v2"]

df = pd.concat([pd.read_parquet(HUB / leg / "results.parquet") for leg in LEGS])
df = df[
    (df.dataset == "goodreads") & (df.dim == 128) & (df.seed == 0) & (df.status == "ok")
]
df = df[
    df.filter_kind.isin(["clause", "bloom"])
    & df.backend.isin(["official-fork", "triton"])
]
df = df.assign(config=df.algo + " " + df.filter_kind + " " + df.params)
rows, stats = [], {}
for k in [100, 1000]:
    x = df.dropna(subset=[f"oracle_recall@{k}", f"heldout_recall@{k}"])
    x = x.drop_duplicates(["config", "sweep"])
    x = x.assign(k=k, oracle=x[f"oracle_recall@{k}"], heldout=x[f"heldout_recall@{k}"])
    exact = x[x.algo.isin(EXACT)].groupby("sweep").heldout.mean()
    x = x.assign(heldout_over_exact=x.heldout / x.sweep.map(exact))
    rows.append(
        x[["k", "sweep", "algo", "config", "oracle", "heldout", "heldout_over_exact"]]
    )
    arms = {  # the study arm (SilverTorch bloom) first; every configuration pooled as the robustness line
        "bloom": x[(x.algo == "silvertorch") & (x.filter_kind == "bloom")],
        "all": x,
    }
    for arm, y in arms.items():
        for sw, g in y.groupby("sweep"):
            if len(g) < 3:
                continue
            stats[f"{arm} k{k} {sw}"] = dict(
                n=len(g),
                pearson=pearsonr(g.oracle, g.heldout)[0],
                spearman=spearmanr(g.oracle, g.heldout)[0],
                exact_heldout=exact[sw],
            )
        stats[f"{arm} k{k} pooled"] = dict(
            n=len(y),
            pearson=pearsonr(y.oracle, y.heldout_over_exact)[0],
            spearman=spearmanr(y.oracle, y.heldout_over_exact)[0],
        )
OUT.mkdir(parents=True, exist_ok=True)
pd.concat(rows).to_csv(OUT / "product_recall.csv", index=False)
(OUT / "product_recall.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
