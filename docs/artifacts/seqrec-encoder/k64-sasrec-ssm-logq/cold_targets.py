"""Fraction of KuaiRand-27K val/test targets that never occur in train.parquet's item_ids (so their
item rows only ever get negative-sampling updates, no positive pull).

    cd evaluation && uv run python ../docs/artifacts/seqrec-encoder/k64-sasrec-ssm-logq/cold_targets.py
"""

import numpy as np
import polars as pl

tr = pl.scan_parquet("/data/kuairand/train.parquet").select(pl.col("item_ids").explode()).unique().collect()
seen = np.zeros(32038726, dtype=bool)
seen[tr["item_ids"].to_numpy()] = True
print("train distinct items (inputs)", int(seen.sum()))
for s in ("val", "test"):
    df = pl.read_parquet(f"/data/kuairand/{s}.parquet", columns=["targets"])
    t = df["targets"].explode().to_numpy()
    print(s, "targets", len(t), "seen-in-train frac", round(float(seen[t].mean()), 4),
          "per-user median", int(df["targets"].list.len().median()))
