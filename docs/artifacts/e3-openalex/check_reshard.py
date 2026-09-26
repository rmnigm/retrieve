"""`openalex reshard` check: every item vector of the smaller catalog is `torch.equal` to its
source row in the larger one, matched through a join on work_id (not reshard's searchsorted).

    cd evaluation && PYTHONPATH=. python ../docs/artifacts/e3-openalex/check_reshard.py \
        /data/openalex /data/openalex-15m
"""

import json
import sys
from pathlib import Path

import polars as pl
import torch

from eval_datasets.layout import load_sharded

small, big = Path(sys.argv[1]), Path(sys.argv[2])
cpu = torch.device("cpu")
a = pl.read_parquet(small / "papers.parquet", columns=["item_id", "work_id"])
b = pl.read_parquet(big / "papers.parquet", columns=["item_id", "work_id"])
pairs = a.join(b, on="work_id", how="left", suffix="_big").sort("item_id")
missing = pairs["item_id_big"].null_count()
got = load_sharded(small / "content_d768" / "shard_index.json", cpu)
src = load_sharded(big / "content_d768" / "shard_index.json", cpu)
equal = missing == 0 and torch.equal(got, src[torch.from_numpy(pairs["item_id_big"].to_numpy() - 1)])
print(json.dumps({"n_small": a.height, "n_big": b.height, "missing_in_big": missing,
                  "all_rows_equal": equal}))  # fmt: skip
sys.exit(0 if equal else 1)
