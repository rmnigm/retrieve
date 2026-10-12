"""Bounded diagnosis of R-k64 (KuaiRand-27K d64): protocol structure, metric ceilings, a most-popular
baseline and k64 on train-seen targets only, all through training.evaluate.evaluate.

    cd evaluation && uv run python ../docs/artifacts/seqrec-encoder/k64-diagnosis/diag.py OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

from training.encode import load_model_for_eval
from training.evaluate import evaluate

DATA = Path("/data/kuairand")
CKPT = Path("/scratch/ckpt/k64-sasrec-ssm-logq/best_model.pt")
TMP = Path("/scratch/k64-diagnosis")
N = 32_038_725
out: dict = {}

counts = np.bincount(
    pl.read_parquet(DATA / "train.parquet", columns=["item_ids"])["item_ids"].explode().to_numpy(),
    minlength=N + 1,
)
seen = counts > 0
out["train_distinct_items"] = int(seen.sum())

q = [0.1, 0.25, 0.5, 0.75, 0.9]
for split in ("val", "test"):
    df = pl.read_parquet(DATA / f"{split}.parquet")
    nt = df["targets"].list.len().to_numpy()
    hl = df["item_ids"].list.len().to_numpy()
    last_ts = df["timestamps"].list.last().to_numpy()
    tgt = df["targets"].explode().to_numpy()
    in_hist = (
        df.select(pl.col("targets").list.set_intersection(pl.col("item_ids")).list.len())
        .to_series().to_numpy()
    )
    top100 = np.argsort(-counts)[:100]
    out[split] = {
        "rows": len(df),
        "targets_per_row_quantiles": dict(zip(map(str, q), np.quantile(nt, q).tolist())),
        "targets_mean": float(nt.mean()),
        "history_len_quantiles": dict(zip(map(str, q), np.quantile(hl, q).tolist())),
        "last_history_ts_minmax": [int(last_ts.min()), int(last_ts.max())],
        "recall@100_ceiling": float(np.mean(np.minimum(100, nt) / nt)),
        "recall@10_ceiling": float(np.mean(np.minimum(10, nt) / nt)),
        "frac_targets_seen_in_train": float(seen[tgt].mean()),
        "frac_targets_in_own_history": float(in_hist.sum() / nt.sum()),
        "frac_targets_in_train_top100": float(np.isin(tgt, top100).mean()),
    }
    TMP.mkdir(exist_ok=True)
    df_seen = df.with_columns(
        pl.col("targets").map_elements(lambda t: [x for x in t if seen[x]], return_dtype=pl.List(pl.Int64))
    ).filter(pl.col("targets").list.len() > 0)
    df_seen.write_parquet(TMP / f"{split}_seen.parquet")
    out[split]["rows_with_a_seen_target"] = len(df_seen)


class Popularity(torch.nn.Module):
    use_time = False

    def __init__(self, counts: np.ndarray):
        super().__init__()
        self.table = torch.from_numpy(counts.astype(np.float32))[:, None]

    def scoring_table(self) -> torch.Tensor:
        return self.table

    def predict_last(self, items, timestamps=None):
        return torch.ones(items.shape[0], 1, device=items.device)


device = torch.device("cuda")
kw = dict(num_items=N, max_length=200, batch_size=1024, ks=(10, 100), device=device)
pop = Popularity(counts).to(device)
pop.table = pop.table.to(device)
model = load_model_for_eval(CKPT, N, device)
for split in ("val", "test"):
    for name, m in (("popularity", pop), ("k64", model)):
        out[split][f"{name}_all_targets"] = evaluate(m, str(DATA / f"{split}.parquet"), **kw)
        out[split][f"{name}_seen_targets_only"] = evaluate(m, str(TMP / f"{split}_seen.parquet"), **kw)
        print(split, name, json.dumps({k: round(v, 4) for k, v in out[split][f"{name}_all_targets"].items()}), flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
