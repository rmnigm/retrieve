"""Temporal-drift check for R-k64: a most-popular baseline counted over the val day's targets
(2022-05-06, the day before the test day, never trained on), scored on test through evaluate; plus
the overlap of test targets with val-day targets and with train.

    cd evaluation && uv run python ../docs/artifacts/seqrec-encoder/k64-diagnosis/recency.py OUT.json
"""

import json
import sys

import numpy as np
import polars as pl
import torch

from training.evaluate import evaluate

DATA = "/data/kuairand"
N = 32_038_725

val_t = pl.read_parquet(f"{DATA}/val.parquet", columns=["targets"])["targets"].explode().to_numpy()
test_t = pl.read_parquet(f"{DATA}/test.parquet", columns=["targets"])["targets"].explode().to_numpy()
train_i = pl.read_parquet(f"{DATA}/train.parquet", columns=["item_ids"])["item_ids"].explode().to_numpy()
val_counts = np.bincount(val_t, minlength=N + 1)
in_val = val_counts > 0
in_train = np.bincount(train_i, minlength=N + 1) > 0
out = {
    "frac_test_targets_in_val_day_targets": float(in_val[test_t].mean()),
    "frac_test_targets_in_train": float(in_train[test_t].mean()),
    "frac_test_targets_in_val_day_but_not_train": float((in_val[test_t] & ~in_train[test_t]).mean()),
    "frac_test_targets_in_neither": float((~in_val[test_t] & ~in_train[test_t]).mean()),
}


class Popularity(torch.nn.Module):
    use_time = False

    def __init__(self, counts, device):
        super().__init__()
        self.table = torch.from_numpy(counts.astype(np.float32))[:, None].to(device)

    def scoring_table(self):
        return self.table

    def predict_last(self, items, timestamps=None):
        return torch.ones(items.shape[0], 1, device=items.device)


device = torch.device("cuda")
out["val_day_popularity_on_test"] = evaluate(
    Popularity(val_counts, device), f"{DATA}/test.parquet", num_items=N, max_length=200,
    batch_size=1024, ks=(10, 100), device=device,
)
print(json.dumps(out, indent=1))
open(sys.argv[1], "w").write(json.dumps(out, indent=2) + "\n")
