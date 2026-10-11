"""Shared-K=256 gBCE collapse: user queries and output rows of the shared-negative checkpoint
vs the published d64 checkpoint (per-position negatives), on the first 2048 test users."""

import json
import sys
from pathlib import Path

import polars as pl
import torch

from training.encode import load_model_for_eval
from training.evaluate import EvalDataset, collate_eval

N = 1866170
dev = torch.device("cuda")
train = pl.read_parquet("/data/yambda-500m/trainer/train.parquet")["item_ids"].explode()
freq = torch.bincount(torch.from_numpy(train.to_numpy()), minlength=N + 1).float()
ds = EvalDataset("/data/yambda-500m/test.parquet", max_length=200)
items, _, _, _ = collate_eval([ds[i] for i in range(2048)])
out = {}
for name, ckpt in [("published_per_position", "/data/yambda-500m/checkpoints/gsasrec-d64-drop0.5"),
                   ("shared_k256", sys.argv[1])]:
    m = load_model_for_eval(Path(ckpt) / "best_model.pt", N, dev)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        q = m.predict_last(items.to(dev)).float()
    table = m.scoring_table().detach().float()
    qn = torch.nn.functional.normalize(q, dim=-1)
    cos = qn @ qn.T
    top10 = (q @ table[1:].T).topk(10).indices
    norms = table[1:].norm(dim=-1).cpu()
    popular = freq[1:] >= freq[1:].quantile(0.999)
    unseen = freq[1:] == 0
    out[name] = {
        "mean_pairwise_query_cosine": ((cos.sum() - 2048) / (2048 * 2047)).item(),
        "distinct_items_in_2048_top10s": int(top10.unique().numel()),
        "top10_share_of_popular_0.1pct": popular[top10.cpu()].float().mean().item(),
        "row_norm_top_0.1pct_freq": norms[popular].mean().item(),
        "row_norm_never_in_train": norms[unseen].mean().item() if unseen.any() else None,
        "row_norm_all": norms.mean().item(),
    }
    print(name, json.dumps(out[name]), flush=True)
    del m
    torch.cuda.empty_cache()
Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + "\n")
