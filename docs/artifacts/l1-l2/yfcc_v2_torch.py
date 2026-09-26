"""L1: LiNR V2 (`backend="torch"`) against the harness's own YFCC-10M `tags_and` oracle blob.

The harness cannot run V2 at d192 (the Triton kernel needs a power-of-two D and the suite folds
V2's torch backend into Triton), so this scores the torch backend directly. Mean over kept rows of
|module ∩ oracle| / |oracle|, the harness's `recall_oracle@1000` (`bench.metrics`).
Run: uv run --no-sync python docs/artifacts/l1-l2/yfcc_v2_torch.py BLOB
"""

import sys

import polars as pl
import torch
import torch.nn.functional as F

from retrieve.modules import ExactAttributeFilter
from retrieve.modules.linr import LiNRV2

DATA = "/data/yfcc10m"
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False

blob = torch.load(sys.argv[1], weights_only=True)
oracle = blob["topk"]
emb = F.normalize(torch.load(f"{DATA}/content_d192/text_emb.pt", map_location=dev).float(), dim=-1)
queries = F.normalize(torch.load(f"{DATA}/content_d192/query_emb.pt").float(), dim=-1)
qa = torch.tensor(pl.read_parquet(f"{DATA}/eval_split.parquet")["query_attrs_narrow"].to_list())
attrs = torch.load(f"{DATA}/item_attrs_narrow.pt", map_location=dev, weights_only=True)
rev = torch.load(f"{DATA}/clause_is_reverse_narrow.pt", map_location=dev, weights_only=True)

m = LiNRV2(k=1000, filter=ExactAttributeFilter(backend="torch"), backend="torch")
m.register_index(emb, item_clause_attrs=attrs, clause_is_reverse=rev)
del emb
keep = ((qa[: oracle.shape[0]] != -1).any(1) & (oracle >= 0).any(1)).nonzero().reshape(-1)
recalls = []
for s in range(0, keep.numel(), 16):
    rows = keep[s : s + 16]
    ids, _ = m(queries[rows].to(dev), qa[rows].to(dev))
    for got, ref in zip(ids.cpu().tolist(), oracle[rows].tolist(), strict=True):
        ref = {v for v in ref if v >= 0}
        recalls.append(len(ref & set(got)) / len(ref))
print(f"linr_v2/torch recall_oracle@1000 = {sum(recalls) / len(recalls):.4f} over {len(recalls)} rows")
