"""L1: which rounding costs the exact algorithms their recall on YFCC-10M d192.

Scores the `tags_and` sweep's first N_Q kept queries four ways and compares each top-1000 with
(a) the harness oracle's method (fp32 `q @ E^T`, TF32 off) and (b) an fp64 top-1000:
  fp16/fp16  fp16 table, fp16 query, fp16 output (PostfilterKNN before L1)
  fp16/fp32  fp16 table, fp16 query, fp32 output (`torch.mm(..., out_dtype=torch.float32)`)
  fp16→fp32  the fp16 table and query upcast to fp32 before the GEMM (exact products, fp32 FMA
             accumulation: separates storage rounding from the tensor-core accumulator)
  fp32/fp32  fp32 table and query (TF32 off)
Run: uv run --no-sync python docs/artifacts/l1-l2/yfcc_precision.py [N_Q]
"""

import sys

import polars as pl
import torch
import torch.nn.functional as F

from retrieve.modules.filters import ExactAttributeFilter

torch.backends.cuda.matmul.allow_tf32 = False
N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
K = 1000
DATA = "/data/yfcc10m"
dev = torch.device("cuda")

emb = torch.load(f"{DATA}/content_d192/text_emb.pt", map_location=dev)
emb = F.normalize(emb.float(), dim=-1)
q_all = F.normalize(torch.load(f"{DATA}/content_d192/query_emb.pt").float(), dim=-1)
qa = torch.tensor(
    pl.read_parquet(f"{DATA}/eval_split.parquet")["query_attrs_narrow"].to_list()
)
attrs = torch.load(f"{DATA}/item_attrs_narrow.pt", map_location=dev, weights_only=True)
rev = torch.load(
    f"{DATA}/clause_is_reverse_narrow.pt", map_location=dev, weights_only=True
)
assert emb.shape[0] == attrs.shape[0], (emb.shape, attrs.shape)
filt = ExactAttributeFilter(backend="triton")
filt.register_index(attrs, clause_is_reverse=rev)

keep = (qa != -1).any(1).nonzero().reshape(-1)[:N_Q]
t32 = emb.t().contiguous()
t16 = emb.half().t().contiguous()
t16up = t16.float()
del emb
torch.cuda.empty_cache()


def topk_masked(s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    s = s.masked_fill(~mask, float("-inf"))
    v, i = torch.topk(s, K, dim=1)
    return torch.where(torch.isfinite(v), i, -1)


def recall(a: torch.Tensor, ref: torch.Tensor) -> tuple[int, int]:
    hit = tot = 0
    for x, r in zip(a.tolist(), ref.tolist(), strict=True):
        r = {v for v in r if v >= 0}
        hit += len(r & {v for v in x if v >= 0})
        tot += len(r)
    return hit, tot


acc = {
    n: {"vs_oracle": [0, 0], "vs_fp64": [0, 0]}
    for n in ("fp16/fp16", "fp16/fp32", "fp16→fp32", "fp32/fp32")
}
for s in range(0, keep.numel(), 16):
    rows = keep[s : s + 16]
    q = q_all[rows].to(dev)
    mask = filt.evaluate_mask(qa[rows].to(dev))
    oracle = topk_masked(q @ t32, mask)
    ref64 = topk_masked(q.double() @ t32.double(), mask)
    got = {
        "fp16/fp16": topk_masked(q.half() @ t16, mask),
        "fp16/fp32": topk_masked(
            torch.mm(q.half(), t16, out_dtype=torch.float32), mask
        ),
        "fp16→fp32": topk_masked(q.half().float() @ t16up, mask),
        "fp32/fp32": oracle,
    }
    for n, ids in got.items():
        for key, ref in (("vs_oracle", oracle), ("vs_fp64", ref64)):
            h, t = recall(ids, ref)
            acc[n][key][0] += h
            acc[n][key][1] += t
h, t = recall(oracle, ref64)
print(
    f"queries {keep.numel()}, k {K}; oracle (fp32) itself vs fp64 over the last batch: {h / t:.4f}"
)
for n, d in acc.items():
    print(n, {k: round(v[0] / v[1], 4) for k, v in d.items()})
