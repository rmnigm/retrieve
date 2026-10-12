"""Reproducer for the inductor ``assert_alignment`` failure on ``bloom_compact``'s ``qb`` input
seen by ``time_compaction.py`` (compiled ``LiNRV2(filter=BloomFilter)`` on goodreads shapes).
Runs the compiled forward at the given batch size and reports pass / fail."""

import sys

import torch
from retrieve.modules import BloomFilter, LiNRV2

DATA = "/workspace/data/goodreads-work-id"
b = int(sys.argv[1])
dev = torch.device("cuda")
attrs = torch.load(f"{DATA}/item_attrs_narrow.pt").to(dev)
n = attrs.shape[0]
g = torch.Generator(device=dev).manual_seed(0)
embs = torch.randn(n, 128, generator=g, device=dev)
embs = (embs / embs.norm(dim=1, keepdim=True)).half()
mod = LiNRV2(100, filter=BloomFilter(m_bits=1024, k_hash=5)).to(dev)
mod.register_index(embs, attrs)
q = torch.full((b, 4), -1, dtype=torch.int64, device=dev)
q[:, 0] = 3
qe = torch.randn(b, 128, generator=g, device=dev)
qe = (qe / qe.norm(dim=1, keepdim=True)).half()
eager = mod(qe, q)
compiled = torch.compile(mod, mode="reduce-overhead", dynamic=False, fullgraph=True)
try:
    for _ in range(5):
        out = compiled(qe, q)
    torch.cuda.synchronize()
    print(f"B={b}: ok, equal to eager: {torch.equal(out[1], eager[1])}")
except AssertionError as e:
    print(f"B={b}: FAIL {str(e).strip().splitlines()[-1]}")
