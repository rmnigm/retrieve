"""Compile every padded kernel at power-of-two widths into TRITON_CACHE_DIR and print one
sha256 per PTX file (kernel name + hash). Run on the pre-L4 tree and on L4 with two fresh cache
dirs; identical output means the power-of-two path compiles to the same machine code."""

import hashlib
import pathlib
import os
import sys

import torch

from retrieve.ops import triton as T

N, B = 512, 2


def _i64(*shape):
    return torch.randint(0, 8, shape, device="cuda")


probe = torch.tensor([[0, 1]] * B, device="cuda")
offs = torch.tensor([0, N // 2, N], device="cuda")
perm = torch.arange(N, device="cuda")
pos = torch.arange(N, device="cuda").repeat(B, 1)
counts = torch.full((B,), N, device="cuda")
for d in (64, 128, 256):
    q, embs = (
        torch.randn(B, d, device="cuda").half(),
        torch.randn(N, d, device="cuda").half(),
    )
    qf, codes = (
        torch.randn(B, d, device="cuda"),
        torch.zeros(N, d, dtype=torch.int8, device="cuda"),
    )
    rev, qa, attrs = (
        torch.zeros(2, dtype=torch.bool, device="cuda"),
        _i64(B, 2),
        _i64(N, 2, 2),
    )
    T.fused_masked_knn_topk(q, embs, pos, counts, 4)
    T.codesigned_probe_score(qf, probe, offs, codes, perm, 0.1, 4, N)
    T.codesigned_probe_score_bloom(
        qf, probe, offs, codes, perm, _i64(B, 10), _i64(512, N // 64), 0.1, 4, N
    )
    T.codesigned_probe_score_exact(
        qf, probe, offs, codes, perm, attrs, rev, qa, 0.1, 4, N
    )
for w in (1, 2, 4, 16):
    qb, sigs = _i64(B, w), _i64(N, w)
    T.oporp_1bit_match_topk_full(qb, sigs, 4)
    T.oporp_1bit_match_topk_indirect(qb, sigs, 4, pos, counts)
    T.bloom_match(qb, sigs)
    T.bloom_compact(qb, sigs)
torch.cuda.synchronize()
rows = []
for p in pathlib.Path(os.environ["TRITON_CACHE_DIR"]).rglob("*.ptx"):
    rows.append(f"{p.stem} {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")
print("\n".join(sorted(rows)))
print(len(rows), "ptx files", file=sys.stderr)
