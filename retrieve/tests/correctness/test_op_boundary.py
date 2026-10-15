"""The Triton ops reject a non-contiguous item-side table at the boundary rather than copying
it on every forward with a per-call ``.contiguous()`` (kernels.md, conventions)."""

from __future__ import annotations

import pytest
import torch

from retrieve.ops import triton as T

N, B = 512, 2


def _i64(*shape):
    return torch.randint(0, 8, shape, device="cuda")


def _strided(t):
    """``t`` as a non-contiguous view of the same shape (every other column of a wider tensor)."""
    wide = torch.stack([t, t], dim=-1).flatten(-2)
    return wide[..., ::2]


def _cases():
    attrs, rev, qa = _i64(N, 2, 2), torch.zeros(2, dtype=torch.bool, device="cuda"), _i64(B, 2)
    sigs, qb = _i64(N, 4), _i64(B, 4)
    embs, q = torch.randn(N, 64, device="cuda").half(), torch.randn(B, 64, device="cuda").half()
    pos, counts = torch.arange(N, device="cuda").repeat(B, 1), torch.full((B,), N, device="cuda")
    return {
        "clause_mask": lambda a=attrs: T.clause_mask(a, rev, qa),
        "clause_compact": lambda a=attrs: T.clause_compact(a, rev, qa),
        "fmkt_items": lambda e=embs: T.fused_masked_knn_topk(q, e, pos, counts, 4),
        "fmkt_candidates": lambda p=pos: T.fused_masked_knn_topk(q, embs, p, counts, 4),
        "oporp_full": lambda s=sigs: T.oporp_1bit_match_topk_full(qb, s, 4),
        "oporp_candidates": lambda p=pos: T.oporp_1bit_match_topk_indirect(qb, sigs, 4, p, counts),
    }, {"attrs": attrs, "sigs": sigs, "embs": embs, "pos": pos}


_TABLE = {
    "clause_mask": "attrs",
    "clause_compact": "attrs",
    "fmkt_items": "embs",
    "fmkt_candidates": "pos",
    "oporp_full": "sigs",
    "oporp_candidates": "pos",
}


@pytest.mark.parametrize("case", list(_TABLE))
def test_non_contiguous_item_table_rejected(case):
    calls, tables = _cases()
    calls[case]()  # the contiguous table is accepted
    with pytest.raises(ValueError, match="must be contiguous"):
        calls[case](_strided(tables[_TABLE[case]]))
