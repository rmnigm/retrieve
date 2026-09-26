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


def _layout():
    """Two clusters of ``N / 2`` items, both probed by every row: ``(probe_ids,
    cluster_offsets, sort_perm)``; the width is ``N``."""
    probe = torch.tensor([[0, 1]] * B, device="cuda")
    return probe, torch.tensor([0, N // 2, N], device="cuda"), torch.arange(N, device="cuda")


def _cases():
    attrs, rev, qa = _i64(N, 2, 2), torch.zeros(2, dtype=torch.bool, device="cuda"), _i64(B, 2)
    sigs, qb = _i64(N, 4), _i64(B, 4)
    qpos, bt = _i64(B, 10), _i64(256, N // 64)
    embs, q = torch.randn(N, 64, device="cuda").half(), torch.randn(B, 64, device="cuda").half()
    pos, counts = torch.arange(N, device="cuda").repeat(B, 1), torch.full((B,), N, device="cuda")
    codes, qf = (
        torch.zeros(N, 64, dtype=torch.int8, device="cuda"),
        torch.randn(B, 64, device="cuda"),
    )
    lay = _layout()
    return {
        "clause_mask": lambda a=attrs: T.clause_mask(a, rev, qa),
        "clause_compact": lambda a=attrs: T.clause_compact(a, rev, qa),
        "bloom_match": lambda s=sigs: T.bloom_match(qb, s),
        "bloom_compact": lambda s=sigs: T.bloom_compact(qb, s),
        "fmkt_items": lambda e=embs: T.fused_masked_knn_topk(q, e, pos, counts, 4),
        "fmkt_candidates": lambda p=pos: T.fused_masked_knn_topk(q, embs, p, counts, 4),
        "oporp_full": lambda s=sigs: T.oporp_1bit_match_topk_full(qb, s, 4),
        "oporp_candidates": lambda p=pos: T.oporp_1bit_match_topk_indirect(qb, sigs, 4, p, counts),
        "cps_codes": lambda c=codes: T.codesigned_probe_score(qf, *lay[:2], c, lay[2], 0.1, 4, N),
        "cps_bloom_transposed": lambda t=bt: T.codesigned_probe_score_bloom(
            qf, *lay[:2], codes, lay[2], qpos, t, 0.1, 4, N
        ),
        "cpse_attrs": lambda a=attrs: T.codesigned_probe_score_exact(
            qf, *lay[:2], codes, lay[2], a, rev, qa, 0.1, 4, N
        ),
    }, {
        "attrs": attrs,
        "sigs": sigs,
        "embs": embs,
        "pos": pos,
        "codes": codes,
        "bt": bt,
    }


_TABLE = {
    "clause_mask": "attrs",
    "clause_compact": "attrs",
    "bloom_match": "sigs",
    "bloom_compact": "sigs",
    "fmkt_items": "embs",
    "fmkt_candidates": "pos",
    "oporp_full": "sigs",
    "oporp_candidates": "pos",
    "cps_codes": "codes",
    "cps_bloom_transposed": "bt",
    "cpse_attrs": "attrs",
}


@pytest.mark.parametrize("case", list(_TABLE))
def test_non_contiguous_item_table_rejected(case):
    calls, tables = _cases()
    calls[case]()  # the contiguous table is accepted
    with pytest.raises(ValueError, match="must be contiguous"):
        calls[case](_strided(tables[_TABLE[case]]))
