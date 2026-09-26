"""The Triton ops reject two input shapes at the boundary rather than deep in a call
(kernels.md, conventions): a non-contiguous item-side table, which a per-call
``.contiguous()`` would copy on every forward, and a non-power-of-two ``tl.arange`` extent
(``D`` or ``W``), which otherwise fails inside the Triton compiler."""

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


def _odd_dim_calls():
    d, w = 96, 3
    q, embs = torch.randn(B, d, device="cuda").half(), torch.randn(N, d, device="cuda").half()
    pos, counts = torch.arange(N, device="cuda").repeat(B, 1), torch.full((B,), N, device="cuda")
    qb, sigs = _i64(B, w), _i64(N, w)
    qf, codes = torch.randn(B, d, device="cuda"), torch.zeros(N, d, dtype=torch.int8, device="cuda")
    lay = _layout()
    rev, qa, attrs = torch.zeros(2, dtype=torch.bool, device="cuda"), _i64(B, 2), _i64(N, 2, 2)
    return {
        "fmkt D=96": lambda: T.fused_masked_knn_topk(q, embs, pos, counts, 4),
        "oporp W=3": lambda: T.oporp_1bit_match_topk_full(qb, sigs, 4),
        "bloom_match W=3": lambda: T.bloom_match(qb, sigs),
        "bloom_compact W=3": lambda: T.bloom_compact(qb, sigs),
        "cps D=96": lambda: T.codesigned_probe_score(qf, *lay[:2], codes, lay[2], 0.1, 4, N),
        "cps_bloom D=96": lambda: T.codesigned_probe_score_bloom(
            qf, *lay[:2], codes, lay[2], _i64(B, 10), _i64(512, N // 64), 0.1, 4, N
        ),
        "cpse D=96": lambda: T.codesigned_probe_score_exact(
            qf, *lay[:2], codes, lay[2], attrs, rev, qa, 0.1, 4, N
        ),
    }


_ODD_DIM_CASES = (
    "fmkt D=96",
    "oporp W=3",
    "bloom_match W=3",
    "bloom_compact W=3",
    "cps D=96",
    "cps_bloom D=96",
    "cpse D=96",
)


@pytest.mark.parametrize("case", _ODD_DIM_CASES)
def test_non_power_of_two_extent_rejected(case):
    with pytest.raises(ValueError, match="must be a power of two"):
        _odd_dim_calls()[case]()
