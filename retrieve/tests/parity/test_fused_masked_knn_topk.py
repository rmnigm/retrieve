"""Triton ``fused_masked_knn_topk`` vs gather + bmm + topk reference.

The reference impl mirrors ``PrefilterKNN._forward_prefilter``: gather the
passing rows into ``[B, P, D]``, score with bmm, top-K locally. Both
``PrefilterKNN`` and this test take ``(positive_indices, counts)`` directly —
the kernel never sees a bool mask.
The local helper here ``compact_mask``-s a random mask only to construct test
inputs.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.triton.linr.fused_masked_knn_topk import (
    FusedMaskedKnnTopkConfig,
    _bucket_p,
    _fused_masked_knn_topk_impl,
    fused_masked_knn_topk,
)
from retrieve.layers.utils.compact import compact_mask
from tests.conftest import make_index, make_mask, make_query
from tests.parity.conftest import assert_topk_matches


def _ref(query, item_embs, mask, k):
    counts = mask.sum(dim=1)
    p = int(counts.max().item())
    sorted_idx = mask.float().argsort(dim=1, descending=True, stable=True)
    reduced_ids = sorted_idx[:, :p]
    reduced_embs = item_embs[reduced_ids]
    scores = torch.bmm(query.unsqueeze(1), reduced_embs.transpose(1, 2)).squeeze(1)
    valid = torch.arange(p, device=query.device).unsqueeze(0) < counts.unsqueeze(1)
    scores = scores.masked_fill(~valid, float("-inf"))
    topk_scores, topk_local = torch.topk(scores, k, dim=1)
    topk_ids = reduced_ids.gather(1, topk_local)
    return topk_ids, topk_scores


@pytest.mark.parametrize(
    "b,n,d,k",
    [
        (1, 1024, 64, 16),
        (16, 16_384, 128, 200),
        # P_real << P_bucket case: pass_rate keeps P_real around ~400-800
        # while bucket=2048; exercises Phase 2's smaller score buffer + grid.
        (8, 8_192, 64, 16),
    ],
)
@pytest.mark.parametrize("pass_rate", [0.05, 0.5])
def test_matches_pure_torch(b, n, d, k, pass_rate):
    embs = make_index(n, d)
    query = make_query(b, d)
    mask = make_mask(b, n, pass_rate=pass_rate)

    pos, counts = compact_mask(mask)
    out_ids, out_scores = fused_masked_knn_topk(query, embs, pos, counts, k)
    ref_ids, ref_scores = _ref(query, embs, mask, k)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


def test_padding_uses_minus_one_when_counts_below_k():
    """Regression: when counts[b] < k, the padding slots must be -1.

    The original bug had two compounding faults:

    1. ``clause_compact`` / ``bloom_compact`` allocated their candidate
       buffer via ``torch.empty`` — uninitialised memory past
       ``counts[bid]``.
    2. After ``torch.topk`` over the ``[B, P]`` score matrix, the bottom
       slots tied at -inf and the gather pulled from those uninitialised
       positions, leaking random int64 values into top-K (e.g.
       ``-4785944168570074265``).

    On goodreads ``c0c4_author`` ~23% of users had counts[b] < 10, so
    ``linr_v2_filter_compact`` reported recall@10 = 0.77 against the
    oracle (which has the same flaw on the symmetric side: torch.topk
    with -inf scores tie-breaks to the lowest item ids 0, 1, 2, …, so
    its padding slots and v2's padding slots disagreed).

    Fix surface: both ``*_compact`` allocate with ``torch.full(..., -1)``
    AND the wrapper post-masks topk_ids to -1 on -inf scores. This test
    exercises the *wrapper* explicitly with a hand-built positive_indices
    buffer that contains plausible-looking-but-stale ids past
    ``counts[b]`` — the wrapper must not return them.
    """
    b, n, d, k = 4, 64, 32, 10
    p = 32  # padded width of the candidate buffer
    embs = make_index(n, d)
    query = make_query(b, d)

    # Build a positive_indices buffer where each row's prefix is real
    # in-range ids and the suffix is "stale" plausible ids. The kernel
    # must respect counts[b] and ignore the stale suffix.
    pos = torch.zeros((b, p), dtype=torch.int64, device=embs.device)
    counts = torch.tensor([3, 5, 0, k], dtype=torch.int64, device=embs.device)
    for bi in range(b):
        cnt = int(counts[bi].item())
        if cnt > 0:
            pos[bi, :cnt] = torch.arange(cnt, device=embs.device, dtype=torch.int64)
        # Stale-but-plausible suffix: real in-range ids that are NOT in
        # the candidate set. A buggy gather will return these.
        pos[bi, cnt:] = torch.arange(n - (p - cnt), n, device=embs.device, dtype=torch.int64)

    out_ids, out_scores = fused_masked_knn_topk(query, embs, pos, counts, k)

    for bi in range(b):
        cnt = int(counts[bi].item())
        # First `cnt` slots: real candidate ids in [0, cnt).
        for slot in range(min(cnt, k)):
            id_ = int(out_ids[bi, slot].item())
            assert 0 <= id_ < cnt, (
                f"row={bi} slot={slot}: expected real candidate in [0,{cnt}), got {id_}"
            )
            assert torch.isfinite(out_scores[bi, slot]), (
                f"row={bi} slot={slot}: real candidate has non-finite score"
            )
        # Remaining slots: must be -1 sentinel, NOT garbage from `pos[bi, cnt:]`.
        for slot in range(cnt, k):
            assert int(out_ids[bi, slot].item()) == -1, (
                f"row={bi} slot={slot}: padding leaked id "
                f"{int(out_ids[bi, slot].item())}, expected -1"
            )
            assert not torch.isfinite(out_scores[bi, slot]), (
                f"row={bi} slot={slot}: padding has finite score"
            )


def test_empty_score_buffer_does_not_leak(monkeypatch):
    """Phase 1 regression: ``all_scores`` is allocated via ``torch.empty``,
    relying on the kernel to write every slot in ``[0, P)`` (real dot or
    ``-inf`` past ``counts[bid]``). Poison every fresh ``torch.empty``
    float32 2-D buffer with ``+1e30`` before the kernel runs — if the
    kernel skipped any slot, top-K would surface that poison value
    (``+1e30`` beats every cosine in ``[-1, 1]``).
    """
    real_empty = torch.empty
    poison = 1e30

    def poisoned_empty(*args, **kwargs):
        t = real_empty(*args, **kwargs)
        if t.dtype == torch.float32 and t.dim() == 2:
            t.fill_(poison)
        return t

    monkeypatch.setattr(torch, "empty", poisoned_empty)

    b, n, d, k = 8, 1024, 64, 10
    embs = make_index(n, d)
    query = make_query(b, d)
    # Mix of pass rates per row so some rows have counts < k (stresses
    # the padding-region writes) and some have counts > k.
    mask = make_mask(b, n, pass_rate=0.05)
    pos, counts = compact_mask(mask)

    out_ids, out_scores = fused_masked_knn_topk(query, embs, pos, counts, k)

    finite = torch.isfinite(out_scores)
    assert finite.any(), "test setup degenerate — no finite scores"
    assert (out_scores[finite] != poison).all(), (
        "torch.empty's +1e30 poison leaked into top-K — kernel left a slot unwritten"
    )
    assert (out_scores[finite].abs() <= 1.5).all(), (
        f"finite scores out of cosine range: max={out_scores[finite].abs().max().item()}"
    )


def test_bucket_p_ladder():
    """``_bucket_p`` rounds runtime P up to a fixed ladder. The kernel's
    ``P: tl.constexpr`` is sized by this value, so two distinct runtime
    widths in the same bucket compile once (autotune's prior cache
    invariant — now a JIT-cache invariant)."""
    assert _bucket_p(1) == 256
    assert _bucket_p(256) == 256
    assert _bucket_p(257) == 2048
    assert _bucket_p(2000) == 2048
    assert _bucket_p(2048) == 2048
    assert _bucket_p(2049) == 16384
    assert _bucket_p(131072) == 131072
    # Past the last static bucket: next power of 2.
    assert _bucket_p(1_048_577) == 1 << 21


def test_config_override_matches_default():
    """Plumbing check: a deliberately-different ``config`` reaches the
    launch and produces the same ids / scores as the default. The
    ``cfg_a != cfg_b`` assertion makes the override observable to a
    reviewer; if the wrapper ignored ``config`` (e.g. forgot to thread
    it to the ``BLOCK_N`` kwarg) the cfg_a/cfg_b run would silently
    use the same tile.
    """
    b, n, d, k = 4, 1024, 64, 16
    embs = make_index(n, d)
    query = make_query(b, d)
    mask = make_mask(b, n, pass_rate=0.1)
    pos, counts = compact_mask(mask)

    cfg_a = FusedMaskedKnnTopkConfig(block_n=64, num_warps=4)
    cfg_b = FusedMaskedKnnTopkConfig(block_n=256, num_warps=8)
    assert cfg_a != cfg_b

    ids_a, scores_a = _fused_masked_knn_topk_impl(query, embs, pos, counts, k, config=cfg_a)
    ids_b, scores_b = _fused_masked_knn_topk_impl(query, embs, pos, counts, k, config=cfg_b)
    # Same inputs → identical output regardless of tile config.
    torch.testing.assert_close(ids_a, ids_b)
    torch.testing.assert_close(scores_a, scores_b)


def test_compact_kernel_initialises_buffer_to_minus_one():
    """Regression: clause_compact / bloom_compact must not return
    uninitialised memory past `counts[bid]`. We feed a mask that passes
    far fewer than ``n`` items per row and assert the compact buffer's
    suffix is filled with -1.
    """
    from retrieve.kernels.triton.filters.clause_compact import clause_compact

    b, n, c, a_max = 4, 1024, 2, 1
    # Items: each item has a fixed value per clause; we build sparse matches.
    item_attrs = torch.full((n, c, a_max), -1, dtype=torch.int64, device="cuda")
    # Make item i have clause-0 value (i % 8). Only items where i % 8 == q_c match.
    item_attrs[:, 0, 0] = torch.arange(n, device="cuda", dtype=torch.int64) % 8
    item_attrs[:, 1, 0] = 0  # clause 1 always matches when query asks for 0
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    # Each query: clause 0 = bid (only items i where i%8 == bid match), clause 1 inactive (-1).
    query_attrs = torch.full((b, c), -1, dtype=torch.int64, device="cuda")
    query_attrs[:, 0] = torch.arange(b, device="cuda", dtype=torch.int64)

    indices, counts = clause_compact(item_attrs, is_reverse, query_attrs)

    # Each row should have exactly n/8 = 128 passing items, leaving most slots empty.
    assert (counts == n // 8).all(), f"counts={counts.tolist()} expected {n // 8}"
    p = indices.shape[1]
    assert p >= n // 8

    # Slots beyond counts[b] must be -1, not random uninitialised memory.
    for bi in range(b):
        cnt = int(counts[bi].item())
        suffix = indices[bi, cnt:]
        assert (suffix == -1).all(), (
            f"row={bi}: clause_compact left non-(-1) values past counts={cnt}; "
            f"sample suffix={suffix[:5].tolist()}"
        )
