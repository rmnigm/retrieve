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

from retrieve.functional import compact_mask
from retrieve.ops import reference
from retrieve.ops.triton.clause_compact import clause_compact
from retrieve.ops.triton.fused_masked_knn_topk import (
    _P_BUCKETS,
    DEFAULT_CONFIG,
    FusedMaskedKnnTopkConfig,
    _bucket_p,
    _fused_masked_knn_topk_impl,
    fused_masked_knn_topk,
)
from tests.conftest import make_index, make_mask, make_query
from tests.parity.conftest import POISON, assert_topk_equal, assert_topk_matches, poison_empty


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
    # fp32 tl.sum vs bmm accumulate in different orders; measured drift <= 6e-8 at D <= 128.
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores, atol=1e-6, rtol=0.0)


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
    """``all_scores`` is allocated via ``torch.empty``, relying on the kernel to write every slot
    in ``[0, P)`` (real dot or ``-inf`` past ``counts[bid]``). The poisoned allocator proves it:
    an unwritten slot would surface ``POISON`` in top-K, and the hit count proves the poisoned
    buffer is the one the kernel wrote."""
    b, n, d, k = 8, 1024, 64, 48
    embs = make_index(n, d)
    query = make_query(b, d)
    # counts land in [35, 61]: rows on both sides of k, so the -inf tail reaches top-K.
    mask = make_mask(b, n, pass_rate=0.05)
    pos, counts = compact_mask(mask)
    ref_ids, ref_scores = reference.fused_masked_knn_topk(query, embs, pos, counts, k)
    assert bool((counts < k).any()) and bool((counts >= k).any())

    hits = poison_empty(monkeypatch, (b, pos.shape[1]))
    out_ids, out_scores = fused_masked_knn_topk(query, embs, pos, counts, k)

    assert hits, "the score buffer no longer comes from torch.empty — the poison never ran"
    assert not (out_scores == POISON).any(), "poison leaked into top-K: a slot went unwritten"
    # fp32 tl.sum vs bmm accumulate in different orders; measured drift <= 6e-8 at D <= 128.
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores, atol=1e-6, rtol=0.0)


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
    # Same inputs → identical output regardless of tile config: the D reduction is per lane.
    assert_topk_equal(ids_a, scores_a, ids_b, scores_b)


def test_compact_kernel_initialises_buffer_to_minus_one():
    """Regression: clause_compact / bloom_compact must not return
    uninitialised memory past `counts[bid]`. We feed a mask that passes
    far fewer than ``n`` items per row and assert the compact buffer's
    suffix is filled with -1.
    """

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


def test_row_alone_equals_row_in_batch_and_item_permutation():
    """Row-local reduction: each row run alone equals its row in the batch; permuting the item
    table (and remapping the candidates) permutes the ids and leaves every score unchanged. At
    ``k = P`` so no tie run is cut by the K boundary."""
    b, n, d = 4, 1024, 64
    embs = make_index(n, d)
    query = make_query(b, d)
    pos, counts = compact_mask(make_mask(b, n, pass_rate=0.2))
    k = pos.shape[1]
    ids, scores = fused_masked_knn_topk(query, embs, pos, counts, k)
    for bi in range(b):
        one = fused_masked_knn_topk(
            query[bi : bi + 1], embs, pos[bi : bi + 1], counts[bi : bi + 1], k
        )
        assert_topk_equal(*one, ids[bi : bi + 1], scores[bi : bi + 1])
    g = torch.Generator(device="cuda").manual_seed(9)
    perm = torch.randperm(n, generator=g, device="cuda")
    inv = torch.argsort(perm)
    p_ids, p_scores = fused_masked_knn_topk(query, embs[perm], inv[pos], counts, k)
    assert_topk_equal(torch.where(p_ids >= 0, perm[p_ids.clamp_min(0)], -1), p_scores, ids, scores)


@pytest.mark.parametrize(
    "p,regime",
    [
        (_P_BUCKETS[0] - 1, "below the first bucket edge"),
        (_P_BUCKETS[0], "on the first bucket edge"),
        (_P_BUCKETS[0] + 1, "one past the first bucket edge"),
        (8 * DEFAULT_CONFIG.block_n, "p % block_n == 0"),
        (8 * DEFAULT_CONFIG.block_n + 1, "p % block_n == 1"),
    ],
)
def test_across_bucket_and_tile_cutoffs(p, regime):
    """``_impl`` launches at ``_bucket_p(p)`` and the public op at ``p``, tiled by
    ``DEFAULT_CONFIG.block_n``: both sides of each cutoff, both entry points, against the
    shared oracle; the two entry points agree bit for bit."""
    if "bucket" in regime:
        assert _bucket_p(p) == (_P_BUCKETS[0] if p <= _P_BUCKETS[0] else _P_BUCKETS[1]), regime
    else:
        assert p % DEFAULT_CONFIG.block_n == int(regime[-1]), regime
    b, n, d, k = 4, 4096, 64, 16
    embs = make_index(n, d)
    query = make_query(b, d)
    g = torch.Generator(device="cuda").manual_seed(p)
    pos = torch.randint(0, n, (b, p), generator=g, device="cuda")
    counts = torch.tensor([p, p - 1, p // 2, 1], dtype=torch.long, device="cuda")
    ref = reference.fused_masked_knn_topk(query, embs, pos, counts, k)
    out = fused_masked_knn_topk(query, embs, pos, counts, k)
    # fp32 tl.sum vs bmm accumulate in different orders; measured drift <= 6e-8 at D <= 128.
    assert_topk_matches(*out, *ref, atol=1e-6, rtol=0.0)
    assert_topk_equal(*_fused_masked_knn_topk_impl(query, embs, pos, counts, k), *out)
