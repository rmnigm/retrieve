"""Triton ``codesigned_probe_score_exact`` vs the same op in ``retrieve.ops.reference`` (the
pure-torch phase 2+3 with the exact predicate)."""

from __future__ import annotations

import pytest
import torch

from retrieve.indexing.quantize import quantize_int8_global
from retrieve.ops import reference
from retrieve.ops.triton.codesigned_probe_score_exact import (
    DEFAULT_CONFIG,
    CodesignedProbeScoreExactConfig,
    _codesigned_probe_score_exact_impl,
    codesigned_probe_score_exact,
)
from tests.conftest import (
    make_attrs,
    make_index,
    make_query,
    make_query_attrs,
)
from tests.parity.conftest import POISON, assert_topk_equal, make_exact, poison_empty


def _make_flat_probed(b, n, p, *, pad_rate=0.1, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    flat = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)
    pad = torch.rand(b, p, generator=g, device="cuda") < pad_rate
    flat[pad] = -1
    return flat


@pytest.mark.parametrize(
    "n,d,p,k,c,a_max",
    [
        (1024, 64, 128, 8, 1, 1),
        (2048, 64, 256, 16, 2, 2),
        (8192, 128, 512, 32, 3, 4),
    ],
)
@pytest.mark.parametrize("b", [1, 16])
def test_codesigned_exact_matches_ref(n, d, p, k, c, a_max, b):
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p)

    attrs = make_attrs(n, c=c, a_max=a_max).long()
    q_attrs = make_query_attrs(b, c=c).long()
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")

    out_ids, out_scores = codesigned_probe_score_exact(
        query,
        flat,
        codes,
        attrs,
        rev,
        q_attrs,
        global_scale,
        k,
    )
    ref_ids, ref_scores = reference.codesigned_probe_score_exact(
        query, flat, codes, attrs, rev, q_attrs, global_scale, k
    )
    assert_topk_equal(out_ids, out_scores, ref_ids, ref_scores)


def test_codesigned_exact_reverse_clause():
    """Setting clause_is_reverse[c]=True inverts the clause's match —
    items previously kept are now dropped (and vice versa) for that clause.
    """
    n, d, p, k, b, c, a_max = 2048, 64, 256, 16, 4, 2, 2
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p, pad_rate=0.0)

    attrs = make_attrs(n, c=c, a_max=a_max).long()
    q_attrs = make_query_attrs(b, c=c, inactive_rate=0.0).long()

    rev_off = torch.zeros(c, dtype=torch.bool, device="cuda")
    rev_on = torch.tensor([True, False], dtype=torch.bool, device="cuda")

    ids_off, scores_off = codesigned_probe_score_exact(
        query,
        flat,
        codes,
        attrs,
        rev_off,
        q_attrs,
        global_scale,
        k,
    )
    ids_on, scores_on = codesigned_probe_score_exact(
        query,
        flat,
        codes,
        attrs,
        rev_on,
        q_attrs,
        global_scale,
        k,
    )
    # Parity against the reference for both reverse configs.
    ref_off_ids, ref_off_scores = reference.codesigned_probe_score_exact(
        query, flat, codes, attrs, rev_off, q_attrs, global_scale, k
    )
    ref_on_ids, ref_on_scores = reference.codesigned_probe_score_exact(
        query, flat, codes, attrs, rev_on, q_attrs, global_scale, k
    )
    assert_topk_equal(ids_off, scores_off, ref_off_ids, ref_off_scores)
    assert_topk_equal(ids_on, scores_on, ref_on_ids, ref_on_scores)


def test_codesigned_exact_inactive_query():
    """``query_clause_attrs[:, c] = -1`` makes clause c always pass —
    matches the all-inactive baseline (no filter)."""
    n, d, p, k, b, c, a_max = 2048, 64, 256, 16, 4, 2, 2
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p, pad_rate=0.0)

    attrs = make_attrs(n, c=c, a_max=a_max).long()
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")

    # All clauses inactive → predicate is identically True; equivalent to
    # the no-filter codesigned_probe_score path on the same inputs.
    q_attrs = torch.full((b, c), -1, dtype=torch.long, device="cuda")
    out_ids, out_scores = codesigned_probe_score_exact(
        query,
        flat,
        codes,
        attrs,
        rev,
        q_attrs,
        global_scale,
        k,
    )

    from retrieve.ops.triton.codesigned_probe_score import (
        _codesigned_probe_score_impl,
    )

    ref_ids, ref_scores = _codesigned_probe_score_impl(query, flat, codes, global_scale, k)
    assert_topk_equal(out_ids, out_scores, ref_ids, ref_scores)


def test_config_override_matches_default():
    """Plumbing check: a deliberately-different ``config`` reaches the
    launch and produces identical ids / scores."""
    n, d, p, k, b, c, a_max = 1024, 64, 128, 8, 4, 2, 2
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p)

    attrs = make_attrs(n, c=c, a_max=a_max).long()
    q_attrs = make_query_attrs(b, c=c).long()
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")

    cfg_a = CodesignedProbeScoreExactConfig(block_p=32, num_warps=4)
    cfg_b = CodesignedProbeScoreExactConfig(block_p=128, num_warps=8)
    assert cfg_a != cfg_b

    ids_a, scores_a = _codesigned_probe_score_exact_impl(
        query,
        flat,
        codes,
        global_scale,
        k,
        item_clause_attrs=attrs,
        clause_is_reverse=rev,
        query_clause_attrs=q_attrs,
        config=cfg_a,
    )
    ids_b, scores_b = _codesigned_probe_score_exact_impl(
        query,
        flat,
        codes,
        global_scale,
        k,
        item_clause_attrs=attrs,
        clause_is_reverse=rev,
        query_clause_attrs=q_attrs,
        config=cfg_b,
    )
    assert_topk_equal(ids_a, scores_a, ids_b, scores_b)


def test_empty_score_buffer_does_not_leak(monkeypatch):
    """The ``[B, P]`` score buffer is ``torch.empty``; the kernel must write every slot (a dot
    or ``-inf`` for padding and predicate rejects). Poisoned allocator, hit count, exact
    parity."""
    n, d, p, k, b, c, a_max = 1024, 64, 300, 8, 4, 2, 2
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p, pad_rate=0.3)
    attrs = make_attrs(n, c=c, a_max=a_max)
    q_attrs = make_query_attrs(b, c=c)
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")
    args = (query, flat, codes, attrs, rev, q_attrs, global_scale, k)
    ref = reference.codesigned_probe_score_exact(*args)

    hits = poison_empty(monkeypatch, (b, p))
    out = codesigned_probe_score_exact(*args)

    assert hits, "the score buffer no longer comes from torch.empty — the poison never ran"
    assert not (out[1] == POISON).any(), "poison leaked into top-K: a slot went unwritten"
    assert_topk_equal(*out, *ref)


def _exact_args(b=4, p=300, n=1024, d=64, c=2, a_max=2):
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    attrs, rev, q_attrs = make_exact(n, b, c=c, a_max=a_max, reverse="mixed")
    return [make_query(b, d), _make_flat_probed(b, n, p), codes, attrs, rev, q_attrs, global_scale]


def test_row_alone_equals_row_in_batch():
    """Row-local reduction: each row run alone (its query, probe pool and clause attrs) equals
    its row in the batch."""
    query, flat, codes, attrs, rev, q_attrs, gs = _exact_args()
    k = 16
    ids, scores = codesigned_probe_score_exact(query, flat, codes, attrs, rev, q_attrs, gs, k)
    for bi in range(query.shape[0]):
        one = codesigned_probe_score_exact(
            query[bi : bi + 1], flat[bi : bi + 1], codes, attrs, rev, q_attrs[bi : bi + 1], gs, k
        )
        assert_topk_equal(*one, ids[bi : bi + 1], scores[bi : bi + 1])


@pytest.mark.parametrize("r", [0, 1])
def test_across_tile_cutoff(r):
    """``P % DEFAULT_CONFIG.block_p`` in {0, 1}: a full last tile and a one-lane last tile."""
    p = 2 * DEFAULT_CONFIG.block_p + r
    assert p % DEFAULT_CONFIG.block_p == r
    args = _exact_args(p=p)
    out = codesigned_probe_score_exact(*args, 32)
    assert_topk_equal(*out, *reference.codesigned_probe_score_exact(*args, 32))


def test_degenerate_rows_give_exact_sentinels():
    """An all-padding row is ``(-1, -inf)`` in every slot; a row with one real id that passes
    (all clauses inactive) has it in slot 0 and ``(-1, -inf)`` after it."""
    query, flat, codes, attrs, rev, q_attrs, gs = _exact_args(b=2, p=64)
    flat = torch.full_like(flat, -1)
    flat[1, 17] = 5
    q_attrs = torch.full_like(q_attrs, -1)
    k = 8
    ids, scores = codesigned_probe_score_exact(query, flat, codes, attrs, rev, q_attrs, gs, k)
    assert torch.equal(ids[0], torch.full((k,), -1, device="cuda"))
    assert torch.equal(ids[1, 1:], torch.full((k - 1,), -1, device="cuda"))
    assert ids[1, 0].item() == 5 and torch.isfinite(scores[1, 0])
    tail = torch.cat([scores[0], scores[1, 1:]])
    assert torch.equal(tail, torch.full_like(tail, float("-inf")))
