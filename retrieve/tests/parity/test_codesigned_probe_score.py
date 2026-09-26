"""Triton ``codesigned_probe_score`` vs ``retrieve.ops.reference`` (the pure-torch phase 2+3)."""

from __future__ import annotations

import pytest
import torch

from retrieve.indexing.bloom_hash import build_signatures, generate_seeds
from retrieve.indexing.quantize import quantize_int8_global
from retrieve.ops import reference
from retrieve.ops.triton.codesigned_probe_score import (
    DEFAULT_CONFIG,
    CodesignedProbeScoreConfig,
    _codesigned_probe_score_impl,
    codesigned_probe_score,
    codesigned_probe_score_bloom,
)
from tests.conftest import (
    make_attrs,
    make_index,
    make_query,
    make_query_attrs,
)
from tests.parity.conftest import POISON, assert_topk_equal, make_bloom, poison_empty


def _make_flat_probed(b, n, p, *, pad_rate=0.1, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    flat = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)
    pad = torch.rand(b, p, generator=g, device="cuda") < pad_rate
    flat[pad] = -1
    return flat


@pytest.mark.parametrize("n,d,p,k", [(1024, 64, 128, 8), (8192, 128, 512, 32)])
@pytest.mark.parametrize("b", [1, 16])
def test_codesigned_no_filters_matches_ref(n, d, p, k, b):
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p)

    out_ids, out_scores = codesigned_probe_score(query, flat, codes, global_scale, k)
    ref_ids, ref_scores = reference.codesigned_probe_score(query, flat, codes, global_scale, k)
    assert_topk_equal(out_ids, out_scores, ref_ids, ref_scores)


@pytest.mark.parametrize("n,d,p,k", [(2048, 64, 256, 16), (8192, 128, 512, 32)])
@pytest.mark.parametrize("b", [1, 16])
def test_codesigned_with_bloom_matches_ref(n, d, p, k, b):
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p)

    attrs = make_attrs(n, c=2, a_max=2)
    q_attrs = make_query_attrs(b, c=2)
    seeds = generate_seeds(k_hash=5, device=embs.device)
    sigs = build_signatures(attrs.long(), seeds, m_bits=512, k_hash=5, word_count=8)
    qb = build_signatures(q_attrs.long().unsqueeze(-1), seeds, m_bits=512, k_hash=5, word_count=8)

    out_ids, out_scores = codesigned_probe_score_bloom(
        query, flat, codes, qb, sigs, global_scale, k
    )
    ref_ids, ref_scores = reference.codesigned_probe_score_bloom(
        query, flat, codes, qb, sigs, global_scale, k
    )
    assert_topk_equal(out_ids, out_scores, ref_ids, ref_scores)


def test_config_override_matches_default():
    """Plumbing check: a deliberately-different ``config`` reaches the
    launch and produces identical ids / scores. Exercises both the
    no-bloom and bloom paths."""
    n, d, p, k, b = 1024, 64, 128, 8, 4
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p)

    cfg_a = CodesignedProbeScoreConfig(block_p=32, num_warps=4)
    cfg_b = CodesignedProbeScoreConfig(block_p=128, num_warps=8)
    assert cfg_a != cfg_b

    # No-bloom path (via _impl since the public op drops config=).
    ids_a, scores_a = _codesigned_probe_score_impl(
        query, flat, codes, global_scale, k, config=cfg_a
    )
    ids_b, scores_b = _codesigned_probe_score_impl(
        query, flat, codes, global_scale, k, config=cfg_b
    )
    assert_topk_equal(ids_a, scores_a, ids_b, scores_b)

    # Bloom path.
    attrs = make_attrs(n, c=2, a_max=2)
    q_attrs = make_query_attrs(b, c=2)
    seeds = generate_seeds(k_hash=5, device=embs.device)
    sigs = build_signatures(attrs.long(), seeds, m_bits=512, k_hash=5, word_count=8)
    qb = build_signatures(q_attrs.long().unsqueeze(-1), seeds, m_bits=512, k_hash=5, word_count=8)
    ids_a, scores_a = _codesigned_probe_score_impl(
        query, flat, codes, global_scale, k, query_bits=qb, bloom_sigs=sigs, config=cfg_a
    )
    ids_b, scores_b = _codesigned_probe_score_impl(
        query, flat, codes, global_scale, k, query_bits=qb, bloom_sigs=sigs, config=cfg_b
    )
    assert_topk_equal(ids_a, scores_a, ids_b, scores_b)


@pytest.mark.parametrize("with_bloom", [False, True])
def test_empty_score_buffer_does_not_leak(monkeypatch, with_bloom):
    """The ``[B, P]`` score buffer is ``torch.empty``; the kernel must write every slot (a dot
    or ``-inf`` for padding and bloom rejects). Poisoned allocator, hit count, exact parity."""
    n, d, p, k, b = 1024, 64, 300, 8, 4
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p, pad_rate=0.3)
    if with_bloom:
        seeds = generate_seeds(k_hash=5, device=embs.device)
        sigs = build_signatures(
            make_attrs(n, c=2, a_max=2), seeds, m_bits=512, k_hash=5, word_count=8
        )
        qb = build_signatures(
            make_query_attrs(b, c=2).unsqueeze(-1), seeds, m_bits=512, k_hash=5, word_count=8
        )
        ref = reference.codesigned_probe_score_bloom(query, flat, codes, qb, sigs, global_scale, k)
        hits = poison_empty(monkeypatch, (b, p))
        out = codesigned_probe_score_bloom(query, flat, codes, qb, sigs, global_scale, k)
    else:
        ref = reference.codesigned_probe_score(query, flat, codes, global_scale, k)
        hits = poison_empty(monkeypatch, (b, p))
        out = codesigned_probe_score(query, flat, codes, global_scale, k)
    assert hits, "the score buffer no longer comes from torch.empty — the poison never ran"
    assert not (out[1] == POISON).any(), "poison leaked into top-K: a slot went unwritten"
    assert_topk_equal(*out, *ref)


def _scored(n=1024, d=64, b=4, p=300):
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    return make_query(b, d), _make_flat_probed(b, n, p), codes, global_scale


def test_all_pass_query_bloom_equals_no_bloom():
    """An all-zero query signature is a subset of every item signature: the bloom op must equal
    the no-bloom op bit for bit (catches ``HAS_QB`` keyed to the wrong variant)."""
    query, flat, codes, global_scale = _scored()
    sigs, _ = make_bloom(codes.shape[0], query.shape[0])
    qb = torch.zeros(query.shape[0], sigs.shape[1], dtype=torch.int64, device="cuda")
    k = 16
    out = codesigned_probe_score_bloom(query, flat, codes, qb, sigs, global_scale, k)
    assert_topk_equal(*out, *codesigned_probe_score(query, flat, codes, global_scale, k))


def test_row_alone_equals_row_in_batch_and_item_permutation():
    """Row-local reduction (the query's int8 scale is per row): each row run alone equals its
    row in the batch; permuting the item table and remapping the probe pool permutes the ids and
    leaves every score unchanged. At ``k = P`` so no tie run is cut by the K boundary."""
    query, flat, codes, global_scale = _scored()
    k = flat.shape[1]
    ids, scores = codesigned_probe_score(query, flat, codes, global_scale, k)
    for bi in range(query.shape[0]):
        one = codesigned_probe_score(query[bi : bi + 1], flat[bi : bi + 1], codes, global_scale, k)
        assert_topk_equal(*one, ids[bi : bi + 1], scores[bi : bi + 1])
    g = torch.Generator(device="cuda").manual_seed(9)
    perm = torch.randperm(codes.shape[0], generator=g, device="cuda")
    p_flat = torch.where(flat >= 0, torch.argsort(perm)[flat.clamp_min(0)], -1)
    p_ids, p_scores = codesigned_probe_score(query, p_flat, codes[perm], global_scale, k)
    assert_topk_equal(torch.where(p_ids >= 0, perm[p_ids.clamp_min(0)], -1), p_scores, ids, scores)


@pytest.mark.parametrize("r", [0, 1])
def test_across_tile_cutoff(r):
    """``P % DEFAULT_CONFIG.block_p`` in {0, 1}: a full last tile and a one-lane last tile."""
    p = 2 * DEFAULT_CONFIG.block_p + r
    assert p % DEFAULT_CONFIG.block_p == r
    query, flat, codes, global_scale = _scored(p=p)
    out = codesigned_probe_score(query, flat, codes, global_scale, 32)
    assert_topk_equal(*out, *reference.codesigned_probe_score(query, flat, codes, global_scale, 32))


def test_degenerate_rows_give_exact_sentinels():
    """An all-padding row is ``(-1, -inf)`` in every slot; a row with one real id has it in
    slot 0 and ``(-1, -inf)`` after it."""
    query, flat, codes, global_scale = _scored(b=2, p=64)
    flat = torch.full_like(flat, -1)
    flat[1, 17] = 5
    k = 8
    ids, scores = codesigned_probe_score(query, flat, codes, global_scale, k)
    assert torch.equal(ids[0], torch.full((k,), -1, device="cuda"))
    assert torch.equal(ids[1, 1:], torch.full((k - 1,), -1, device="cuda"))
    assert ids[1, 0].item() == 5 and torch.isfinite(scores[1, 0])
    tail = torch.cat([scores[0], scores[1, 1:]])
    assert torch.equal(tail, torch.full_like(tail, float("-inf")))
