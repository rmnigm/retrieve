"""Triton ``oporp_1bit_match_topk`` vs torch popcount reference."""

from __future__ import annotations

import pytest
import torch

from retrieve.functional import popcount_int64
from retrieve.indexing.quantize import (
    project_oporp_1bit_query,
    project_simhash_1bit_query,
    quantize_oporp_1bit,
    quantize_simhash_1bit,
)
from retrieve.ops.triton.oporp_1bit_match_topk import (
    _N_BUCKETS,
    DEFAULT_CONFIG,
    Oporp1BitMatchTopkConfig,
    _bucket_n,
    _oporp_1bit_match_topk_impl,
    oporp_1bit_match_topk_full,
    oporp_1bit_match_topk_indirect,
)
from tests.conftest import make_index, make_query
from tests.parity.conftest import POISON, assert_topk_equal, poison_empty


def _make_bits(quant: str, embs: torch.Tensor, query: torch.Tensor, k_bits: int):
    """Build (item_bits, query_bits) for either quantizer at the given k_bits.

    SimHash uses ``r`` instead of ``(signs, perm)``; the kernel only sees
    ``[N, W]`` int64 bit-words, so the algorithm is opaque to it.
    """
    if quant == "oporp":
        item_bits, signs, perm = quantize_oporp_1bit(embs, seed=0, k_bits=k_bits)
        query_bits = project_oporp_1bit_query(query, signs, perm, k_bits=k_bits)
    elif quant == "simhash":
        item_bits, r = quantize_simhash_1bit(embs, k_bits=k_bits, seed=0)
        query_bits = project_simhash_1bit_query(query, r)
    else:
        raise ValueError(f"unknown quant: {quant}")
    return item_bits, query_bits


def _ref_full(query_bits: torch.Tensor, item_bits: torch.Tensor, k: int):
    d_total = 64 * item_bits.shape[1]
    xor = query_bits.unsqueeze(1) ^ item_bits.unsqueeze(0)
    hamming = popcount_int64(xor).sum(dim=-1)
    scores = (d_total - 2 * hamming).to(torch.float32)
    topk_scores, topk_ids = torch.topk(scores, k, dim=1)
    return topk_ids.to(torch.long), topk_scores


def _ref_indices(
    query_bits: torch.Tensor,
    item_bits: torch.Tensor,
    pos: torch.Tensor,
    counts: torch.Tensor,
    k: int,
):
    d_total = 64 * item_bits.shape[1]
    cand_bits = item_bits[pos]  # [B, P, W]
    xor = query_bits.unsqueeze(1) ^ cand_bits
    hamming = popcount_int64(xor).sum(dim=-1)
    scores = (d_total - 2 * hamming).to(torch.float32)
    p = pos.shape[1]
    valid = torch.arange(p, device=pos.device).unsqueeze(0) < counts.unsqueeze(1)
    scores = scores.masked_fill(~valid, float("-inf"))
    actual_k = min(k, p)
    topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
    topk_ids = torch.where(torch.isfinite(topk_scores), pos.gather(1, topk_local), -1)
    if actual_k < k:
        b = pos.shape[0]
        pad = k - actual_k
        topk_ids = torch.cat(
            [topk_ids, torch.full((b, pad), -1, dtype=torch.long, device=pos.device)],
            dim=1,
        )
        topk_scores = torch.cat(
            [
                topk_scores,
                torch.full((b, pad), float("-inf"), dtype=torch.float32, device=pos.device),
            ],
            dim=1,
        )
    return topk_ids, topk_scores


@pytest.mark.parametrize("quant", ["oporp", "simhash"])
@pytest.mark.parametrize("n,d,k", [(1024, 128, 8), (8192, 128, 32), (4096, 256, 16)])
@pytest.mark.parametrize("b", [1, 16])
def test_oporp_1bit_full_matches_torch(n, d, k, b, quant):
    embs = make_index(n, d)
    query = make_query(b, d)
    item_bits, query_bits = _make_bits(quant, embs, query, k_bits=d)

    out_ids, out_scores = oporp_1bit_match_topk_full(query_bits, item_bits, k)
    ref_ids, ref_scores = _ref_full(query_bits, item_bits, k)
    assert_topk_equal(out_ids, out_scores, ref_ids, ref_scores)


@pytest.mark.parametrize("quant", ["oporp", "simhash"])
@pytest.mark.parametrize("n,d,p,k", [(1024, 128, 128, 8), (8192, 128, 512, 32)])
@pytest.mark.parametrize("b", [1, 16])
def test_oporp_1bit_indices_matches_torch(n, d, p, k, b, quant):
    embs = make_index(n, d)
    query = make_query(b, d)
    item_bits, query_bits = _make_bits(quant, embs, query, k_bits=d)

    g = torch.Generator(device="cuda").manual_seed(n + d + p + k + b)
    pos = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)
    counts = torch.full((b,), p, dtype=torch.long, device="cuda")

    out_ids, out_scores = oporp_1bit_match_topk_indirect(query_bits, item_bits, k, pos, counts)
    ref_ids, ref_scores = _ref_indices(query_bits, item_bits, pos, counts, k)
    assert_topk_equal(out_ids, out_scores, ref_ids, ref_scores)


def test_partial_counts_handled():
    n, d, p, k, b = 1024, 128, 128, 8, 4
    embs = make_index(n, d)
    query = make_query(b, d)
    item_bits, signs, perm = quantize_oporp_1bit(embs, seed=0)
    query_bits = project_oporp_1bit_query(query, signs, perm)

    g = torch.Generator(device="cuda").manual_seed(0)
    pos = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)
    counts = torch.tensor([p, p // 2, 4, 1], dtype=torch.long, device="cuda")

    out_ids, out_scores = oporp_1bit_match_topk_indirect(query_bits, item_bits, k, pos, counts)
    for bi in range(b):
        valid_pool = set(pos[bi, : counts[bi].item()].tolist())
        for j in range(k):
            if torch.isfinite(out_scores[bi, j]):
                assert out_ids[bi, j].item() in valid_pool


def test_score_relation_holds():
    """Score must equal D - 2 * hamming (sanity for the kernel arithmetic)."""
    n, d, k, b = 256, 128, 4, 2
    embs = make_index(n, d)
    query = make_query(b, d)
    item_bits, signs, perm = quantize_oporp_1bit(embs, seed=0)
    query_bits = project_oporp_1bit_query(query, signs, perm)

    out_ids, out_scores = oporp_1bit_match_topk_full(query_bits, item_bits, k)
    for bi in range(b):
        for j in range(k):
            iid = int(out_ids[bi, j].item())
            xor = query_bits[bi] ^ item_bits[iid]
            hamming = int(popcount_int64(xor).sum().item())
            expected = d - 2 * hamming
            assert out_scores[bi, j].item() == expected


def test_bucket_n_ladder():
    """``_bucket_n`` rounds runtime candidate width up to a fixed ladder so
    the kernel's ``N: tl.constexpr`` only takes a handful of distinct
    values (one JIT compile per bucket × W)."""
    assert _bucket_n(1) == 4096
    assert _bucket_n(4096) == 4096
    assert _bucket_n(4097) == 65536
    assert _bucket_n(65536) == 65536
    assert _bucket_n(1_000_000) == 1_048_576
    assert _bucket_n(1_048_577) == 16_777_216
    # Past the last static bucket: next power of 2.
    assert _bucket_n(20_000_000) == 1 << 25


def test_config_override_matches_default_full_and_indexed():
    """Plumbing check: a deliberately-different ``config`` reaches the
    launch and produces identical ids / scores. Exercises both
    full-scan and has-indices paths. The public custom_ops drop the
    ``config`` kwarg (schema can't carry dataclasses), so this calls
    ``_oporp_1bit_match_topk_impl`` directly."""
    n, d, p, k, b = 1024, 128, 128, 8, 4
    embs = make_index(n, d)
    query = make_query(b, d)
    item_bits, signs, perm = quantize_oporp_1bit(embs, seed=0)
    query_bits = project_oporp_1bit_query(query, signs, perm)

    cfg_a = Oporp1BitMatchTopkConfig(block_n=64, num_warps=4)
    cfg_b = Oporp1BitMatchTopkConfig(block_n=256, num_warps=8)
    assert cfg_a != cfg_b

    # Full-scan path.
    ids_a, scores_a = _oporp_1bit_match_topk_impl(
        query_bits, item_bits, k, None, None, config=cfg_a
    )
    ids_b, scores_b = _oporp_1bit_match_topk_impl(
        query_bits, item_bits, k, None, None, config=cfg_b
    )
    assert_topk_equal(ids_a, scores_a, ids_b, scores_b)

    # Has-indices path.
    g = torch.Generator(device="cuda").manual_seed(42)
    pos = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)
    counts = torch.full((b,), p, dtype=torch.long, device="cuda")
    ids_a, scores_a = _oporp_1bit_match_topk_impl(
        query_bits, item_bits, k, pos, counts, config=cfg_a
    )
    ids_b, scores_b = _oporp_1bit_match_topk_impl(
        query_bits, item_bits, k, pos, counts, config=cfg_b
    )
    assert_topk_equal(ids_a, scores_a, ids_b, scores_b)


@pytest.mark.parametrize("path", ["full", "indirect"])
def test_empty_score_buffer_does_not_leak(monkeypatch, path):
    """The score buffer is ``torch.empty`` — ``[B, N]`` on the full scan, ``[B,
    max(_bucket_n(P), _bucket_n(k))]`` on the indirect path, whose lanes past ``counts[b]`` and
    past ``P`` must be written ``-inf``. Poisoned allocator, hit count, exact parity."""
    n, d, p, k, b = 1024, 128, 100, 8, 4
    embs = make_index(n, d)
    query = make_query(b, d)
    item_bits, query_bits = _make_bits("oporp", embs, query, k_bits=d)
    if path == "full":
        ref = _ref_full(query_bits, item_bits, k)
        hits = poison_empty(monkeypatch, (b, n))
        out = oporp_1bit_match_topk_full(query_bits, item_bits, k)
    else:
        g = torch.Generator(device="cuda").manual_seed(5)
        pos = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)
        counts = torch.tensor([p, 50, 3, 0], dtype=torch.long, device="cuda")
        ref = _ref_indices(query_bits, item_bits, pos, counts, k)
        hits = poison_empty(monkeypatch, (b, max(_bucket_n(p), _bucket_n(k))))
        out = oporp_1bit_match_topk_indirect(query_bits, item_bits, k, pos, counts)
    assert hits, "the score buffer no longer comes from torch.empty — the poison never ran"
    assert not (out[1] == POISON).any(), "poison leaked into top-K: a slot went unwritten"
    assert_topk_equal(*out, *ref)


def _oporp_data(n=1024, d=128, b=4):
    embs = make_index(n, d)
    item_bits, query_bits = _make_bits("oporp", embs, make_query(b, d), k_bits=d)
    return item_bits, query_bits


def test_indirect_over_every_item_equals_full_scan():
    """``positive_indices = arange(N)`` with ``counts = N`` is the full scan: a constexpr keyed
    to the wrong ``HAS_INDICES`` variant, or a reused compilation, breaks this."""
    item_bits, query_bits = _oporp_data()
    n, b, k = item_bits.shape[0], query_bits.shape[0], 16
    pos = torch.arange(n, device="cuda").expand(b, n).contiguous()
    counts = torch.full((b,), n, dtype=torch.long, device="cuda")
    out = oporp_1bit_match_topk_indirect(query_bits, item_bits, k, pos, counts)
    assert_topk_equal(*out, *oporp_1bit_match_topk_full(query_bits, item_bits, k))


def test_row_alone_equals_row_in_batch_and_item_permutation():
    """Row-local reduction: each row run alone equals its row in the batch; permuting the item
    table permutes the returned ids and leaves every score unchanged."""
    item_bits, query_bits = _oporp_data()
    k = 16
    ids, scores = oporp_1bit_match_topk_full(query_bits, item_bits, k)
    for bi in range(query_bits.shape[0]):
        one = oporp_1bit_match_topk_full(query_bits[bi : bi + 1], item_bits, k)
        assert_topk_equal(*one, ids[bi : bi + 1], scores[bi : bi + 1])
    # k = N: a tie run cut by the K boundary would resolve by item order, which the
    # permutation changes.
    n = item_bits.shape[0]
    g = torch.Generator(device="cuda").manual_seed(9)
    perm = torch.randperm(n, generator=g, device="cuda")
    p_ids, p_scores = oporp_1bit_match_topk_full(query_bits, item_bits[perm], n)
    assert_topk_equal(perm[p_ids], p_scores, *oporp_1bit_match_topk_full(query_bits, item_bits, n))


@pytest.mark.parametrize(
    "p,regime",
    [
        (_N_BUCKETS[0] - 1, "below the first bucket edge"),
        (_N_BUCKETS[0], "on the first bucket edge"),
        (_N_BUCKETS[0] + 1, "one past the first bucket edge"),
        (8 * DEFAULT_CONFIG.block_n, "p % block_n == 0"),
        (8 * DEFAULT_CONFIG.block_n + 1, "p % block_n == 1"),
    ],
)
def test_indirect_across_bucket_and_tile_cutoffs(p, regime):
    """The indirect launch width comes from ``_N_BUCKETS`` and its tiles from
    ``DEFAULT_CONFIG.block_n``; both sides of each cutoff are exact against the oracle."""
    item_bits, query_bits = _oporp_data(n=8192)
    b, k = query_bits.shape[0], 32
    width = max(_bucket_n(p), _bucket_n(k))
    if "bucket" in regime:
        assert width == (_N_BUCKETS[0] if p <= _N_BUCKETS[0] else _N_BUCKETS[1]), regime
    else:
        assert p % DEFAULT_CONFIG.block_n == int(regime[-1]), regime
    g = torch.Generator(device="cuda").manual_seed(p)
    pos = torch.randint(0, item_bits.shape[0], (b, p), generator=g, device="cuda")
    counts = torch.tensor([p, p - 1, p // 2, 1], dtype=torch.long, device="cuda")
    out = oporp_1bit_match_topk_indirect(query_bits, item_bits, k, pos, counts)
    assert_topk_equal(*out, *_ref_indices(query_bits, item_bits, pos, counts, k))


def test_degenerate_rows_give_exact_sentinels():
    """A ``counts = 0`` row is ``(-1, -inf)`` in every slot; a ``counts = 1`` row has its one
    candidate in slot 0 and ``(-1, -inf)`` after it."""
    item_bits, query_bits = _oporp_data(b=2)
    k, p = 8, 64
    pos = torch.arange(p, device="cuda").expand(2, p).contiguous()
    counts = torch.tensor([0, 1], dtype=torch.long, device="cuda")
    ids, scores = oporp_1bit_match_topk_indirect(query_bits, item_bits, k, pos, counts)
    assert torch.equal(ids[0], torch.full((k,), -1, device="cuda"))
    assert torch.equal(ids[1, 1:], torch.full((k - 1,), -1, device="cuda"))
    assert ids[1, 0].item() == 0 and torch.isfinite(scores[1, 0])
    tail = torch.cat([scores[0], scores[1, 1:]])
    assert torch.equal(tail, torch.full_like(tail, float("-inf")))
