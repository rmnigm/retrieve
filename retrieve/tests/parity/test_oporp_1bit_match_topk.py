"""Triton ``oporp_1bit_match_topk`` vs torch popcount reference."""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.triton.linr.oporp_1bit_match_topk import (
    Oporp1BitMatchTopkConfig,
    _bucket_n,
    _oporp_1bit_match_topk_impl,
    oporp_1bit_match_topk_full,
    oporp_1bit_match_topk_indirect,
)
from retrieve.layers.utils.quantize import (
    popcount_int64,
    project_oporp_1bit_query,
    quantize_oporp_1bit,
)
from tests.conftest import make_index, make_query
from tests.parity.conftest import assert_topk_matches


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
    topk_ids = pos.gather(1, topk_local)
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


@pytest.mark.parametrize("n,d,k", [(1024, 128, 8), (8192, 128, 32), (4096, 256, 16)])
@pytest.mark.parametrize("b", [1, 16])
def test_oporp_1bit_full_matches_torch(n, d, k, b):
    embs = make_index(n, d)
    query = make_query(b, d)
    item_bits, signs, perm = quantize_oporp_1bit(embs, seed=0)
    query_bits = project_oporp_1bit_query(query, signs, perm)

    out_ids, out_scores = oporp_1bit_match_topk_full(query_bits, item_bits, k)
    ref_ids, ref_scores = _ref_full(query_bits, item_bits, k)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


@pytest.mark.parametrize("n,d,p,k", [(1024, 128, 128, 8), (8192, 128, 512, 32)])
@pytest.mark.parametrize("b", [1, 16])
def test_oporp_1bit_indices_matches_torch(n, d, p, k, b):
    embs = make_index(n, d)
    query = make_query(b, d)
    item_bits, signs, perm = quantize_oporp_1bit(embs, seed=0)
    query_bits = project_oporp_1bit_query(query, signs, perm)

    g = torch.Generator(device="cuda").manual_seed(n + d + p + k + b)
    pos = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)
    counts = torch.full((b,), p, dtype=torch.long, device="cuda")

    out_ids, out_scores = oporp_1bit_match_topk_indirect(
        query_bits, item_bits, k, pos, counts
    )
    ref_ids, ref_scores = _ref_indices(query_bits, item_bits, pos, counts, k)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


def test_partial_counts_handled():
    n, d, p, k, b = 1024, 128, 128, 8, 4
    embs = make_index(n, d)
    query = make_query(b, d)
    item_bits, signs, perm = quantize_oporp_1bit(embs, seed=0)
    query_bits = project_oporp_1bit_query(query, signs, perm)

    g = torch.Generator(device="cuda").manual_seed(0)
    pos = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)
    counts = torch.tensor([p, p // 2, 4, 1], dtype=torch.long, device="cuda")

    out_ids, out_scores = oporp_1bit_match_topk_indirect(
        query_bits, item_bits, k, pos, counts
    )
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
            assert abs(out_scores[bi, j].item() - expected) < 1e-3


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
    torch.testing.assert_close(ids_a, ids_b)
    torch.testing.assert_close(scores_a, scores_b)

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
    torch.testing.assert_close(ids_a, ids_b)
    torch.testing.assert_close(scores_a, scores_b)
