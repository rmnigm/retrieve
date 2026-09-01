"""Triton ``codesigned_probe_score`` vs the pure-torch SilverTorch phase 2+3."""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.silvertorch.codesigned_probe_score import (
    CodesignedProbeScoreConfig,
    _codesigned_probe_score_impl,
    codesigned_probe_score,
    codesigned_probe_score_bloom,
)
from retrieve.layers.filters.bloom_hash import build_signatures, generate_seeds
from retrieve.layers.utils.quantize import quantize_int8_global
from tests.conftest import (
    make_attrs,
    make_index,
    make_query,
    make_query_attrs,
)
from tests.parity.conftest import assert_topk_matches
from tests.parity.conftest import ref_cps_phase23 as _ref_phase23


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
    ref_ids, ref_scores = _ref_phase23(query, flat, codes, global_scale, k)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


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
    ref_ids, ref_scores = _ref_phase23(query, flat, codes, global_scale, k, qb=qb, bloom_sigs=sigs)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


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
    torch.testing.assert_close(ids_a, ids_b)
    torch.testing.assert_close(scores_a, scores_b)

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
    torch.testing.assert_close(ids_a, ids_b)
    torch.testing.assert_close(scores_a, scores_b)
