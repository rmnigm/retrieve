"""Triton ``codesigned_probe_score`` vs the pure-torch SilverTorch phase 2+3."""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.triton.silvertorch.codesigned_probe_score import (
    codesigned_probe_score,
)
from retrieve.layers.silvertorch.bloom import _build_signatures, _generate_seeds
from retrieve.layers.utils.quantize import quantize_int8
from tests.conftest import (
    make_attrs,
    make_index,
    make_mask,
    make_query,
    make_query_attrs,
)
from tests.parity.conftest import assert_topk_matches


def _ref_phase23(
    query,
    flat_items,
    item_codes,
    item_scales,
    k,
    *,
    qb=None,
    bloom_sigs=None,
    mask=None,
):
    valid = flat_items >= 0
    safe = flat_items.clamp(min=0)

    keep = valid
    if qb is not None:
        probed_sigs = bloom_sigs[safe]
        match = (qb.unsqueeze(1) & probed_sigs) == qb.unsqueeze(1)
        keep = keep & match.all(dim=-1)
    if mask is not None:
        keep = keep & mask.gather(1, safe)

    codes = item_codes[safe].float()
    scales = item_scales[safe]
    scores = torch.einsum("bd,bpd->bp", query, codes) * scales
    scores = scores.masked_fill(~keep, float("-inf"))

    actual_k = min(k, scores.shape[1])
    topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
    topk_ids = flat_items.gather(1, topk_local)
    return topk_ids, topk_scores


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
    codes, scales = quantize_int8(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p)

    out_ids, out_scores = codesigned_probe_score(
        query, flat, codes, scales, k
    )
    ref_ids, ref_scores = _ref_phase23(query, flat, codes, scales, k)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


@pytest.mark.parametrize("n,d,p,k", [(2048, 64, 256, 16), (8192, 128, 512, 32)])
@pytest.mark.parametrize("b", [1, 16])
def test_codesigned_with_bloom_matches_ref(n, d, p, k, b):
    embs = make_index(n, d)
    codes, scales = quantize_int8(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p)

    attrs = make_attrs(n, c=2, a_max=2)
    q_attrs = make_query_attrs(b, c=2)
    seeds = _generate_seeds(k_hash=5, device=embs.device)
    sigs = _build_signatures(attrs.long(), seeds, m_bits=512, k_hash=5, word_count=8)
    qb = _build_signatures(
        q_attrs.long().unsqueeze(-1), seeds, m_bits=512, k_hash=5, word_count=8
    )

    out_ids, out_scores = codesigned_probe_score(
        query, flat, codes, scales, k, query_bits=qb, bloom_sigs=sigs
    )
    ref_ids, ref_scores = _ref_phase23(
        query, flat, codes, scales, k, qb=qb, bloom_sigs=sigs
    )
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


@pytest.mark.parametrize("n,d,p,k", [(2048, 64, 256, 16), (8192, 128, 512, 32)])
@pytest.mark.parametrize("b", [1, 16])
def test_codesigned_with_external_mask_matches_ref(n, d, p, k, b):
    embs = make_index(n, d)
    codes, scales = quantize_int8(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p)
    mask = make_mask(b, n, pass_rate=0.3)

    out_ids, out_scores = codesigned_probe_score(
        query, flat, codes, scales, k, mask=mask
    )
    ref_ids, ref_scores = _ref_phase23(query, flat, codes, scales, k, mask=mask)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


@pytest.mark.parametrize("n,d,p,k", [(2048, 64, 256, 16)])
@pytest.mark.parametrize("b", [16])
def test_codesigned_bloom_and_mask_matches_ref(n, d, p, k, b):
    embs = make_index(n, d)
    codes, scales = quantize_int8(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p)
    mask = make_mask(b, n, pass_rate=0.5)

    attrs = make_attrs(n, c=2, a_max=2)
    q_attrs = make_query_attrs(b, c=2)
    seeds = _generate_seeds(k_hash=5, device=embs.device)
    sigs = _build_signatures(attrs.long(), seeds, m_bits=512, k_hash=5, word_count=8)
    qb = _build_signatures(
        q_attrs.long().unsqueeze(-1), seeds, m_bits=512, k_hash=5, word_count=8
    )

    out_ids, out_scores = codesigned_probe_score(
        query, flat, codes, scales, k, query_bits=qb, bloom_sigs=sigs, mask=mask
    )
    ref_ids, ref_scores = _ref_phase23(
        query, flat, codes, scales, k, qb=qb, bloom_sigs=sigs, mask=mask
    )
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


def test_codesigned_pads_when_p_less_than_k():
    n, d, p, k, b = 1024, 64, 8, 32, 4
    embs = make_index(n, d)
    codes, scales = quantize_int8(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p, pad_rate=0.0)

    out_ids, out_scores = codesigned_probe_score(query, flat, codes, scales, k)
    assert out_ids.shape == (b, k)
    assert out_scores.shape == (b, k)
    # Last (k - p) entries on every row must be padding (-1 / -inf).
    assert (out_ids[:, p:] == -1).all()
    assert torch.isinf(out_scores[:, p:]).all()
