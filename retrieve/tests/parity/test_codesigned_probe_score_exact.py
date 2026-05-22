"""Triton ``codesigned_probe_score_exact`` vs pure-torch reference."""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.triton.silvertorch.codesigned_probe_score_exact import (
    CodesignedProbeScoreExactConfig,
    _codesigned_probe_score_exact_impl,
    codesigned_probe_score_exact,
)
from retrieve.layers.utils.quantize import quantize_int8, quantize_int8_global
from tests.conftest import (
    make_attrs,
    make_index,
    make_query,
    make_query_attrs,
)
from tests.parity.conftest import assert_topk_matches


def _ref_phase23_exact(
    query,
    flat_items,
    item_codes,
    global_scale,
    k,
    *,
    item_clause_attrs,
    clause_is_reverse,
    query_clause_attrs,
):
    """Reference: int8×int8 → int32 dot with one global scale + per-row query
    scale, gated by the exact-clause predicate (AND over clauses, OR over
    A_max within each clause, XOR with reverse, OR with inactive sentinel).
    """
    valid = flat_items >= 0
    safe = flat_items.clamp(min=0)

    qa = query_clause_attrs.long()
    gathered = item_clause_attrs[safe]  # [B, P, C, A_max]
    q = qa.unsqueeze(1).unsqueeze(-1)  # [B, 1, C, 1]
    clause_match = (gathered == q).any(dim=-1)  # [B, P, C]
    rev = clause_is_reverse.unsqueeze(0).unsqueeze(0)  # [1, 1, C]
    clause_match = torch.where(rev, ~clause_match, clause_match)
    inactive = (qa == -1).unsqueeze(1)  # [B, 1, C]
    keep = valid & (clause_match | inactive).all(dim=-1)

    q_codes, q_scales = quantize_int8(query)
    codes = item_codes[safe].float()  # [B, P, D]
    scores = torch.einsum("bd,bpd->bp", q_codes.float(), codes)
    scores = scores * q_scales.unsqueeze(1) * global_scale
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
        query, flat, codes, attrs, rev, q_attrs, global_scale, k,
    )
    ref_ids, ref_scores = _ref_phase23_exact(
        query,
        flat,
        codes,
        global_scale,
        k,
        item_clause_attrs=attrs,
        clause_is_reverse=rev,
        query_clause_attrs=q_attrs,
    )
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


def test_codesigned_exact_pads_when_p_less_than_k():
    n, d, p, k, b, c, a_max = 1024, 64, 8, 32, 4, 2, 2
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    flat = _make_flat_probed(b, n, p, pad_rate=0.0)

    attrs = make_attrs(n, c=c, a_max=a_max).long()
    q_attrs = make_query_attrs(b, c=c).long()
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")

    out_ids, out_scores = codesigned_probe_score_exact(
        query, flat, codes, attrs, rev, q_attrs, global_scale, k,
    )
    assert out_ids.shape == (b, k)
    assert out_scores.shape == (b, k)
    # Last (k - p) entries on every row must be padding (-1 / -inf).
    assert (out_ids[:, p:] == -1).all()
    assert torch.isinf(out_scores[:, p:]).all()


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
        query, flat, codes, attrs, rev_off, q_attrs, global_scale, k,
    )
    ids_on, scores_on = codesigned_probe_score_exact(
        query, flat, codes, attrs, rev_on, q_attrs, global_scale, k,
    )
    # Parity against the reference for both reverse configs.
    ref_off_ids, ref_off_scores = _ref_phase23_exact(
        query, flat, codes, global_scale, k,
        item_clause_attrs=attrs, clause_is_reverse=rev_off, query_clause_attrs=q_attrs,
    )
    ref_on_ids, ref_on_scores = _ref_phase23_exact(
        query, flat, codes, global_scale, k,
        item_clause_attrs=attrs, clause_is_reverse=rev_on, query_clause_attrs=q_attrs,
    )
    assert_topk_matches(ids_off, scores_off, ref_off_ids, ref_off_scores)
    assert_topk_matches(ids_on, scores_on, ref_on_ids, ref_on_scores)


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
        query, flat, codes, attrs, rev, q_attrs, global_scale, k,
    )

    from retrieve.kernels.triton.silvertorch.codesigned_probe_score import (
        _codesigned_probe_score_impl,
    )
    ref_ids, ref_scores = _codesigned_probe_score_impl(
        query, flat, codes, global_scale, k
    )
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


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
        query, flat, codes, global_scale, k,
        item_clause_attrs=attrs, clause_is_reverse=rev, query_clause_attrs=q_attrs,
        config=cfg_a,
    )
    ids_b, scores_b = _codesigned_probe_score_exact_impl(
        query, flat, codes, global_scale, k,
        item_clause_attrs=attrs, clause_is_reverse=rev, query_clause_attrs=q_attrs,
        config=cfg_b,
    )
    torch.testing.assert_close(ids_a, ids_b)
    torch.testing.assert_close(scores_a, scores_b)
