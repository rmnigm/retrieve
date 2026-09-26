"""Every scoring kernel against an fp64 oracle of the same operation on the same inputs (plan L5).

The other parity files compare ``ops.triton`` against ``ops.reference`` at the *same* input dtype,
so an accumulation-width defect on both sides cancels and passes; ``fused_masked_knn_topk``
accumulated fp16 through every gate this way (L4 §6.1: 0.0276 max abs error). Each test here
asks for every candidate (``k = P``), maps the returned ids back to the fp64 truth, and asserts a
bound that fp32 accumulation meets and fp16 does not — the second half is asserted on the same
inputs, so the bound's discriminating power is checked on every run, not claimed.

Inputs are goodreads-like on purpose: unnormalised, ``|score|`` up to ~10², partial sums two orders
above the top scores.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from retrieve.functional import compact_mask, popcount_int64
from retrieve.indexing.quantize import quantize_int8, quantize_int8_global
from retrieve.modules.knn import PostfilterKNN, PrefilterKNN
from retrieve.ops.triton.codesigned_probe_score import codesigned_probe_score
from retrieve.ops.triton.codesigned_probe_score_exact import codesigned_probe_score_exact
from retrieve.ops.triton.fused_masked_knn_topk import fused_masked_knn_topk
from retrieve.ops.triton.oporp_1bit_match_topk import oporp_1bit_match_topk_indirect
from tests.conftest import (
    assert_topk_id_sets_match,
    make_index,
    make_mask,
    make_query,
    recall_at_k,
)
from tests.parity.conftest import make_exact, make_probe_family

FP32_DOT_ABS = 1e-4  # measured: fp32 ≤ 1.4e-5 at D=256, |s| ≤ 140; the fp16 tree ≥ 6e-2
INT8_DEQUANT_REL = 1e-6  # measured: 1.1e-7 (int32 dot exact; two fp32 multiplies)
NEAR_TIE_DOT_ABS = 2e-6  # cuBLAS's tensor-core accumulator: 1.1e-6 at D=128; fp16 quantum 4.9e-4


def _rows(ids):
    return torch.arange(ids.shape[0], device=ids.device)[:, None].expand_as(ids)


def _fp16_tree(prod):
    while prod.shape[-1] > 1:
        prod = (prod[..., 0::2] + prod[..., 1::2]).half()
    return prod[..., 0]


@pytest.mark.parametrize("d", [64, 128, 256])
def test_fused_masked_knn_topk_accumulates_fp32(d):
    b, n = 16, 16_384
    embs = make_index(n, d, normalized=False).half()
    query = (make_query(b, d, normalized=False) * 2).half()
    pos, counts = compact_mask(make_mask(b, n, pass_rate=0.5))
    p = pos.shape[1]

    ids, scores = fused_masked_knn_topk(query, embs, pos, counts, p)
    finite = torch.isfinite(scores)
    rows = embs[ids.clamp_min(0)]
    q = query[_rows(ids)]
    truth = (rows.double() * q.double()).sum(-1)

    err = (scores.double() - truth).abs()[finite].max().item()
    err_fp16 = (_fp16_tree(rows * q).double() - truth).abs()[finite].max().item()
    assert err <= FP32_DOT_ABS, f"kernel vs fp64: {err:.3e}"
    assert err_fp16 > FP32_DOT_ABS, f"the bound does not discriminate fp16: {err_fp16:.3e}"


def _int8_family(d, b=8):
    """A probe with an identity ``sort_perm``, so a returned id is also its row of the
    cluster-sorted ``codes`` the oracle reads."""
    lay = make_probe_family(b, 64, 128, 8)
    lay = replace(lay, sort_perm=torch.arange(lay.n, device="cuda"))
    query = make_query(b, d)
    codes, global_scale = quantize_int8_global(make_index(lay.n, d))
    return query, lay, codes, global_scale


def _assert_int8_dequant(query, codes, global_scale, ids, scores):
    q_codes, q_scales = quantize_int8(query)
    finite = torch.isfinite(scores)
    rows = _rows(ids)
    dot = (q_codes[rows].double() * codes[ids.clamp_min(0)].double()).sum(-1)
    truth = dot * q_scales[rows].double() * global_scale
    nonzero = finite & (dot != 0)

    rel = ((scores.double() - truth).abs() / truth.abs())[nonzero].max().item()
    rel_fp16 = ((dot.half().double() - dot).abs() / dot.abs())[nonzero].max().item()
    assert rel <= INT8_DEQUANT_REL, f"kernel vs fp64: {rel:.3e}"
    assert rel_fp16 > INT8_DEQUANT_REL, f"the bound does not discriminate fp16: {rel_fp16:.3e}"


@pytest.mark.parametrize("d", [64, 128])
def test_codesigned_probe_score_int32_dot_fp32_dequant(d):
    query, lay, codes, global_scale = _int8_family(d)
    ids, scores = codesigned_probe_score(
        query, lay.probe_ids, lay.cluster_offsets, codes, lay.sort_perm, global_scale,
        lay.width, lay.width,
    )  # fmt: skip
    _assert_int8_dequant(query, codes, global_scale, ids, scores)


@pytest.mark.parametrize("d", [64, 128])
def test_codesigned_probe_score_exact_int32_dot_fp32_dequant(d):
    query, lay, codes, global_scale = _int8_family(d)
    attrs, rev, q_attrs = make_exact(lay.n, query.shape[0])
    ids, scores = codesigned_probe_score_exact(
        query, lay.probe_ids, lay.cluster_offsets, codes, lay.sort_perm, attrs, rev, q_attrs,
        global_scale, lay.width, lay.width,
    )  # fmt: skip
    _assert_int8_dequant(query, codes, global_scale, ids, scores)


def test_oporp_1bit_match_topk_is_exact():
    b, n, w = 8, 4096, 4
    g = torch.Generator(device="cuda").manual_seed(5)
    item_bits = torch.randint(-(2**63), 2**63 - 1, (n, w), generator=g, device="cuda")
    query_bits = torch.randint(-(2**63), 2**63 - 1, (b, w), generator=g, device="cuda")
    pos, counts = compact_mask(make_mask(b, n, pass_rate=0.5))
    p = pos.shape[1]

    ids, scores = oporp_1bit_match_topk_indirect(query_bits, item_bits, p, pos, counts)
    finite = torch.isfinite(scores)
    hamming = popcount_int64(query_bits[_rows(ids)] ^ item_bits[ids.clamp_min(0)]).sum(-1)
    truth = (64 * w - 2 * hamming).double()
    assert torch.equal(scores.double()[finite], truth[finite])


def _near_tied(b, n, d, *, seed=11):
    """YFCC-shaped scores: unit-norm queries near one direction ``u`` and items at cosine
    ``c ∈ [0.80, 0.81]`` to it, so every row's top-k spans a few fp16 quanta (2⁻¹¹ near 0.8)
    and the fp16 score of a whole run of items is the same number. Returned fp16, the precision
    the modules store, so the fp64 truth is of the exact same inputs."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    u = F.normalize(torch.randn(d, generator=g, device="cuda"), dim=0)
    query = F.normalize(u + 0.002 * torch.randn(b, d, generator=g, device="cuda"), dim=1)
    v = torch.randn(n, d, generator=g, device="cuda")
    v = F.normalize(v - (v @ u)[:, None] * u, dim=1)
    c = 0.80 + 0.01 * torch.rand(n, 1, generator=g, device="cuda")
    items = c * u + (1 - c**2).sqrt() * v
    return query.half(), items.half()


def _linr_scores(path, query, items, mask, k):
    """``(ids, scores)`` of one exact LiNR scoring path; ``mask`` is ``None`` on the unmasked
    paths."""
    if path.startswith("postfilter"):
        m = PostfilterKNN(k=k)
        m.register_index(items)
        return m(query, mask=mask)
    m = PrefilterKNN(k=k, backend=path.removeprefix("prefilter-"))
    m.register_index(items)
    if mask is None:
        return m(query)
    pos, counts = compact_mask(mask)
    return m(query, candidate_ids=pos, counts=counts)


@pytest.mark.parametrize(
    "path,masked",
    [
        ("postfilter", False),
        ("postfilter", True),
        ("prefilter-triton", False),
        ("prefilter-torch", True),
        ("prefilter-triton", True),
    ],
)
def test_exact_linr_selects_the_fp64_topk_near_ties(path, masked):
    """The exact LiNR scorers return fp32 scores within ``NEAR_TIE_DOT_ABS`` of the fp64 dot of
    their own fp16 inputs and select the fp64 top-k (up to ties inside that bound). An fp16
    score rounds the boundary runs together and misses; asserted on the same inputs, so the
    gate's discriminating power is checked, not claimed."""
    b, n, d, k = 8, 8192, 128, 100
    query, items = _near_tied(b, n, d)
    mask = make_mask(b, n, pass_rate=0.5) if masked else None

    ids, scores = _linr_scores(path, query, items, mask, k)
    truth = query.double() @ items.double().t()
    if mask is not None:
        truth = truth.masked_fill(~mask, float("-inf"))
    ref_scores, ref_ids = torch.topk(truth, k, dim=1)

    assert scores.dtype == torch.float32
    err = (scores.double() - truth.gather(1, ids)).abs().max().item()
    assert err <= NEAR_TIE_DOT_ABS, f"module vs fp64: {err:.3e}"
    for row in range(b):
        assert_topk_id_sets_match(
            ids, scores, ref_ids, ref_scores, row, atol=NEAR_TIE_DOT_ABS, rtol=0.0
        )

    fp16_ids = torch.topk(truth.half(), k, dim=1).indices
    assert recall_at_k(fp16_ids, ref_ids) < 0.95, "the case does not discriminate fp16 scores"
