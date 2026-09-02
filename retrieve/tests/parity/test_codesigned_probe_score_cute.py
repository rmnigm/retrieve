"""CuTe DSL ``codesigned_probe_score`` vs the CUDA C++ backend, the Triton kernels and the
pure-torch reference.

The cute backend is a one-to-one port of the CUDA C++ backend (same three kernels plus
the generic fallback, same transposed bloom index, same cluster-major 1-bit mask
layout, same fp32 epilogue), so it inherits every gate of
``test_codesigned_probe_score_cuda.py`` — tolerance-based vs the reference, bit-exact vs
Triton, both mask kernels vs their row-wise torch predicates, config plumbing, the
must-reject configs (``ValueError``s from the host module here, where the C++ launcher
raised ``TORCH_CHECK``), the tiny-``max_size`` carry stress and ``opcheck`` — and adds
the one that is the point of the port: **``torch.equal`` against the cuda backend** on
the full ``[B, P]`` score buffers and on the raw mask words, for every scorer
specialization (``D ∈ {64, 128, 256}`` + the generic ``D=96``), every filter mode and
every ``unroll``.

``build_transposed_sigs`` / ``words_per_cluster`` are imported from the cuda module, not
re-implemented, so their bit-layout test stays in the cuda file.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.silvertorch.codesigned_probe_score import (
    _codesigned_probe_score_impl,
)
from retrieve.kernels.silvertorch.codesigned_probe_score_cuda import (
    CodesignedProbeScoreCudaConfig,
    _bloom_partial_mask_cuda_impl,
    _clause_partial_mask_cuda_impl,
    _cps_cuda_prep,
    _cpse_cuda_prep,
    _load_ext,
    words_per_cluster,
)
from retrieve.kernels.silvertorch.codesigned_probe_score_cute import (
    CodesignedProbeScoreCuteConfig,
    _bloom_partial_mask_cute_impl,
    _clause_partial_mask_cute_impl,
    _codesigned_probe_score_cute_impl,
    _codesigned_probe_score_exact_cute_impl,
    _cps_cute_scores,
    codesigned_probe_score_bloom_cute,
    codesigned_probe_score_cute,
    codesigned_probe_score_exact_cute,
)
from retrieve.kernels.silvertorch.codesigned_probe_score_exact import (
    _codesigned_probe_score_exact_impl,
)
from retrieve.layers.filters.exact_attribute import clause_subset_match
from retrieve.layers.utils.quantize import quantize_int8_global
from tests.conftest import make_index, make_query, require_cps_cuda, require_cps_cute
from tests.parity.conftest import (
    assert_ids_equal_up_to_ties,
    assert_topk_matches,
    make_bloom,
    make_exact,
    make_probe_family,
    ref_cps_phase23,
)


@pytest.fixture(autouse=True)
def _needs_cute():
    require_cps_cute()


# d=96 exercises the generic runtime-D kernel (the Triton kernel needs power-of-2 D,
# so the generic path is gated on the torch reference rather than the bit-exact test).
@pytest.mark.parametrize(
    "n_lists,max_size,n_probe,d,k",
    [(16, 64, 4, 64, 8), (64, 96, 8, 128, 32), (32, 64, 8, 96, 16)],
)
@pytest.mark.parametrize("b", [1, 16])
def test_cute_no_filters_matches_ref(n_lists, max_size, n_probe, d, k, b):
    _, _, flat, n = make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)

    out_ids, out_scores = _codesigned_probe_score_cute_impl(query, flat, codes, global_scale, k)
    ref_ids, ref_scores = ref_cps_phase23(query, flat, codes, global_scale, k)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


@pytest.mark.parametrize(
    "n_lists,max_size,n_probe,d,k", [(32, 64, 8, 64, 16), (64, 96, 8, 128, 32)]
)
@pytest.mark.parametrize("b", [1, 16])
def test_cute_with_bloom_matches_ref(n_lists, max_size, n_probe, d, k, b):
    padded, probe_ids, flat, n = make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    sigs, sigs_t, qb = make_bloom(n, b, padded)

    out_ids, out_scores = _codesigned_probe_score_cute_impl(
        query,
        flat,
        codes,
        global_scale,
        k,
        query_bits=qb,
        bloom_sigs_t=sigs_t,
        probe_ids=probe_ids,
    )
    ref_ids, ref_scores = ref_cps_phase23(
        query, flat, codes, global_scale, k, qb=qb, bloom_sigs=sigs
    )
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


# d=96 exercises the generic runtime-D kernel; max_size=96 gives wpc=2 with a 32-bit
# pad tail in the second mask word.
@pytest.mark.parametrize(
    "n_lists,max_size,n_probe,d,k", [(16, 64, 4, 64, 8), (64, 96, 8, 128, 32), (32, 64, 8, 96, 16)]
)
@pytest.mark.parametrize("reverse", ["none", "mixed"])
@pytest.mark.parametrize("b", [1, 16])
def test_cute_exact_matches_ref(n_lists, max_size, n_probe, d, k, reverse, b):
    _, _, flat, n = make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    attrs, rev, q_attrs = make_exact(n, b, reverse=reverse)

    out_ids, out_scores = _codesigned_probe_score_exact_cute_impl(
        query,
        flat,
        codes,
        global_scale,
        k,
        item_clause_attrs=attrs,
        clause_is_reverse=rev,
        query_clause_attrs=q_attrs,
        max_size=max_size,
    )
    ref_ids, ref_scores = ref_cps_phase23(
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


@pytest.mark.parametrize("with_bloom", [False, True])
# d covers all three scorer specializations; the generic runtime-D fallback is gated
# against the torch reference above and against cuda below (Triton needs power-of-2 D).
@pytest.mark.parametrize("d", [64, 128, 256])
def test_cute_matches_triton_bitexact(d, with_bloom):
    """Same probe family through both backends → scores bit-identical; ids identical
    up to permutation within tied scores (see the cuda twin for why the ids get that
    one notch of slack — ``torch.topk`` tie order is not promised stable)."""
    b, n_lists, max_size, n_probe, k = 16, 64, 96, 8, 32
    padded, probe_ids, flat, n = make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)

    if with_bloom:
        sigs, sigs_t, qb = make_bloom(n, b, padded)
        tri_ids, tri_scores = _codesigned_probe_score_impl(
            query, flat, codes, global_scale, k, query_bits=qb, bloom_sigs=sigs
        )
        out_ids, out_scores = _codesigned_probe_score_cute_impl(
            query,
            flat,
            codes,
            global_scale,
            k,
            query_bits=qb,
            bloom_sigs_t=sigs_t,
            probe_ids=probe_ids,
        )
    else:
        tri_ids, tri_scores = _codesigned_probe_score_impl(query, flat, codes, global_scale, k)
        out_ids, out_scores = _codesigned_probe_score_cute_impl(query, flat, codes, global_scale, k)

    assert torch.equal(out_scores, tri_scores), "scores must be bit-identical to Triton"
    assert_ids_equal_up_to_ties(out_ids, tri_ids, out_scores)


@pytest.mark.parametrize("reverse", ["none", "mixed"])
@pytest.mark.parametrize("d", [64, 128, 256])
def test_cute_exact_matches_triton_bitexact(d, reverse):
    """Exact mode, same argument as the bloom case: scores bit-identical, ids identical
    up to permutation within tied scores."""
    b, n_lists, max_size, n_probe, k = 16, 64, 96, 8, 32
    _, _, flat, n = make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    attrs, rev, q_attrs = make_exact(n, b, reverse=reverse)

    tri_ids, tri_scores = _codesigned_probe_score_exact_impl(
        query,
        flat,
        codes,
        global_scale,
        k,
        item_clause_attrs=attrs,
        clause_is_reverse=rev,
        query_clause_attrs=q_attrs,
    )
    out_ids, out_scores = _codesigned_probe_score_exact_cute_impl(
        query,
        flat,
        codes,
        global_scale,
        k,
        item_clause_attrs=attrs,
        clause_is_reverse=rev,
        query_clause_attrs=q_attrs,
        max_size=max_size,
    )

    assert torch.equal(out_scores, tri_scores), "scores must be bit-identical to Triton"
    assert_ids_equal_up_to_ties(out_ids, tri_ids, out_scores)


def _cuda_full_scores(
    q_codes, q_scales, mask, flat, codes, out, *, global_scale, has_mask, max_size, cfg
):
    """The cuda ``cps_scores`` launcher into ``out`` — phase 3 alone, no top-K — so the
    whole ``[B, P]`` buffer can be compared, not just the K survivors."""
    _load_ext().cps_scores(
        q_codes,
        q_scales,
        mask,
        flat,
        codes.contiguous(),
        out,
        float(global_scale),
        has_mask,
        max_size,
        cfg.block_p,
        cfg.num_warps,
        cfg.unroll,
    )
    return out


# The gate this file exists for. All three scorer specializations plus the generic
# runtime-D kernel (d=96), every filter mode, every UNROLL: the cute backend must
# reproduce the cuda backend bit for bit on the raw phase-2 mask words and on the full
# [B, P] phase-3 score buffer (every slot: real dot or -inf), not merely on the top-K.
@pytest.mark.parametrize("unroll", [1, 2, 4])
@pytest.mark.parametrize("mode", ["none", "bloom", "exact"])
@pytest.mark.parametrize("d", [64, 128, 256, 96])
def test_cute_matches_cuda_bitexact(d, mode, unroll):
    require_cps_cuda()
    b, n_lists, max_size, n_probe = 16, 64, 96, 8
    padded, probe_ids, flat, n = make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    cfg_cuda = CodesignedProbeScoreCudaConfig(block_p=128, num_warps=8, unroll=unroll)
    cfg_cute = CodesignedProbeScoreCuteConfig(block_p=128, num_warps=8, unroll=unroll)

    if mode == "exact":
        attrs, rev, q_attrs = make_exact(n, b, reverse="mixed")
        exact_kw = dict(
            item_clause_attrs=attrs,
            clause_is_reverse=rev,
            query_clause_attrs=q_attrs,
            max_size=max_size,
        )
        q_codes, q_scales, flat, scores_cuda = _cpse_cuda_prep(query, flat, codes, **exact_kw)
        mask_cuda = _clause_partial_mask_cuda_impl(flat, attrs, rev, q_attrs, max_size)
        mask_cute = _clause_partial_mask_cute_impl(flat, attrs, rev, q_attrs, max_size)
        has_mask = True
    else:
        bloom_kw = dict(query_bits=None, bloom_sigs_t=None, probe_ids=None)
        if mode == "bloom":
            _, sigs_t, qb = make_bloom(n, b, padded)
            bloom_kw = dict(query_bits=qb, bloom_sigs_t=sigs_t, probe_ids=probe_ids)
        q_codes, q_scales, flat, scores_cuda, has_mask, max_size = _cps_cuda_prep(
            query, flat, codes, **bloom_kw
        )
        if has_mask:
            mask_cuda = _bloom_partial_mask_cuda_impl(qb, sigs_t, probe_ids, max_size)
            mask_cute = _bloom_partial_mask_cute_impl(qb, sigs_t, probe_ids, max_size)
        else:
            # Never dereferenced with has_mask=False (same dummy as the wrappers use).
            mask_cuda = mask_cute = torch.empty(1, 1, dtype=torch.int64, device="cuda")

    assert torch.equal(mask_cuda, mask_cute), "phase-2 mask words must be bit-identical"

    scores_cute = torch.empty_like(scores_cuda)
    _cuda_full_scores(
        q_codes,
        q_scales,
        mask_cuda,
        flat,
        codes,
        scores_cuda,
        global_scale=global_scale,
        has_mask=has_mask,
        max_size=max_size,
        cfg=cfg_cuda,
    )
    _cps_cute_scores(
        q_codes,
        q_scales,
        mask_cute,
        flat,
        codes,
        scores_cute,
        global_scale=global_scale,
        has_mask=has_mask,
        max_size=max_size,
        cfg=cfg_cute,
    )
    assert scores_cuda.shape == flat.shape == (b, n_probe * 96)
    assert torch.equal(scores_cuda, scores_cute), "[B, P] scores must be bit-identical to cuda"
    # Sanity: the comparison is not vacuous — some slots were rejected, some scored.
    finite = torch.isfinite(scores_cute)
    assert finite.any()
    if has_mask:
        assert not finite.all()


@pytest.mark.parametrize("b", [1, 4])
def test_bloom_mask_matches_rowwise(b):
    """Phase-2 kernel vs the row-wise subset test evaluated in torch: the unpacked
    mask bits over every probed slot must equal ``(qb & ~sig) == 0`` (padding slots
    behave as all-zero signatures — pass iff qb has no set bits)."""
    n_lists, max_size, n_probe = 16, 90, 4
    padded, probe_ids, _, n = make_probe_family(b, n_lists, max_size, n_probe)
    sigs, sigs_t, qb = make_bloom(n, b, padded)

    mask = _bloom_partial_mask_cute_impl(qb, sigs_t, probe_ids, max_size)
    wpc = words_per_cluster(max_size)
    assert mask.shape == (b, n_probe * wpc)

    probed = padded[probe_ids]  # [b, n_probe, max_size]
    safe = probed.clamp_min(0)
    gathered = torch.where(
        (probed >= 0).unsqueeze(-1), sigs[safe], torch.zeros((), dtype=torch.int64, device="cuda")
    )  # [b, n_probe, max_size, w]
    expected = ((qb[:, None, None, :] & ~gathered) == 0).all(dim=-1)  # [b, n_probe, max_size]

    words = mask.reshape(b, n_probe, wpc)
    slots = torch.arange(max_size, device="cuda")
    actual = (words[:, :, slots // 64] >> (slots % 64)) & 1
    assert torch.equal(actual.bool(), expected)


@pytest.mark.parametrize("b", [1, 4])
def test_clause_mask_matches_rowwise(b):
    """Phase-2 clause kernel vs ``clause_subset_match`` evaluated in torch over every
    probed slot. ``max_size=90`` is not a multiple of 64, so the second word of each span
    carries a 38-bit pad tail that must come out zero."""
    n_lists, max_size, n_probe = 16, 90, 4
    _, _, flat, n = make_probe_family(b, n_lists, max_size, n_probe)
    attrs, rev, q_attrs = make_exact(n, b, reverse="mixed")

    mask = _clause_partial_mask_cute_impl(flat, attrs, rev, q_attrs, max_size)
    wpc = words_per_cluster(max_size)
    assert mask.shape == (b, n_probe * wpc)

    gathered = attrs[flat.clamp_min(0)]  # [b, P, C, A_max]
    expected = clause_subset_match(gathered, q_attrs, rev) & (flat >= 0)  # [b, P]
    expected = expected.reshape(b, n_probe, max_size)

    words = mask.reshape(b, n_probe, wpc)
    slots = torch.arange(max_size, device="cuda")
    actual = (words[:, :, slots // 64] >> (slots % 64)) & 1
    assert torch.equal(actual.bool(), expected)

    tail = torch.arange(max_size, wpc * 64, device="cuda")
    tail_bits = (words[:, :, tail // 64] >> (tail % 64)) & 1
    assert not tail_bits.any(), "pad tail of the last mask word must be 0"


def test_config_override_matches_default():
    """Plumbing check: deliberately-different ``config``s reach the launch and produce
    **bit-identical** ids / scores on the plain, bloom and exact paths — ``block_p`` /
    ``num_warps`` only re-tile the item range and ``unroll`` only changes how many items
    a segment keeps in flight."""
    b, n_lists, max_size, n_probe, d, k = 4, 16, 64, 4, 64, 8
    padded, probe_ids, flat, n = make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)

    cfg_a = CodesignedProbeScoreCuteConfig(block_p=128, num_warps=4)
    others = [
        CodesignedProbeScoreCuteConfig(block_p=512, num_warps=8),
        CodesignedProbeScoreCuteConfig(block_p=128, num_warps=4, unroll=2),
        CodesignedProbeScoreCuteConfig(block_p=512, num_warps=8, unroll=4),
    ]
    assert all(cfg != cfg_a for cfg in others)
    assert cfg_a.unroll == 1, "the baseline leg must be the un-unrolled loop"

    ids_a, scores_a = _codesigned_probe_score_cute_impl(
        query, flat, codes, global_scale, k, config=cfg_a
    )
    _, sigs_t, qb = make_bloom(n, b, padded)
    bloom_kw = dict(query_bits=qb, bloom_sigs_t=sigs_t, probe_ids=probe_ids)
    bids_a, bscores_a = _codesigned_probe_score_cute_impl(
        query, flat, codes, global_scale, k, **bloom_kw, config=cfg_a
    )
    attrs, rev, q_attrs = make_exact(n, b)
    exact_kw = dict(
        item_clause_attrs=attrs,
        clause_is_reverse=rev,
        query_clause_attrs=q_attrs,
        max_size=max_size,
    )
    eids_a, escores_a = _codesigned_probe_score_exact_cute_impl(
        query, flat, codes, global_scale, k, **exact_kw, config=cfg_a
    )

    for cfg in others:
        ids_b, scores_b = _codesigned_probe_score_cute_impl(
            query, flat, codes, global_scale, k, config=cfg
        )
        assert torch.equal(scores_a, scores_b), f"plain scores differ under {cfg}"
        assert torch.equal(ids_a, ids_b), f"plain ids differ under {cfg}"

        ids_b, scores_b = _codesigned_probe_score_cute_impl(
            query, flat, codes, global_scale, k, **bloom_kw, config=cfg
        )
        assert torch.equal(bscores_a, scores_b), f"bloom scores differ under {cfg}"
        assert torch.equal(bids_a, ids_b), f"bloom ids differ under {cfg}"

        ids_b, scores_b = _codesigned_probe_score_exact_cute_impl(
            query, flat, codes, global_scale, k, **exact_kw, config=cfg
        )
        assert torch.equal(escores_a, scores_b), f"exact scores differ under {cfg}"
        assert torch.equal(eids_a, ids_b), f"exact ids differ under {cfg}"


def test_invalid_config_rejected():
    """The host module carries every ``TORCH_CHECK`` of the C++ launchers as a Python
    check (the DSL kernels have no device-side checks at all), so a config the tuner
    could never produce is a ``ValueError`` here rather than the extension's
    ``RuntimeError``: ``block_p`` must tile evenly into ``num_warps`` warps, and
    ``unroll`` is a ``Constexpr`` of the specialized scorer, so only the enumerated
    values dispatch."""
    b, n_lists, max_size, n_probe, d, k = 2, 8, 64, 2, 64, 4
    _, _, flat, n = make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)

    with pytest.raises(ValueError, match="block_p must be a positive multiple"):
        _codesigned_probe_score_cute_impl(
            query,
            flat,
            codes,
            global_scale,
            k,
            config=CodesignedProbeScoreCuteConfig(block_p=12, num_warps=8),
        )
    with pytest.raises(ValueError, match="unroll must be 1, 2 or 4"):
        _codesigned_probe_score_cute_impl(
            query,
            flat,
            codes,
            global_scale,
            k,
            config=CodesignedProbeScoreCuteConfig(block_p=128, num_warps=4, unroll=3),
        )


@pytest.mark.parametrize("d", [64, 128])
def test_unrolled_mask_indexing_tiny_clusters(d):
    """Carry-loop stress for the mask path: the scorer tracks each segment's
    ``(cluster, slot)`` incrementally instead of dividing by ``max_size`` per item, and
    ``max_size=3`` < the ``SPW * UNROLL`` step means one iteration crosses several cluster
    spans (and at ``d=64`` the warp also runs two segments). ``unroll=4`` must still read
    the same mask bit for every slot as ``unroll=1``."""
    b, n_lists, max_size, n_probe, k = 4, 16, 3, 8, 4
    padded, probe_ids, flat, n = make_probe_family(b, n_lists, max_size, n_probe)
    assert flat.shape[1] == n_probe * max_size == 24
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    _, sigs_t, qb = make_bloom(n, b, padded)

    kw = dict(query_bits=qb, bloom_sigs_t=sigs_t, probe_ids=probe_ids)
    cfg_1 = CodesignedProbeScoreCuteConfig(block_p=128, num_warps=4, unroll=1)
    cfg_4 = CodesignedProbeScoreCuteConfig(block_p=128, num_warps=4, unroll=4)
    ids_1, scores_1 = _codesigned_probe_score_cute_impl(
        query, flat, codes, global_scale, k, **kw, config=cfg_1
    )
    ids_4, scores_4 = _codesigned_probe_score_cute_impl(
        query, flat, codes, global_scale, k, **kw, config=cfg_4
    )
    assert torch.equal(scores_1, scores_4), "unrolled mask indexing changed the scores"
    assert torch.equal(ids_1, ids_4), "unrolled mask indexing changed the ids"


# All three specializations plus d=96 for the generic runtime-D fallback: the op
# contract is D-independent, but the launch it wraps is not, so opcheck's
# fake-vs-real and schema passes should see every dispatch arm.
@pytest.mark.parametrize("d", [64, 128, 256, 96])
def test_opcheck(d):
    """``torch.library.opcheck`` validates the custom-op contract (schema, fake
    kernel, behavior under the compile APIs) for all three registered ops."""
    b, n_lists, max_size, n_probe, k = 2, 8, 64, 2, 4
    padded, probe_ids, flat, n = make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)

    torch.library.opcheck(codesigned_probe_score_cute, (query, flat, codes, global_scale, k))

    _, sigs_t, qb = make_bloom(n, b, padded)
    torch.library.opcheck(
        codesigned_probe_score_bloom_cute,
        (query, flat, codes, qb, sigs_t, probe_ids, global_scale, k),
    )

    attrs, rev, q_attrs = make_exact(n, b)
    torch.library.opcheck(
        codesigned_probe_score_exact_cute,
        (query, flat, codes, attrs, rev, q_attrs, global_scale, k, max_size),
    )
