"""CUDA ``codesigned_probe_score`` vs the pure-torch reference and the Triton kernels.

The CUDA backend implements the paper's two-kernel co-design (phase-2 filter → 1-bit
partial masks → masked dp4a scoring). Both filter modes go through it: bloom evaluates
a transposed index against probed cluster ids where the Triton kernel takes row-wise
signatures, and exact evaluates the clause predicate over the probed ids where the
Triton kernel fuses it into the scoring tile. Either way the predicate is
boolean-identical and the int32 dot / fp32 dequant are exact, so on the same probe
family the two backends must produce **bit-identical scores** — asserted with
``torch.equal``, a stronger gate than the tolerance-based reference comparison — and
**ids identical up to permutation within tied scores** (``torch.topk`` does not
promise a stable tie order, and ties occur in this data).
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
    _codesigned_probe_score_cuda_impl,
    _codesigned_probe_score_exact_cuda_impl,
    build_transposed_sigs,
    codesigned_probe_score_bloom_cuda,
    codesigned_probe_score_cuda,
    codesigned_probe_score_exact_cuda,
    words_per_cluster,
)
from retrieve.kernels.silvertorch.codesigned_probe_score_exact import (
    _codesigned_probe_score_exact_impl,
)
from retrieve.layers.filters.bloom_hash import build_signatures, generate_seeds
from retrieve.layers.filters.exact_attribute import clause_subset_match
from retrieve.layers.utils.quantize import quantize_int8_global
from tests.conftest import make_attrs, make_index, make_query, make_query_attrs, require_cps_cuda
from tests.parity.conftest import (
    assert_ids_equal_up_to_ties,
    assert_topk_matches,
    ref_cps_phase23,
)


@pytest.fixture(autouse=True)
def _needs_cuda_ext():
    require_cps_cuda()


def _make_probe_family(b, n_lists, max_size, n_probe, *, pad_rate=0.1, seed=7):
    """Synthetic padded IVF layout + probed clusters: ``padded [n_lists, max_size]``
    holds each item id at most once (scattered via randperm, ``-1`` padding),
    ``flat = padded[probe_ids].reshape(b, -1)`` mirrors the layer's phase 1."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = n_lists * max_size
    padded = torch.randperm(n, generator=g, device="cuda").reshape(n_lists, max_size)
    pad = torch.rand(n_lists, max_size, generator=g, device="cuda") < pad_rate
    padded[pad] = -1
    probe_ids = torch.randint(0, n_lists, (b, n_probe), generator=g, device="cuda")
    flat = padded[probe_ids].reshape(b, -1)
    return padded, probe_ids, flat, n


def _make_bloom(n, b, padded, *, m_bits=512, k_hash=5):
    attrs = make_attrs(n, c=2, a_max=2)
    q_attrs = make_query_attrs(b, c=2)
    seeds = generate_seeds(k_hash=k_hash, device=attrs.device)
    w = m_bits // 64
    sigs = build_signatures(attrs.long(), seeds, m_bits=m_bits, k_hash=k_hash, word_count=w)
    qb = build_signatures(
        q_attrs.long().unsqueeze(-1), seeds, m_bits=m_bits, k_hash=k_hash, word_count=w
    )
    sigs_t = build_transposed_sigs(sigs, padded)
    return sigs, sigs_t, qb


def _make_exact(n, b, *, c=2, a_max=2, reverse="none", n_vocab=8):
    """Item / query clause attrs + reverse flags. The small vocabulary keeps the
    predicate's pass rate high enough that top-K is not all ``-inf``; ``make_query_attrs``
    leaves ~20% of clauses inactive (``-1``). ``reverse="mixed"`` flips clause 0 only, so
    one call exercises both branches of the XOR."""
    attrs = make_attrs(n, c=c, a_max=a_max, n_vocab=n_vocab).long()
    q_attrs = make_query_attrs(b, c=c, n_vocab=n_vocab).long()
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")
    if reverse == "mixed":
        rev[0] = True
    return attrs, rev, q_attrs


# d=96 exercises the generic runtime-D kernel (the Triton kernel needs power-of-2 D,
# so the generic path is gated on the torch reference rather than the bit-exact test).
@pytest.mark.parametrize(
    "n_lists,max_size,n_probe,d,k",
    [(16, 64, 4, 64, 8), (64, 96, 8, 128, 32), (32, 64, 8, 96, 16)],
)
@pytest.mark.parametrize("b", [1, 16])
def test_cuda_no_filters_matches_ref(n_lists, max_size, n_probe, d, k, b):
    _, _, flat, n = _make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)

    out_ids, out_scores = _codesigned_probe_score_cuda_impl(query, flat, codes, global_scale, k)
    ref_ids, ref_scores = ref_cps_phase23(query, flat, codes, global_scale, k)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


@pytest.mark.parametrize(
    "n_lists,max_size,n_probe,d,k", [(32, 64, 8, 64, 16), (64, 96, 8, 128, 32)]
)
@pytest.mark.parametrize("b", [1, 16])
def test_cuda_with_bloom_matches_ref(n_lists, max_size, n_probe, d, k, b):
    padded, probe_ids, flat, n = _make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    sigs, sigs_t, qb = _make_bloom(n, b, padded)

    out_ids, out_scores = _codesigned_probe_score_cuda_impl(
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
def test_cuda_exact_matches_ref(n_lists, max_size, n_probe, d, k, reverse, b):
    _, _, flat, n = _make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    attrs, rev, q_attrs = _make_exact(n, b, reverse=reverse)

    out_ids, out_scores = _codesigned_probe_score_exact_cuda_impl(
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
# d covers all three template specializations; the generic runtime-D fallback is
# gated against the torch reference above (Triton requires power-of-2 D).
@pytest.mark.parametrize("d", [64, 128, 256])
def test_cuda_matches_triton_bitexact(d, with_bloom):
    """Same probe family through both backends → scores bit-identical; ids identical
    up to permutation within tied scores.

    The int32 dot is associativity-exact (|dot| <= 2²² — no overflow), the dequant is
    the same left-associated fp32 multiply pair, the bloom predicate is
    boolean-identical (transposed AND over set query bits ⇔ row-wise subset test),
    and both share the host topk/gather epilogue — so the ``[B, P]`` score tensors are
    equal bit-for-bit and ``torch.equal`` is the hard gate on them. The *ids* get one
    notch of slack because ``torch.topk`` documents its tie order as "not guaranteed
    stable across invocations", and ties do occur here (int dots over ~±1e5 with
    P≈768 → about one tied pair per row); a tie permutation is not a kernel bug."""
    b, n_lists, max_size, n_probe, k = 16, 64, 96, 8, 32
    padded, probe_ids, flat, n = _make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)

    if with_bloom:
        sigs, sigs_t, qb = _make_bloom(n, b, padded)
        tri_ids, tri_scores = _codesigned_probe_score_impl(
            query, flat, codes, global_scale, k, query_bits=qb, bloom_sigs=sigs
        )
        out_ids, out_scores = _codesigned_probe_score_cuda_impl(
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
        out_ids, out_scores = _codesigned_probe_score_cuda_impl(query, flat, codes, global_scale, k)

    assert torch.equal(out_scores, tri_scores), "scores must be bit-identical to Triton"
    assert_ids_equal_up_to_ties(out_ids, tri_ids, out_scores)


@pytest.mark.parametrize("reverse", ["none", "mixed"])
@pytest.mark.parametrize("d", [64, 128, 256])
def test_cuda_exact_matches_triton_bitexact(d, reverse):
    """Exact mode, same argument as the bloom case: scores bit-identical, ids identical
    up to permutation within tied scores.

    The clause predicate is the same AND-of-OR/XOR/inactive boolean function on both
    sides (a warp-ballot mask word here, ``common.clause_pass`` over a tile there), the
    int32 dot is associativity-exact, the dequant is the same fp32 multiply pair, and
    both share the host topk/gather — so the score tensors are equal bit-for-bit. Ids
    are gated up to tie permutation for the reason spelled out in the bloom twin."""
    b, n_lists, max_size, n_probe, k = 16, 64, 96, 8, 32
    _, _, flat, n = _make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    attrs, rev, q_attrs = _make_exact(n, b, reverse=reverse)

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
    out_ids, out_scores = _codesigned_probe_score_exact_cuda_impl(
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


def test_build_transposed_sigs_bits():
    """Bit-level roundtrip of the transposed index: for every (cluster, slot), row m
    of ``sigs_t`` carries exactly bit m of the slot item's row-wise signature (0 for
    padding slots)."""
    n_lists, max_size, w = 8, 90, 4  # non-multiple-of-64 max_size exercises the pad tail
    padded, _, _, n = _make_probe_family(1, n_lists, max_size, 1, pad_rate=0.2)
    g = torch.Generator(device="cuda").manual_seed(3)
    ii = torch.iinfo(torch.int64)
    sigs = torch.randint(ii.min, ii.max, (n, w), generator=g, dtype=torch.int64, device="cuda")

    sigs_t = build_transposed_sigs(sigs, padded)
    wpc = words_per_cluster(max_size)
    assert sigs_t.shape == (w * 64, n_lists * wpc)

    m_bits = w * 64
    m = torch.arange(m_bits, device="cuda")
    for c in range(n_lists):
        for s in range(max_size):
            word = sigs_t[:, c * wpc + s // 64]  # [m_bits]
            actual = (word >> (s % 64)) & 1
            item = int(padded[c, s].item())
            if item < 0:
                expected = torch.zeros(m_bits, dtype=torch.int64, device="cuda")
            else:
                expected = (sigs[item][m // 64] >> (m % 64)) & 1
            assert torch.equal(actual, expected), f"cluster {c} slot {s}"


@pytest.mark.parametrize("b", [1, 4])
def test_bloom_mask_matches_rowwise(b):
    """Phase-2 kernel vs the row-wise subset test evaluated in torch: the unpacked
    mask bits over every probed slot must equal ``(qb & ~sig) == 0`` (padding slots
    behave as all-zero signatures — pass iff qb has no set bits)."""
    n_lists, max_size, n_probe = 16, 90, 4
    padded, probe_ids, _, n = _make_probe_family(b, n_lists, max_size, n_probe)
    sigs, sigs_t, qb = _make_bloom(n, b, padded)

    mask = _bloom_partial_mask_cuda_impl(qb, sigs_t, probe_ids, max_size)
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
    _, _, flat, n = _make_probe_family(b, n_lists, max_size, n_probe)
    attrs, rev, q_attrs = _make_exact(n, b, reverse="mixed")

    mask = _clause_partial_mask_cuda_impl(flat, attrs, rev, q_attrs, max_size)
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
    **bit-identical** ids / scores on the plain, bloom and exact paths.

    ``block_p``/``num_warps`` only re-tile the item range and ``unroll`` only changes how
    many items a segment keeps in flight — the dp4a order, the segment reduction and the
    fp32 epilogue are the same expressions at every config — so ``torch.equal`` is the
    honest gate here, exactly as in the Triton comparison above."""
    b, n_lists, max_size, n_probe, d, k = 4, 16, 64, 4, 64, 8
    padded, probe_ids, flat, n = _make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)

    cfg_a = CodesignedProbeScoreCudaConfig(block_p=128, num_warps=4)
    others = [
        CodesignedProbeScoreCudaConfig(block_p=512, num_warps=8),
        CodesignedProbeScoreCudaConfig(block_p=128, num_warps=4, unroll=2),
        CodesignedProbeScoreCudaConfig(block_p=512, num_warps=8, unroll=4),
    ]
    assert all(cfg != cfg_a for cfg in others)
    assert cfg_a.unroll == 1, "the baseline leg must be the un-unrolled loop"

    ids_a, scores_a = _codesigned_probe_score_cuda_impl(
        query, flat, codes, global_scale, k, config=cfg_a
    )
    _, sigs_t, qb = _make_bloom(n, b, padded)
    bloom_kw = dict(query_bits=qb, bloom_sigs_t=sigs_t, probe_ids=probe_ids)
    bids_a, bscores_a = _codesigned_probe_score_cuda_impl(
        query, flat, codes, global_scale, k, **bloom_kw, config=cfg_a
    )
    attrs, rev, q_attrs = _make_exact(n, b)
    exact_kw = dict(
        item_clause_attrs=attrs,
        clause_is_reverse=rev,
        query_clause_attrs=q_attrs,
        max_size=max_size,
    )
    eids_a, escores_a = _codesigned_probe_score_exact_cuda_impl(
        query, flat, codes, global_scale, k, **exact_kw, config=cfg_a
    )

    for cfg in others:
        ids_b, scores_b = _codesigned_probe_score_cuda_impl(
            query, flat, codes, global_scale, k, config=cfg
        )
        assert torch.equal(scores_a, scores_b), f"plain scores differ under {cfg}"
        assert torch.equal(ids_a, ids_b), f"plain ids differ under {cfg}"

        ids_b, scores_b = _codesigned_probe_score_cuda_impl(
            query, flat, codes, global_scale, k, **bloom_kw, config=cfg
        )
        assert torch.equal(bscores_a, scores_b), f"bloom scores differ under {cfg}"
        assert torch.equal(bids_a, ids_b), f"bloom ids differ under {cfg}"

        ids_b, scores_b = _codesigned_probe_score_exact_cuda_impl(
            query, flat, codes, global_scale, k, **exact_kw, config=cfg
        )
        assert torch.equal(escores_a, scores_b), f"exact scores differ under {cfg}"
        assert torch.equal(eids_a, ids_b), f"exact ids differ under {cfg}"


def test_invalid_config_rejected():
    """The launcher's ``TORCH_CHECK``s are the last line of defence for a config the tuner
    could never produce: ``block_p`` must tile evenly into ``num_warps`` warps, and
    ``unroll`` is a kernel template parameter, so only the instantiated values dispatch."""
    b, n_lists, max_size, n_probe, d, k = 2, 8, 64, 2, 64, 4
    _, _, flat, n = _make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)

    with pytest.raises(RuntimeError, match="block_p must be a positive multiple"):
        _codesigned_probe_score_cuda_impl(
            query,
            flat,
            codes,
            global_scale,
            k,
            config=CodesignedProbeScoreCudaConfig(block_p=12, num_warps=8),
        )
    with pytest.raises(RuntimeError, match="unroll must be 1, 2 or 4"):
        _codesigned_probe_score_cuda_impl(
            query,
            flat,
            codes,
            global_scale,
            k,
            config=CodesignedProbeScoreCudaConfig(block_p=128, num_warps=4, unroll=3),
        )


@pytest.mark.parametrize("d", [64, 128])
def test_unrolled_mask_indexing_tiny_clusters(d):
    """Carry-loop stress for the mask path: the scorer tracks each segment's
    ``(cluster, slot)`` incrementally instead of dividing by ``max_size`` per item, and
    ``max_size=3`` < the ``SPW * UNROLL`` step means one iteration crosses several cluster
    spans (and at ``d=64`` the warp also runs two segments). ``unroll=4`` must still read
    the same mask bit for every slot as ``unroll=1``."""
    b, n_lists, max_size, n_probe, k = 4, 16, 3, 8, 4
    padded, probe_ids, flat, n = _make_probe_family(b, n_lists, max_size, n_probe)
    assert flat.shape[1] == n_probe * max_size == 24
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)
    _, sigs_t, qb = _make_bloom(n, b, padded)

    kw = dict(query_bits=qb, bloom_sigs_t=sigs_t, probe_ids=probe_ids)
    cfg_1 = CodesignedProbeScoreCudaConfig(block_p=128, num_warps=4, unroll=1)
    cfg_4 = CodesignedProbeScoreCudaConfig(block_p=128, num_warps=4, unroll=4)
    ids_1, scores_1 = _codesigned_probe_score_cuda_impl(
        query, flat, codes, global_scale, k, **kw, config=cfg_1
    )
    ids_4, scores_4 = _codesigned_probe_score_cuda_impl(
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
    padded, probe_ids, flat, n = _make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, global_scale = quantize_int8_global(embs)
    query = make_query(b, d)

    torch.library.opcheck(codesigned_probe_score_cuda, (query, flat, codes, global_scale, k))

    _, sigs_t, qb = _make_bloom(n, b, padded)
    torch.library.opcheck(
        codesigned_probe_score_bloom_cuda,
        (query, flat, codes, qb, sigs_t, probe_ids, global_scale, k),
    )

    attrs, rev, q_attrs = _make_exact(n, b)
    torch.library.opcheck(
        codesigned_probe_score_exact_cuda,
        (query, flat, codes, attrs, rev, q_attrs, global_scale, k, max_size),
    )
