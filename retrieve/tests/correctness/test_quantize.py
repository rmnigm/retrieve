"""Quantizer correctness: int8 roundtrip and 1-bit Sign-OPORP / SimHash properties."""

from __future__ import annotations

import pytest
import torch

from retrieve.layers.utils.quantize import (
    _build_oporp,
    _pack_signs_to_int64,
    popcount_int64,
    project_oporp_1bit_query,
    project_simhash_1bit_query,
    quantize_int8,
    quantize_oporp_1bit,
    quantize_simhash_1bit,
)
from tests.conftest import make_index, make_query


def test_int8_roundtrip_error_bounded():
    embs = make_index(n=2048, d=128, normalized=False)
    codes, scales = quantize_int8(embs)
    assert codes.dtype == torch.int8
    assert scales.dtype == torch.float32
    assert codes.shape == embs.shape
    assert scales.shape == (embs.shape[0],)

    reconstructed = codes.float() * scales.unsqueeze(1)
    err = (embs - reconstructed).abs()
    abs_max = embs.abs().amax(dim=1, keepdim=True)
    bound = abs_max / 127.0 + 1e-6
    assert (err <= bound).all()


def test_int8_codes_in_range():
    embs = make_index(n=128, d=64)
    codes, _ = quantize_int8(embs)
    assert codes.min() >= -127
    assert codes.max() <= 127


def test_oporp_shapes_and_dtypes():
    embs = make_index(n=512, d=128)
    bits, signs, perm = quantize_oporp_1bit(embs, seed=0)
    assert bits.dtype == torch.int64
    assert bits.shape == (512, 128 // 64)
    assert signs.dtype == torch.int8
    assert signs.shape == (128,)
    assert perm.dtype == torch.int64
    assert perm.shape == (128,)
    assert set(perm.tolist()) == set(range(128))
    assert torch.all((signs == 1) | (signs == -1))


def test_oporp_query_consistency():
    """Same OPORP applied to embs (item-side) and query side yields aligned bits."""
    embs = make_index(n=64, d=128)
    query = make_query(b=4, d=128)

    bits, signs, perm = quantize_oporp_1bit(embs, seed=42)
    query_bits = project_oporp_1bit_query(query, signs, perm)

    # The query path of an emb row must match the item-side bits exactly.
    bits_via_query_path = project_oporp_1bit_query(embs, signs, perm)
    assert torch.equal(bits, bits_via_query_path)
    assert query_bits.shape == (4, 128 // 64)


def test_oporp_dot_product_proxy():
    """Sign-quantized similarity should correlate with true cosine similarity."""
    embs = make_index(n=512, d=128, seed=0)
    query = make_index(n=8, d=128, seed=1)

    bits, signs, perm = quantize_oporp_1bit(embs, seed=7)
    query_bits = project_oporp_1bit_query(query, signs, perm)
    d_total = 64 * bits.shape[1]
    xor = query_bits.unsqueeze(1) ^ bits.unsqueeze(0)
    hamming_sim = (d_total - 2 * popcount_int64(xor).sum(dim=-1)).to(torch.float32)
    cosine = query @ embs.t()

    # Per-row Pearson correlation between hamming_sim and cosine should be high.
    for i in range(query.shape[0]):
        h = hamming_sim[i] - hamming_sim[i].mean()
        c = cosine[i] - cosine[i].mean()
        denom = (h.norm() * c.norm()).clamp_min(1e-8)
        corr = (h * c).sum() / denom
        assert corr > 0.5, f"row {i}: corr={corr:.3f} too low"


def test_oporp_d_must_be_multiple_of_64():
    embs = torch.randn(32, 100, device="cuda")
    try:
        quantize_oporp_1bit(embs, seed=0)
    except ValueError as e:
        assert "multiple of 64" in str(e)
        return
    raise AssertionError("expected ValueError for non-multiple-of-64 D")


def test_int8_clamp_at_extremes():
    """Inputs at ±abs_max round to exactly ±127 with no overflow."""
    embs = torch.tensor(
        [
            [1.0, -1.0, 0.5, -0.25, 1e-8, 0.0, 0.7, -0.7],
            [2.0, -2.0, 1.0, -0.5, 1e-8, 0.0, 1.4, -1.4],
        ],
        device="cuda",
    )
    codes, scales = quantize_int8(embs)
    # The two ±abs_max columns must round to ±127 exactly.
    assert codes[0, 0].item() == 127
    assert codes[0, 1].item() == -127
    assert codes[1, 0].item() == 127
    assert codes[1, 1].item() == -127
    # No overflow into the int8 range past 127.
    assert codes.max() <= 127
    assert codes.min() >= -127
    # Scale equals abs_max / 127 (per-row).
    assert torch.allclose(scales, torch.tensor([1.0, 2.0], device="cuda") / 127.0, atol=1e-6)


def test_int8_zero_input_no_nan():
    """All-zero input must not divide by zero (clamp_min(1e-8) guard)."""
    embs = torch.zeros(4, 8, device="cuda")
    codes, scales = quantize_int8(embs)
    assert torch.all(codes == 0)
    assert torch.isfinite(scales).all()


def test_oporp_seed_determinism():
    """Same seed → identical (bits, signs, perm); different seed → different bits."""
    embs = make_index(n=128, d=128, seed=0)
    a = quantize_oporp_1bit(embs, seed=42)
    b = quantize_oporp_1bit(embs, seed=42)
    assert torch.equal(a[0], b[0])  # bits
    assert torch.equal(a[1], b[1])  # signs
    assert torch.equal(a[2], b[2])  # perm

    c = quantize_oporp_1bit(embs, seed=43)
    # Bits should almost certainly differ at this size; signs and perm definitely.
    assert not torch.equal(a[1], c[1]) or not torch.equal(a[2], c[2])


def test_project_oporp_query_alone():
    """Direct test of the query projection helper, separate from V3."""
    embs = make_index(n=64, d=128, seed=0)
    bits, signs, perm = quantize_oporp_1bit(embs, seed=0)

    # Applying the same projection to the original embs must reproduce ``bits``.
    requeried = project_oporp_1bit_query(embs, signs, perm)
    assert torch.equal(requeried, bits)

    # Independent query batch — shape and dtype.
    query = make_query(b=4, d=128, seed=1)
    out = project_oporp_1bit_query(query, signs, perm)
    assert out.shape == (4, 128 // 64)
    assert out.dtype == torch.int64


def test_project_oporp_query_d_must_match_perm():
    """Mismatched D between query and OPORP buffers should raise (perm dim mismatch)."""
    embs = make_index(n=64, d=128, seed=0)
    _, signs, perm = quantize_oporp_1bit(embs, seed=0)
    bad_query = torch.randn(4, 64, device="cuda")
    try:
        project_oporp_1bit_query(bad_query, signs, perm)
    except (RuntimeError, IndexError):
        return
    raise AssertionError("expected an error when query D differs from OPORP D")


def test_popcount_int64_against_python_reference():
    """Cross-check the SWAR popcount against ``bin(x).count('1')`` row-wise."""
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randint(
        torch.iinfo(torch.int64).min,
        torch.iinfo(torch.int64).max,
        (32,),
        generator=g,
        dtype=torch.int64,
        device="cuda",
    )
    got = popcount_int64(x).cpu().tolist()

    expected = []
    for v in x.cpu().tolist():
        # bin() on negatives loses the sign bit; reinterpret as unsigned 64-bit.
        u = v & 0xFFFFFFFFFFFFFFFF
        expected.append(bin(u).count("1"))
    assert got == expected


def test_popcount_int64_known_constants():
    x = torch.tensor([0, 1, 2, 3, -1, 0x5555555555555555], dtype=torch.int64, device="cuda")
    expected = torch.tensor([0, 1, 1, 2, 64, 32], dtype=torch.int32, device="cuda")
    got = popcount_int64(x)
    assert torch.equal(got, expected)


def test_popcount_int64_rejects_non_int64():
    x = torch.zeros(4, dtype=torch.int32, device="cuda")
    try:
        popcount_int64(x)
    except TypeError:
        return
    raise AssertionError("expected TypeError for non-int64 input")


# ---------------------------------------------------------------------------
# OPORP k_bits parameter: regression at default, paper-faithful at k_bits<D.
# ---------------------------------------------------------------------------


def _oporp_legacy_bits(embs: torch.Tensor, seed: int) -> torch.Tensor:
    """Reference for the pre-fix Sign-OPORP path: no bin-sum, no L2."""
    signs, perm = _build_oporp(embs.shape[1], seed, embs.device)
    proj = (embs * signs.to(embs.dtype)).index_select(1, perm)
    return _pack_signs_to_int64(proj)


def test_oporp_default_kbits_matches_today():
    """Regression: default k_bits → bits byte-identical to the pre-fix path
    (Rademacher signs cancel in XOR, perm is a popcount-axis bijection, and
    the L2 normalize divides by a positive scalar — sign-invariant)."""
    embs = make_index(n=512, d=128, seed=0)
    legacy = _oporp_legacy_bits(embs, seed=0)
    new_default, _, _ = quantize_oporp_1bit(embs, seed=0)
    assert torch.equal(legacy, new_default)


def test_oporp_kbits_lt_d_matches_torch_reference():
    """k_bits=64 on D=128: lib output equals a pure-torch paper-pipeline
    reference (bin-sum → L2 normalize → sign-pack). Covers both item and
    query paths."""
    embs = make_index(n=128, d=128, seed=0)
    query = make_query(b=8, d=128, seed=1)
    k_bits = 64

    bits, signs, perm = quantize_oporp_1bit(embs, seed=3, k_bits=k_bits)

    # Pure-torch reference (item side).
    proj = (embs * signs.to(embs.dtype)).index_select(1, perm)
    binned_ref = proj.view(embs.shape[0], k_bits, 128 // k_bits).sum(dim=-1)
    sketch_ref = binned_ref / binned_ref.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    bits_ref = _pack_signs_to_int64(sketch_ref)
    assert bits.shape == (128, k_bits // 64)
    assert torch.equal(bits, bits_ref)

    # Query path mirrors.
    q_bits = project_oporp_1bit_query(query, signs, perm, k_bits=k_bits)
    qproj = (query * signs.to(query.dtype)).index_select(1, perm)
    qbin = qproj.view(query.shape[0], k_bits, 128 // k_bits).sum(dim=-1)
    qsketch = qbin / qbin.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q_ref = _pack_signs_to_int64(qsketch)
    assert torch.equal(q_bits, q_ref)


@pytest.mark.parametrize("k_bits", [128, 64])
def test_oporp_l2_normalize_is_bit_invariant(k_bits):
    """L2 normalize divides each row by a positive scalar — sign-invariant.
    Bits with the L2 step equal bits without it. Pins the paper-faithful +
    bit-stable property at any k_bits."""
    d = 128
    embs = make_index(n=64, d=d, seed=5)
    bits_with_l2, signs, perm = quantize_oporp_1bit(embs, seed=2, k_bits=k_bits)

    # Reference: same pipeline but skip the L2 normalize.
    proj = (embs * signs.to(embs.dtype)).index_select(1, perm)
    binned = proj.view(embs.shape[0], k_bits, d // k_bits).sum(dim=-1)
    bits_without_l2 = _pack_signs_to_int64(binned)

    assert torch.equal(bits_with_l2, bits_without_l2)


def test_oporp_kbits_validation():
    """Raise on indivisible k_bits and on k_bits not a multiple of 64."""
    embs = make_index(n=4, d=128, seed=0)
    with pytest.raises(ValueError, match="k_bits must divide D"):
        quantize_oporp_1bit(embs, k_bits=96)  # 128 % 96 != 0
    with pytest.raises(ValueError, match="multiple of 64"):
        quantize_oporp_1bit(embs, k_bits=32)  # divides 128 but not %64==0

    # Query path mirrors the same validation.
    _, signs, perm = quantize_oporp_1bit(embs, k_bits=128)
    query = make_query(b=2, d=128, seed=1)
    with pytest.raises(ValueError, match="k_bits must divide D"):
        project_oporp_1bit_query(query, signs, perm, k_bits=96)
    with pytest.raises(ValueError, match="multiple of 64"):
        project_oporp_1bit_query(query, signs, perm, k_bits=32)


# ---------------------------------------------------------------------------
# SimHash 1-bit quantization.
# ---------------------------------------------------------------------------


def test_simhash_shapes_and_dtypes():
    embs = make_index(n=256, d=128, seed=0)
    bits, r = quantize_simhash_1bit(embs, k_bits=256, seed=0)
    assert bits.dtype == torch.int64
    assert bits.shape == (256, 256 // 64)
    assert r.dtype == torch.float32
    assert r.shape == (256, 128)


def test_simhash_query_consistency():
    """Applying ``project_simhash_1bit_query`` to ``embs`` reproduces the
    register-time bits (mirrors ``test_oporp_query_consistency``)."""
    embs = make_index(n=64, d=128, seed=0)
    query = make_query(b=4, d=128, seed=1)
    bits, r = quantize_simhash_1bit(embs, k_bits=256, seed=11)

    bits_via_query_path = project_simhash_1bit_query(embs, r)
    assert torch.equal(bits, bits_via_query_path)

    q_bits = project_simhash_1bit_query(query, r)
    assert q_bits.shape == (4, 256 // 64)
    assert q_bits.dtype == torch.int64


def test_simhash_dot_product_proxy():
    """At k_bits = 4D, SimHash Hamming similarity should correlate with
    cosine similarity (per-row Pearson > 0.5) on isotropic data."""
    d = 128
    embs = make_index(n=512, d=d, seed=0)
    query = make_index(n=8, d=d, seed=1)

    k_bits = 4 * d
    bits, r = quantize_simhash_1bit(embs, k_bits=k_bits, seed=7)
    q_bits = project_simhash_1bit_query(query, r)

    xor = q_bits.unsqueeze(1) ^ bits.unsqueeze(0)
    d_total = 64 * bits.shape[1]
    hamming_sim = (d_total - 2 * popcount_int64(xor).sum(dim=-1)).to(torch.float32)
    cosine = query @ embs.t()

    for i in range(query.shape[0]):
        h = hamming_sim[i] - hamming_sim[i].mean()
        c = cosine[i] - cosine[i].mean()
        denom = (h.norm() * c.norm()).clamp_min(1e-8)
        corr = (h * c).sum() / denom
        assert corr > 0.5, f"row {i}: corr={corr:.3f} too low"


def test_simhash_kbits_validation():
    embs = make_index(n=4, d=128, seed=0)
    with pytest.raises(ValueError, match="multiple of 64"):
        quantize_simhash_1bit(embs, k_bits=100)


def test_simhash_seed_determinism():
    """Same seed → identical (bits, R); different seed → different R."""
    embs = make_index(n=64, d=128, seed=0)
    a_bits, a_r = quantize_simhash_1bit(embs, k_bits=128, seed=42)
    b_bits, b_r = quantize_simhash_1bit(embs, k_bits=128, seed=42)
    assert torch.equal(a_bits, b_bits)
    assert torch.equal(a_r, b_r)

    _, c_r = quantize_simhash_1bit(embs, k_bits=128, seed=43)
    assert not torch.equal(a_r, c_r)
