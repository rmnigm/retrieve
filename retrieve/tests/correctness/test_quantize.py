"""Quantizer correctness: int8 roundtrip and 1-bit Sign-OPORP properties."""

from __future__ import annotations

import torch

from retrieve.layers.utils.quantize import (
    popcount_int64,
    project_oporp_1bit_query,
    quantize_int8,
    quantize_oporp_1bit,
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
