from __future__ import annotations

import torch
from torch import Tensor


def popcount_int64(x: Tensor) -> Tensor:
    """Hamming weight (popcount) over an int64 tensor of any shape.

    Bit-twiddle algorithm — matches the Triton-side ``_popcount_int64`` so
    torch reference and Triton kernel agree bit-exact.
    """
    if x.dtype != torch.int64:
        raise TypeError(f"popcount_int64 expects int64, got {x.dtype}")
    M1 = 0x5555555555555555
    M2 = 0x3333333333333333
    M4 = 0x0F0F0F0F0F0F0F0F
    H01 = 0x0101010101010101
    x = x - ((x >> 1) & M1)
    x = (x & M2) + ((x >> 2) & M2)
    x = (x + (x >> 4)) & M4
    return ((x * H01) >> 56).to(torch.int32)


def quantize_int8(embs: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric per-item INT8 quantization.

    ``embs ≈ codes.float() * scales.unsqueeze(1)``.
    """
    abs_max = embs.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = (abs_max / 127.0).squeeze(1)
    codes = (embs / abs_max * 127.0).round().clamp(-128, 127).to(torch.int8)
    return codes, scales


def _build_oporp(d: int, seed: int, device: torch.device) -> tuple[Tensor, Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    signs = (torch.randint(0, 2, (d,), generator=g, device=device, dtype=torch.int8) * 2 - 1)
    perm = torch.randperm(d, generator=g, device=device)
    return signs, perm


def _pack_signs_to_int64(values: Tensor) -> Tensor:
    """Pack sign-quantized [..., D] floats into [..., W] int64 bit words.

    Bit ``b`` of word ``w`` is set iff ``values[..., 64*w + b] > 0``.
    """
    *prefix, d = values.shape
    if d % 64 != 0:
        raise ValueError(f"projected dim must be a multiple of 64, got {d}")
    w = d // 64
    bits = (values > 0).to(torch.int64)
    bits = bits.reshape(*prefix, w, 64)
    shifts = torch.arange(64, device=values.device, dtype=torch.int64)
    return (bits << shifts).sum(dim=-1)


def quantize_oporp_1bit(
    embs: Tensor,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sign-OPORP 1-bit quantization.

    OPORP = One Permutation + One Random projection (Li et al., 2019). For each
    item ``x``, the projection is ``(signs * x)[perm]``; the result is then
    sign-quantized and packed into 64-bit words. Cheap: O(D) state, not O(D²).

    Returns:
        bits: ``[N, W]`` int64 packed sign bits, ``W = D // 64``.
        signs: ``[D]`` int8 in {-1, +1} — Rademacher sign vector.
        perm: ``[D]`` int64 — permutation applied after the sign flip.

    Apply the same ``(signs, perm)`` to a query via ``project_oporp_1bit_query``
    to land in the same bit space. Similarity is then
    ``D - 2 * popcount(query_bits ^ item_bits)``.
    """
    if embs.dim() != 2:
        raise ValueError(f"expected 2-D [N, D] embeddings, got shape {tuple(embs.shape)}")
    n, d = embs.shape
    if d % 64 != 0:
        raise ValueError(f"D must be a multiple of 64 for 1-bit packing, got {d}")
    signs, perm = _build_oporp(d, seed, embs.device)
    proj = (embs * signs.to(embs.dtype)).index_select(1, perm)
    bits = _pack_signs_to_int64(proj)
    return bits, signs, perm


def project_oporp_1bit_query(
    query: Tensor,
    signs: Tensor,
    perm: Tensor,
) -> Tensor:
    """Apply the same Sign-OPORP projection to a query batch.

    Returns ``[B, W]`` int64 packed sign bits.
    """
    if query.dim() != 2:
        raise ValueError(f"expected 2-D [B, D] query, got shape {tuple(query.shape)}")
    proj = (query * signs.to(query.dtype)).index_select(1, perm)
    return _pack_signs_to_int64(proj)
