from __future__ import annotations

import torch
from torch import Tensor


def popcount_int64(x: Tensor) -> Tensor:
    """Hamming weight (popcount) over an int64 tensor of any shape.

    Bit-twiddle algorithm — matches the Triton-side ``_popcount_int64`` so torch reference and
    Triton kernel agree bit-exact."""
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

    ``embs ≈ codes.float() * scales.unsqueeze(1)``."""
    abs_max = embs.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = (abs_max / 127.0).squeeze(1)
    codes = (embs / abs_max * 127.0).round().clamp(-128, 127).to(torch.int8)
    return codes, scales


def quantize_int8_global(embs: Tensor) -> tuple[Tensor, float]:
    """Symmetric per-tensor INT8 quantization (SilverTorch paper §4.2): one global scalar scale, so
    the kernel does one scalar multiply per item at the cost of coarser reconstruction than the
    per-row ``quantize_int8``.

    Returns (codes [N, D] int8, scale float); embs ≈ codes.float() * scale."""
    abs_max = embs.abs().amax().clamp(min=1e-8)
    scale = float((abs_max / 127.0).item())
    codes = (embs / abs_max * 127.0).round().clamp(-128, 127).to(torch.int8)
    return codes, scale


def _build_oporp(d: int, seed: int, device: torch.device) -> tuple[Tensor, Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    signs = torch.randint(0, 2, (d,), generator=g, device=device, dtype=torch.int8) * 2 - 1
    perm = torch.randperm(d, generator=g, device=device)
    return signs, perm


def _pack_signs_to_int64(values: Tensor) -> Tensor:
    """Pack sign-quantized [..., D] floats into [..., W] int64 bit words.

    Bit ``b`` of word ``w`` is set iff ``values[..., 64*w + b] > 0``."""
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
    k_bits: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sign-OPORP 1-bit quantization (Li et al. 2019): sign-flip, permute, bin into k_bits groups,
    L2-normalize, pack to 64-bit words. k_bits=0 resolves to D (bit-stable default).

    Returns (bits [N, k_bits//64] int64, signs [D] int8, perm [D] int64); query sim = k_bits -
    2*popcount(q ^ item)."""
    if embs.dim() != 2:
        raise ValueError(f"expected 2-D [N, D] embeddings, got shape {tuple(embs.shape)}")
    n, d = embs.shape
    if d % 64 != 0:
        raise ValueError(f"D must be a multiple of 64 for 1-bit packing, got {d}")
    if k_bits == 0:
        k_bits = d
    if d % k_bits != 0:
        raise ValueError(f"k_bits must divide D; got k_bits={k_bits}, D={d}")
    if k_bits % 64 != 0:
        raise ValueError(f"k_bits must be a multiple of 64, got {k_bits}")
    b = d // k_bits
    signs, perm = _build_oporp(d, seed, embs.device)
    proj = (embs * signs.to(embs.dtype)).index_select(1, perm)
    binned = proj.view(n, k_bits, b).sum(dim=-1)
    sketch = binned / binned.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return _pack_signs_to_int64(sketch), signs, perm


def project_oporp_1bit_query(
    query: Tensor,
    signs: Tensor,
    perm: Tensor,
    k_bits: int = 0,
) -> Tensor:
    """Apply the same Sign-OPORP projection to a query batch; ``k_bits`` must match the value passed
    to ``quantize_oporp_1bit`` (``signs``/``perm`` are ``[D]`` and don't encode it), with
    ``k_bits=0`` resolving to ``D``.

    Returns [B, k_bits//64] int64 packed sign bits."""
    if query.dim() != 2:
        raise ValueError(f"expected 2-D [B, D] query, got shape {tuple(query.shape)}")
    b_size, d = query.shape
    if k_bits == 0:
        k_bits = d
    if d % k_bits != 0:
        raise ValueError(f"k_bits must divide D; got k_bits={k_bits}, D={d}")
    if k_bits % 64 != 0:
        raise ValueError(f"k_bits must be a multiple of 64, got {k_bits}")
    bin_w = d // k_bits
    proj = (query * signs.to(query.dtype)).index_select(1, perm)
    binned = proj.view(b_size, k_bits, bin_w).sum(dim=-1)
    sketch = binned / binned.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return _pack_signs_to_int64(sketch)


def _build_simhash_R(d: int, k_bits: int, seed: int, device: torch.device) -> Tensor:
    g = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(k_bits, d, generator=g, device=device)


def quantize_simhash_1bit(
    embs: Tensor,
    k_bits: int,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """SimHash 1-bit quantization (Charikar 2002 / Manku 2007): each output bit = sign(r_j · x), r_j
    ~ N(0, I_D); unlike Sign-OPORP it admits k_bits > D since every bit mixes all coordinates.

    Returns (bits [N, k_bits//64] int64, r [k_bits, D] fp32); query sim = k_bits - 2*popcount(q ^
    item)."""
    if embs.dim() != 2:
        raise ValueError(f"expected 2-D [N, D] embeddings, got shape {tuple(embs.shape)}")
    if k_bits % 64 != 0:
        raise ValueError(f"k_bits must be a multiple of 64, got {k_bits}")
    r = _build_simhash_R(embs.shape[1], k_bits, seed, embs.device)
    return _pack_signs_to_int64(embs @ r.t()), r


def project_simhash_1bit_query(query: Tensor, r: Tensor) -> Tensor:
    """Apply the same SimHash projection to a query batch.

    Returns ``[B, k_bits // 64]`` int64 packed sign bits."""
    if query.dim() != 2:
        raise ValueError(f"expected 2-D [B, D] query, got shape {tuple(query.shape)}")
    return _pack_signs_to_int64(query @ r.t())
