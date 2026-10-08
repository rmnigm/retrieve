from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def quantize_int8(embs: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric per-item INT8 quantization.

    ``embs ≈ codes.float() * scales.unsqueeze(1)``."""
    abs_max = embs.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = (abs_max / 127.0).squeeze(1)
    # Inductor's fp32 `/` is approximate; a correctly rounded fp64 quotient rounded to fp32 is
    # eager's exact fp32 quotient, so compiled codes equal eager's (kernels.md § quantize_int8).
    codes = ((embs.double() / abs_max).float() * 127.0).round().clamp(-128, 127).to(torch.int8)
    return codes, scales


_CODE_CHUNK_ROWS = 1 << 16  # build-time quantization chunk: 256 MiB of fp32 at D = 1024


def _abs_max(embs: Tensor) -> Tensor:
    """``embs.abs().amax()`` without the ``[N, D]`` ``abs`` temporary: ``max(amax, -amin)`` is
    the same value bit for bit (negation is exact)."""
    return torch.maximum(embs.amax(), -embs.amin()).clamp(min=1e-8)


def _codes(embs: Tensor, abs_max: Tensor) -> Tensor:
    return (embs / abs_max * 127.0).round().clamp(-128, 127).to(torch.int8)


def quantize_int8_global(embs: Tensor, rows: Tensor | None = None) -> tuple[Tensor, float]:
    """Symmetric per-tensor INT8 quantization (SilverTorch paper §4.2): one global scalar scale, so
    the kernel does one scalar multiply per item at the cost of coarser reconstruction than the
    per-row ``quantize_int8``.

    Build-time: the codes are written chunk by chunk into the ``[N, D]`` int8 output, so the
    transient is one chunk rather than two fp32 copies of the table (a 10M × 768 fp32 index is
    28.6 GiB; two more copies do not fit on an 80 GB card).
    ``rows`` (a permutation, e.g. SilverTorch's ``sort_perm``) writes the codes of
    ``embs[rows]`` — the same codes, reordered, without a second int8 table.

    Returns (codes [N, D] int8, scale float); embs ≈ codes.float() * scale."""
    abs_max = _abs_max(embs)
    codes = torch.empty(embs.shape, dtype=torch.int8, device=embs.device)
    for start in range(0, embs.shape[0], _CODE_CHUNK_ROWS):
        stop = start + _CODE_CHUNK_ROWS
        chunk = embs[start:stop] if rows is None else embs[rows[start:stop]]
        codes[start:stop] = _codes(chunk, abs_max)
    return codes, float((abs_max / 127.0).item())


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
    w = d // 64
    bits = (values > 0).to(torch.int64)
    bits = bits.reshape(*prefix, w, 64)
    shifts = torch.arange(64, device=values.device, dtype=torch.int64)
    return (bits << shifts).sum(dim=-1)


def _pack_chunked(embs: Tensor, w: int, project: Callable[[Tensor], Tensor]) -> Tensor:
    """``project(embs)`` ([N, D] → [N, W] int64 bit words) written chunk by chunk into the output,
    as ``quantize_int8_global`` does its codes: every step of both 1-bit projections is
    row-local, so the transient is one chunk's instead of several full-corpus copies."""
    bits = torch.empty((embs.shape[0], w), dtype=torch.int64, device=embs.device)
    for start in range(0, embs.shape[0], _CODE_CHUNK_ROWS):
        stop = start + _CODE_CHUNK_ROWS
        bits[start:stop] = project(embs[start:stop])
    return bits


def _oporp_k_bits(d: int, k_bits: int) -> int:
    """The shared ``k_bits`` validation (``k_bits=0`` resolves to ``D``)."""
    if k_bits == 0:
        k_bits = d
    if d % k_bits != 0:
        raise ValueError(f"k_bits must divide D; got k_bits={k_bits}, D={d}")
    if k_bits % 64 != 0:
        raise ValueError(f"k_bits (D when 0) must be a multiple of 64, got {k_bits}")
    return k_bits


def _oporp_project(x: Tensor, signs: Tensor, perm: Tensor, k_bits: int) -> Tensor:
    """Shared Sign-OPORP chain (index- and query-side): sign-flip → permute → bin into ``k_bits``
    groups → L2-normalize → pack to 64-bit words."""
    b_or_n, d = x.shape
    k_bits = _oporp_k_bits(d, k_bits)
    bin_w = d // k_bits
    proj = (x * signs.to(x.dtype)).index_select(1, perm)
    binned = proj.view(b_or_n, k_bits, bin_w).sum(dim=-1)
    sketch = binned / binned.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return _pack_signs_to_int64(sketch)


def quantize_oporp_1bit(
    embs: Tensor,
    seed: int = 0,
    k_bits: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sign-OPORP 1-bit quantization (Li et al. 2019): sign-flip, permute, bin into k_bits groups,
    L2-normalize, pack to 64-bit words. k_bits=0 resolves to D (bit-stable default). Build-time,
    in row chunks (``_pack_chunked``): bit-identical to ``project_oporp_1bit_query`` on the table.

    Returns (bits [N, k_bits//64] int64, signs [D] int8, perm [D] int64); query sim = k_bits -
    2*popcount(q ^ item)."""
    if embs.dim() != 2:
        raise ValueError(f"expected 2-D [N, D] embeddings, got shape {tuple(embs.shape)}")
    k_bits = _oporp_k_bits(embs.shape[1], k_bits)
    signs, perm = _build_oporp(embs.shape[1], seed, embs.device)
    bits = _pack_chunked(embs, k_bits // 64, lambda x: _oporp_project(x, signs, perm, k_bits))
    return bits, signs, perm


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
    return _oporp_project(query, signs, perm, k_bits)


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
    Build-time, in row chunks (``_pack_chunked``).

    Returns (bits [N, k_bits//64] int64, r [k_bits, D] fp32); query sim = k_bits - 2*popcount(q ^
    item)."""
    if embs.dim() != 2:
        raise ValueError(f"expected 2-D [N, D] embeddings, got shape {tuple(embs.shape)}")
    if k_bits % 64 != 0:
        raise ValueError(f"k_bits must be a multiple of 64, got {k_bits}")
    r = _build_simhash_R(embs.shape[1], k_bits, seed, embs.device)
    return _pack_chunked(embs, k_bits // 64, lambda x: _pack_signs_to_int64(x @ r.t())), r


def project_simhash_1bit_query(query: Tensor, r: Tensor) -> Tensor:
    """Apply the same SimHash projection to a query batch.

    Returns ``[B, k_bits // 64]`` int64 packed sign bits."""
    if query.dim() != 2:
        raise ValueError(f"expected 2-D [B, D] query, got shape {tuple(query.shape)}")
    return _pack_signs_to_int64(query @ r.t())
