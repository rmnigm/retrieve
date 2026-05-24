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


def quantize_int8_global(embs: Tensor) -> tuple[Tensor, float]:
    """Symmetric per-tensor INT8 quantization — one scalar scale for the index.

    The SilverTorch paper's int8 ANN scheme (§4.2): "compute global min/max
    values across all embeddings, scale them to [-128, 127], and assign
    integer representations accordingly." A single scalar lets the kernel
    apply one scalar multiply per item in the dequant epilogue, vs a
    per-item gather under per-row scales — at the cost of coarser
    reconstruction (an outlier row stretches the global scale, narrowing
    the int8 lattice for every other row).

    Returns ``(codes[N, D] int8, scale: float)`` with reconstruction
    ``embs ≈ codes.float() * scale``. For higher quality at the cost of an
    extra ``[N]`` fp32 buffer + a per-item gather in the kernel, use
    ``quantize_int8`` and store its per-row scales instead.
    """
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
    k_bits: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sign-OPORP 1-bit quantization.

    OPORP = One Permutation + One Random projection (Li et al., 2019). For each
    item ``x``, the projection is ``(signs * x)[perm]``, then binned into
    ``k_bits`` groups of ``D / k_bits`` and summed, L2-normalized, and
    sign-quantized into packed 64-bit words. Cheap: O(D) state, not O(D²).

    ``k_bits=0`` (the default) resolves to ``D`` — the operating point at
    which the bin-sum is the identity reshape and output bits equal
    ``sign((signs * x)[perm])``. At ``k_bits = D`` the L2 normalize divides
    each row by a positive scalar and ``sign(x/|x|) = sign(x)``, so the L2
    step is a no-op on the bits and the output is byte-identical to the
    pre-fix implementation. Choose ``k_bits < D`` (must divide ``D`` and be
    a multiple of 64) to follow the paper's full sketch — a smaller bit
    budget at higher distortion.

    Returns:
        bits: ``[N, k_bits // 64]`` int64 packed sign bits.
        signs: ``[D]`` int8 in {-1, +1} — Rademacher sign vector.
        perm: ``[D]`` int64 — permutation applied after the sign flip.

    Apply the same ``(signs, perm, k_bits)`` to a query via
    ``project_oporp_1bit_query`` to land in the same bit space. Similarity is
    then ``k_bits - 2 * popcount(query_bits ^ item_bits)``.
    """
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
    """Apply the same Sign-OPORP projection to a query batch.

    ``k_bits`` must match the value passed to ``quantize_oporp_1bit`` at
    register time (``signs`` and ``perm`` are both ``[D]`` and do not encode
    it). ``k_bits=0`` (the default) resolves to ``D = query.shape[1]`` — the
    bit-stable operating point.

    Returns ``[B, k_bits // 64]`` int64 packed sign bits.

    Pure tensor flow — multiply, index_select, bin-sum, L2 normalize, > 0,
    reshape, shifted sum. Called from inside ``OneBitKNN.forward``; the
    eval-side algo wrapper compiles its forward with ``mode="reduce-overhead"``,
    so this work is captured into the outer cudagraph.
    """
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
    """SimHash 1-bit quantization (Charikar 2002 / Manku 2007).

    Each output bit = ``sign(r_j · x)`` for independent ``r_j ~ N(0, I_D)``.
    Cost is one fp32 matmul plus a bit-pack. Unlike Sign-OPORP at ``k = D``,
    SimHash admits ``k_bits > D`` for recall-vs-memory trades the OPORP family
    cannot reach — each output bit mixes all input coordinates.

    Returns:
        bits: ``[N, k_bits // 64]`` int64 packed sign bits.
        r: ``[k_bits, D]`` fp32 Gaussian projection matrix.

    Apply the same ``r`` to a query via ``project_simhash_1bit_query`` to land
    in the same bit space. Similarity is then
    ``k_bits - 2 * popcount(query_bits ^ item_bits)``.
    """
    if embs.dim() != 2:
        raise ValueError(f"expected 2-D [N, D] embeddings, got shape {tuple(embs.shape)}")
    if k_bits % 64 != 0:
        raise ValueError(f"k_bits must be a multiple of 64, got {k_bits}")
    r = _build_simhash_R(embs.shape[1], k_bits, seed, embs.device)
    return _pack_signs_to_int64(embs @ r.t()), r


def project_simhash_1bit_query(query: Tensor, r: Tensor) -> Tensor:
    """Apply the same SimHash projection to a query batch.

    Returns ``[B, k_bits // 64]`` int64 packed sign bits.
    """
    if query.dim() != 2:
        raise ValueError(f"expected 2-D [B, D] query, got shape {tuple(query.shape)}")
    return _pack_signs_to_int64(query @ r.t())
