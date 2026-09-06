"""Bloom signature hashing — the single home of the hash math.

``generate_seeds`` / ``generate_clause_salt`` / ``build_signatures`` /
``build_query_signatures`` are the public API consumed by
:class:`~retrieve.layers.filters.bloom.BloomFilter` and ``SilverTorch``'s fused
bloom mode. The hash is pinned: persisted ``bloom_sigs`` buffers must stay
bit-valid across refactors, so any change here invalidates every stored index.
"""

from __future__ import annotations

import torch
from torch import Tensor

# splitmix64 finalizer constants, expressed as signed int64 (torch has no uint64);
# named once — the per-clause salt derivation below is their only consumer.
_SALT_C1 = 0x9E3779B97F4A7C15 - (1 << 64)
_SALT_C2 = 0xBF58476D1CE4E5B9 - (1 << 64)

_BUILD_SIGS_BATCH = 131072  # rows per chunk; bounds peak alloc to ~B*word_count*64*8 bytes


def generate_seeds(k_hash: int, device: torch.device) -> Tensor:
    """``[k_hash, 2]`` int64 odd multiplier pairs from a fixed CPU generator —
    deterministic across processes and devices, so signatures built anywhere agree."""
    g = torch.Generator(device="cpu")
    g.manual_seed(0x515C0DE)
    seeds = torch.randint(
        low=torch.iinfo(torch.int64).min,
        high=torch.iinfo(torch.int64).max,
        size=(k_hash, 2),
        dtype=torch.int64,
        generator=g,
    )
    seeds = seeds | 1  # force odd multipliers
    return seeds.to(device)


def _mix64(x: Tensor, c1: Tensor, c2: Tensor) -> Tensor:
    h = x ^ (x >> 33)
    h = h * c1
    h = h ^ (h >> 33)
    h = h * c2
    h = h ^ (h >> 33)
    return h


def generate_clause_salt(c_dim: int, device: torch.device) -> Tensor:
    """``[C]`` int64 per-clause hash salt — ``_mix64`` of the clause index under the
    splitmix64 constants. Pure function of ``c_dim`` (no seed), so it is identical on
    every device and every call; ``BloomFilter`` / ``SilverTorch`` register it once as
    the ``clause_salt`` buffer at ``register_index`` so the per-forward query build
    issues no host→device copy (the previous per-call ``torch.tensor(_SALT_C1,
    device=cuda)`` was a pageable H2D copy that broke raw CUDA-graph capture and cost
    ~0.4 ms of launch overhead per eager bloom forward)."""
    clause_ids = torch.arange(c_dim, dtype=torch.int64, device=device)
    return _mix64(
        clause_ids,
        torch.tensor(_SALT_C1, dtype=torch.int64, device=device),
        torch.tensor(_SALT_C2, dtype=torch.int64, device=device),
    )


def _expand_clause_salt(clause_salt: Tensor, c_dim: int, a_max: int) -> Tensor:
    """``[C]`` salt → the ``[1, C*A, 1]`` broadcast shape ``_signature_batch`` XORs in
    (clause ``c`` repeated over its ``A`` attribute slots). Views only — no copy, no
    host round trip."""
    if clause_salt.dim() != 1 or clause_salt.shape[0] != c_dim:
        raise ValueError(
            f"clause_salt must be a [C={c_dim}] int64 tensor, got shape {tuple(clause_salt.shape)}"
        )
    return clause_salt.view(c_dim, 1).expand(c_dim, a_max).reshape(1, c_dim * a_max, 1)


def _resolve_clause_salt(
    clause_salt: Tensor | None, c_dim: int, a_max: int, device: torch.device
) -> Tensor:
    """The registered ``[C]`` buffer if given, else a fresh one (standalone callers —
    parity tests, the tuner). Either way the bits XORed into the hash are the same."""
    if clause_salt is None:
        clause_salt = generate_clause_salt(c_dim, device)
    return _expand_clause_salt(clause_salt, c_dim, a_max)


def _signature_batch(
    flat: Tensor,
    valid: Tensor,
    seeds: Tensor,
    m_bits: int,
    k_hash: int,
    word_count: int,
    clause_salt: Tensor,
) -> Tensor:
    """Hash + salt + scatter + word-pack for one ``[b, C*A]`` slab → ``[b, W]``.

    The shared core of the two builders below. Keying is by ``(clause_idx, value)``:
    the per-clause salt namespaces the hash by feature key (paper eq. 3) — without
    it, value V in any clause collides with V in another clause, a documented
    false-positive leak (see kernels.md → bloom_match). Padding (``-1``, via
    ``valid``) is detected before keying, so the ``m_bits`` sentinel sink is
    unaffected.
    """
    b = flat.shape[0]
    seed_c1 = seeds[:, 0].view(1, 1, k_hash)
    seed_c2 = seeds[:, 1].view(1, 1, k_hash)
    shifts = torch.arange(64, dtype=torch.int64, device=flat.device)

    h = _mix64(flat.unsqueeze(-1) + seed_c1, seed_c1, seed_c2)
    h = h ^ clause_salt
    positions = h & (m_bits - 1)
    flat_pos = positions.reshape(b, -1)
    valid_expanded = valid.unsqueeze(-1).expand(-1, -1, k_hash).reshape(b, -1)
    safe_pos = torch.where(valid_expanded, flat_pos, torch.full_like(flat_pos, m_bits))
    bit_grid = torch.zeros(b, m_bits + 1, dtype=torch.bool, device=flat.device)
    bit_grid.scatter_(1, safe_pos, torch.ones_like(safe_pos, dtype=torch.bool))
    bit_grid = bit_grid[:, :m_bits].view(b, word_count, 64).long()
    return (bit_grid << shifts).sum(dim=-1)


def build_signatures(
    attrs: Tensor,
    seeds: Tensor,
    m_bits: int,
    k_hash: int,
    word_count: int,
    *,
    clause_salt: Tensor | None = None,
) -> Tensor:
    """Index-side signature build: chunked over ``_BUILD_SIGS_BATCH`` rows.

    Dense ``[N, word_count, 64]`` int64 would be ~22 GiB at N=2.7M, m_bits=1024;
    chunking bounds the per-batch peak (~1 GiB at 131072 rows) so it fits
    alongside the index. ``clause_salt`` is the ``[C]`` buffer from
    :func:`generate_clause_salt`; ``None`` derives it on the fly (same bits)."""
    leading = attrs.shape[:-2]
    n = 1
    for d in leading:
        n *= d
    c_dim = attrs.shape[-2]
    a_max = attrs.shape[-1]

    flat = attrs.reshape(n, c_dim * a_max)
    salt = _resolve_clause_salt(clause_salt, c_dim, a_max, attrs.device)

    out = torch.empty(n, word_count, dtype=torch.int64, device=attrs.device)
    for s in range(0, n, _BUILD_SIGS_BATCH):
        e = min(s + _BUILD_SIGS_BATCH, n)
        batch = flat[s:e]
        out[s:e] = _signature_batch(batch, batch != -1, seeds, m_bits, k_hash, word_count, salt)

    out_shape = list(leading) + [word_count]
    return out.reshape(out_shape) if leading else out.reshape(word_count)


def build_query_signatures(
    attrs: Tensor,
    seeds: Tensor,
    m_bits: int,
    k_hash: int,
    word_count: int,
    *,
    clause_salt: Tensor | None = None,
) -> Tensor:
    """Query-side signature build: one loop-free ``_signature_batch`` call — a
    Python ``range()`` loop would make dynamo specialize on the trip count under
    ``torch.compile(dynamic=True)``. Query batches are small, so the unchunked
    peak alloc is fine. Pass the module's ``clause_salt`` buffer so the forward
    is free of host→device copies (see :func:`generate_clause_salt`)."""
    leading = attrs.shape[:-2]
    n = 1
    for d in leading:
        n *= d
    c_dim = attrs.shape[-2]
    a_max = attrs.shape[-1]

    flat = attrs.reshape(n, c_dim * a_max)
    salt = _resolve_clause_salt(clause_salt, c_dim, a_max, attrs.device)
    out = _signature_batch(flat, flat != -1, seeds, m_bits, k_hash, word_count, salt)

    out_shape = list(leading) + [word_count]
    return out.reshape(out_shape) if leading else out.reshape(word_count)


def bloom_subset_match(qb: Tensor, sigs: Tensor) -> Tensor:
    """``[B, W]`` query sigs vs ``[B, P, W]`` gathered sigs → ``[B, P]`` bool.

    Per-word subset test ``(qb & sig) == qb``, AND-reduced over words — the
    single torch-side definition, shared by ``BloomFilter.evaluate_subset`` and
    SilverTorch's eager bloom path."""
    match = (qb.unsqueeze(1) & sigs) == qb.unsqueeze(1)  # [B, P, W]
    return match.all(dim=-1)
