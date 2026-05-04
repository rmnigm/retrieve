from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import FilterModule


class BloomFilter(FilterModule):
    """Per-item Bloom-signature attribute filter (paper-strict, conjunctive).

    Items: ``[N, C, A_max]`` int64 with ``-1`` padding. Each item's signature is
    the OR of ``k_hash`` hash positions per non-pad attribute, packed into
    ``W = m_bits // 64`` int64 words. Query: ``[B, C]`` int64 (single attribute
    per clause; ``-1`` is inactive). Subset test ``(qb & sigs) == qb`` per word,
    AND-reduced.

    No reverse / NOT — that path stays in ``ExactAttributeFilter``.
    """

    bloom_sigs: Tensor  # [N, W] int64
    hash_seeds: Tensor  # [k_hash, 2] int64

    def __init__(self, m_bits: int, k_hash: int) -> None:
        super().__init__()
        if m_bits <= 0 or (m_bits & (m_bits - 1)) != 0:
            raise ValueError(f"m_bits must be a positive power of 2, got {m_bits}")
        if m_bits % 64 != 0:
            raise ValueError(f"m_bits must be a multiple of 64, got {m_bits}")
        if k_hash <= 0:
            raise ValueError(f"k_hash must be positive, got {k_hash}")
        self.m_bits = m_bits
        self.k_hash = k_hash
        self.word_count = m_bits // 64

    def register_index(
        self,
        item_clause_attrs: Tensor,
        item_embs: Tensor | None = None,
        clause_is_reverse: Tensor | None = None,
    ) -> None:
        if clause_is_reverse is not None and bool(clause_is_reverse.any().item()):
            raise ValueError(
                "BloomFilter is paper-strict: clause_is_reverse must be all-False "
                "(no NOT). Use ExactAttributeFilter (filter_kind='clause') for reverse clauses."
            )
        seeds = _generate_seeds(self.k_hash, device=item_clause_attrs.device)
        self.register_buffer("hash_seeds", seeds)
        sigs = _build_signatures(
            item_clause_attrs.long(),
            seeds,
            self.m_bits,
            self.k_hash,
            self.word_count,
        )
        self.register_buffer("bloom_sigs", sigs)

    def _build_query_sigs(self, query_clause_attrs: Tensor) -> Tensor:
        return _build_signatures(
            query_clause_attrs.long().unsqueeze(-1),
            self.hash_seeds,
            self.m_bits,
            self.k_hash,
            self.word_count,
        )

    def evaluate_mask(self, query_clause_attrs: Tensor) -> Tensor:
        qb = self._build_query_sigs(query_clause_attrs)  # [B, W]
        if qb.is_cuda:
            from retrieve.kernels.triton.silvertorch.bloom_match import bloom_match

            return bloom_match(qb, self.bloom_sigs)
        match = (qb.unsqueeze(1) & self.bloom_sigs.unsqueeze(0)) == qb.unsqueeze(1)
        return match.all(dim=-1)

    def evaluate_indices(self, query_clause_attrs: Tensor) -> tuple[Tensor, Tensor]:
        """Returns ``(positive_indices [B, P] int64, counts [B] int64)``.

        On CUDA: routes to the fused ``bloom_compact`` Triton kernel — no
        ``[B, N]`` bool intermediate ever materialized. On CPU: ABC default
        (``compact_mask(self.evaluate_mask(qa))``). Output id order within a
        row is unspecified (atomics) — callers that care must sort.
        """
        qb = self._build_query_sigs(query_clause_attrs)  # [B, W]
        if qb.is_cuda:
            from retrieve.kernels.triton.filters.bloom_compact import bloom_compact

            return bloom_compact(qb, self.bloom_sigs)
        from retrieve.layers.utils.compact import compact_mask

        return compact_mask(self.evaluate_mask(query_clause_attrs))

    def evaluate_subset(
        self,
        query_clause_attrs: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor:
        qb = self._build_query_sigs(query_clause_attrs)  # [B, W]
        sigs = self.bloom_sigs[candidate_ids]  # [B, P, W]
        match = (qb.unsqueeze(1) & sigs) == qb.unsqueeze(1)  # [B, P, W]
        return match.all(dim=-1)


def _generate_seeds(k_hash: int, device: torch.device) -> Tensor:
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


_BUILD_SIGS_BATCH = 131072  # rows per chunk; bounds peak alloc to ~B*word_count*64*8 bytes


def _build_signatures(
    attrs: Tensor,
    seeds: Tensor,
    m_bits: int,
    k_hash: int,
    word_count: int,
) -> Tensor:
    leading = attrs.shape[:-2]
    n = 1
    for d in leading:
        n *= d
    c_dim = attrs.shape[-2]
    a_max = attrs.shape[-1]

    flat = attrs.reshape(n, c_dim * a_max)
    seed_c1 = seeds[:, 0].view(1, 1, k_hash)
    seed_c2 = seeds[:, 1].view(1, 1, k_hash)
    shifts = torch.arange(64, dtype=torch.int64, device=attrs.device)

    # The dense path materializes ``[N, word_count, 64]`` int64 (~22 GiB at
    # N=2.7M, m_bits=1024). Process in chunks: per-batch peak is bounded by
    # ``batch_size * c * a * k_hash`` int64 + ``batch_size * word_count * 64``
    # int64. At batch=131072 that's ~1 GiB scratch — fits comfortably even
    # alongside the loaded item embeddings + ExactAttributeFilter on a 40 GiB GPU.
    out = torch.empty(n, word_count, dtype=torch.int64, device=attrs.device)
    for s in range(0, n, _BUILD_SIGS_BATCH):
        e = min(s + _BUILD_SIGS_BATCH, n)
        b = e - s
        batch = flat[s:e]
        valid = batch != -1
        h = _mix64(batch.unsqueeze(-1) + seed_c1, seed_c1, seed_c2)
        positions = h & (m_bits - 1)
        flat_pos = positions.reshape(b, c_dim * a_max * k_hash)
        valid_expanded = valid.unsqueeze(-1).expand(-1, -1, k_hash).reshape(b, -1)
        safe_pos = torch.where(valid_expanded, flat_pos, torch.full_like(flat_pos, m_bits))
        bit_grid = torch.zeros(b, m_bits + 1, dtype=torch.bool, device=attrs.device)
        bit_grid.scatter_(1, safe_pos, torch.ones_like(safe_pos, dtype=torch.bool))
        bit_grid = bit_grid[:, :m_bits].view(b, word_count, 64).long()
        out[s:e] = (bit_grid << shifts).sum(dim=-1)

    out_shape = list(leading) + [word_count]
    return out.reshape(out_shape) if leading else out.reshape(word_count)
