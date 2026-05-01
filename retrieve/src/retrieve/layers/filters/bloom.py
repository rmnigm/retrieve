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

    No reverse / NOT — that path stays in ``ClauseIndex``.
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
    ) -> None:
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
    valid = flat != -1

    seed_c1 = seeds[:, 0].view(1, 1, k_hash)
    seed_c2 = seeds[:, 1].view(1, 1, k_hash)

    h = _mix64(flat.unsqueeze(-1) + seed_c1, seed_c1, seed_c2)
    positions = h & (m_bits - 1)

    flat_pos = positions.reshape(n, c_dim * a_max * k_hash)
    valid_expanded = valid.unsqueeze(-1).expand(-1, -1, k_hash).reshape(n, -1)
    safe_pos = torch.where(valid_expanded, flat_pos, torch.full_like(flat_pos, m_bits))

    bit_grid = torch.zeros(n, m_bits + 1, dtype=torch.bool, device=attrs.device)
    src = torch.ones_like(safe_pos, dtype=torch.bool)
    bit_grid.scatter_(1, safe_pos, src)
    bit_grid = bit_grid[:, :m_bits]

    bit_grid = bit_grid.view(n, word_count, 64).long()
    shifts = torch.arange(64, dtype=torch.int64, device=attrs.device)
    sigs = (bit_grid << shifts).sum(dim=-1)

    out_shape = list(leading) + [word_count]
    return sigs.reshape(out_shape) if leading else sigs.reshape(word_count)
