from __future__ import annotations

import torch
from torch import Tensor


class BloomIndex(torch.nn.Module):
    """Per-item Bloom-signature attribute filter."""

    bloom_sigs: Tensor
    hash_seeds: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.m_bits: int = 0
        self.k_hash: int = 0
        self.word_count: int = 0

    def register_index(
        self,
        item_clause_attrs: Tensor,
        m_bits: int,
        k_hash: int,
    ) -> None:
        if m_bits <= 0 or (m_bits & (m_bits - 1)) != 0:
            raise ValueError(f"m_bits must be a positive power of 2, got {m_bits}")
        if m_bits % 64 != 0:
            raise ValueError(f"m_bits must be a multiple of 64, got {m_bits}")
        if k_hash <= 0:
            raise ValueError(f"k_hash must be positive, got {k_hash}")

        self.m_bits = m_bits
        self.k_hash = k_hash
        self.word_count = m_bits // 64

        seeds = _generate_seeds(k_hash, device=item_clause_attrs.device)
        self.register_buffer("hash_seeds", seeds)

        sigs = _build_signatures(
            item_clause_attrs.long(),
            seeds,
            self.m_bits,
            self.k_hash,
            self.word_count,
        )
        self.register_buffer("bloom_sigs", sigs)

    def evaluate(self, query_clause_attrs: Tensor) -> Tensor:
        attrs_3d = query_clause_attrs.long().unsqueeze(-1)  # [B, C, 1]
        qb = _build_signatures(
            attrs_3d,
            self.hash_seeds,
            self.m_bits,
            self.k_hash,
            self.word_count,
        )
        # qb: [B, W]; sigs: [N, W]; subset test: (qb & sig) == qb
        match = (qb.unsqueeze(1) & self.bloom_sigs.unsqueeze(0)) == qb.unsqueeze(1)
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
