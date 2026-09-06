from __future__ import annotations

from torch import Tensor

from retrieve.interfaces import Backend, FilterModule
from retrieve.kernels.filters.bloom_compact import bloom_compact
from retrieve.kernels.silvertorch.bloom_match import bloom_match
from retrieve.layers.filters.bloom_hash import (
    bloom_subset_match,
    build_query_signatures,
    build_signatures,
    generate_clause_salt,
    generate_seeds,
)
from retrieve.layers.utils.compact import compact_mask


class BloomFilter(FilterModule):
    """Per-item Bloom-signature attribute filter (paper-strict, conjunctive). Items ``[N, C,
    A_max]`` int64 (``-1`` pad) become per-item signatures of ``W = m_bits // 64`` words; a query
    ``[B, C]`` int64 (``-1`` inactive) matches by the subset test ``(qb & sigs) == qb`` per word,
    AND-reduced. No reverse / NOT — that stays in ``ExactAttributeFilter``. ``backend="triton"``
    (default) fuses via ``bloom_match``/``bloom_compact``; ``backend="torch"`` runs the same test
    eager."""

    bloom_sigs: Tensor  # [N, W] int64
    hash_seeds: Tensor  # [k_hash, 2] int64
    clause_salt: Tensor  # [C] int64

    def __init__(self, m_bits: int, k_hash: int, backend: Backend = "triton") -> None:
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
        self.backend = backend

    def register_index(
        self,
        item_clause_attrs: Tensor,
        *,
        clause_is_reverse: Tensor | None = None,
    ) -> None:
        if clause_is_reverse is not None and bool(clause_is_reverse.any().item()):
            raise ValueError(
                "BloomFilter is paper-strict: clause_is_reverse must be all-False "
                "(no NOT). Use ExactAttributeFilter (filter_kind='clause') for reverse clauses."
            )
        device = item_clause_attrs.device
        seeds = generate_seeds(self.k_hash, device=device)
        self.register_buffer("hash_seeds", seeds)
        # Per-clause salt registered once so the per-forward query build issues no
        # host→device copy (bloom_hash.generate_clause_salt).
        salt = generate_clause_salt(item_clause_attrs.shape[1], device=device)
        self.register_buffer("clause_salt", salt)
        sigs = build_signatures(
            item_clause_attrs.long(),
            seeds,
            self.m_bits,
            self.k_hash,
            self.word_count,
            clause_salt=salt,
        )
        self.register_buffer("bloom_sigs", sigs)

    def _build_query_sigs(self, query_clause_attrs: Tensor) -> Tensor:
        return build_query_signatures(
            query_clause_attrs.long().unsqueeze(-1),
            self.hash_seeds,
            self.m_bits,
            self.k_hash,
            self.word_count,
            clause_salt=self.clause_salt,
        )

    def evaluate_mask(self, query_clause_attrs: Tensor) -> Tensor:
        qb = self._build_query_sigs(query_clause_attrs)  # [B, W]
        if self.backend == "triton":
            return bloom_match(qb, self.bloom_sigs)
        match = (qb.unsqueeze(1) & self.bloom_sigs.unsqueeze(0)) == qb.unsqueeze(1)
        return match.all(dim=-1)

    def evaluate_indices(self, query_clause_attrs: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (positive_indices [B, P] int64, counts [B] int64); within-row id order is
        unspecified (triton atomics), so callers that care must sort."""
        qb = self._build_query_sigs(query_clause_attrs)  # [B, W]
        if self.backend == "triton":
            return bloom_compact(qb, self.bloom_sigs)
        return compact_mask(self.evaluate_mask(query_clause_attrs))

    def evaluate_subset(
        self,
        query_clause_attrs: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor:
        qb = self._build_query_sigs(query_clause_attrs)  # [B, W]
        sigs = self.bloom_sigs[candidate_ids]  # [B, P, W]
        return bloom_subset_match(qb, sigs)
