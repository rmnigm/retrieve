from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import Backend, FilterModule
from retrieve.kernels.filters.clause_compact import clause_compact
from retrieve.kernels.filters.clause_mask import clause_mask
from retrieve.layers.utils.compact import compact_mask


class ExactAttributeFilter(FilterModule):
    """Standalone exact clause-attribute filter, decoupled from retrieval (compose via
    ``evaluate_mask`` / ``evaluate_indices``). ``backend="triton"`` (default) fuses the
    dense/compact paths via ``clause_mask``/``clause_compact``; ``backend="torch"`` runs
    broadcast equality over a ``[B, N, C, A_max]`` bool intermediate."""

    item_clause_attrs: Tensor  # [N, C, A_max] int64
    clause_is_reverse: Tensor  # [C] bool

    def __init__(self, backend: Backend = "triton") -> None:
        super().__init__()
        self.backend = backend

    def register_index(
        self,
        item_clause_attrs: Tensor,
        clause_is_reverse: Tensor | None = None,
        item_embs: Tensor | None = None,
    ) -> None:
        c = item_clause_attrs.shape[1]
        self.register_buffer("item_clause_attrs", item_clause_attrs)
        if clause_is_reverse is None:
            clause_is_reverse = torch.zeros(c, dtype=torch.bool, device=item_clause_attrs.device)
        self.register_buffer("clause_is_reverse", clause_is_reverse)

    def evaluate_mask(self, query_clause_attrs: Tensor) -> Tensor:
        """Returns [B, N] bool. ``backend="triton"`` uses the fused ``clause_mask`` kernel;
        ``backend="torch"`` materializes the full ``[B, N, C, A_max]`` bool grid."""
        if self.backend == "triton":
            return clause_mask(
                self.item_clause_attrs,
                self.clause_is_reverse,
                query_clause_attrs,
            )
        q = query_clause_attrs.unsqueeze(1).unsqueeze(-1)
        ic = self.item_clause_attrs.unsqueeze(0)
        match = q == ic
        clause_pass = match.any(dim=-1)
        rev = self.clause_is_reverse.unsqueeze(0).unsqueeze(0)
        clause_pass = torch.where(rev, ~clause_pass, clause_pass)
        inactive = (query_clause_attrs == -1).unsqueeze(1)
        clause_pass = clause_pass | inactive
        return clause_pass.all(dim=-1)

    def evaluate_indices(self, query_clause_attrs: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (positive_indices [B, P] int64, counts [B] int64); within-row id order is
        unspecified (triton atomics), so callers that care must sort."""
        if self.backend == "triton":
            return clause_compact(
                self.item_clause_attrs,
                self.clause_is_reverse,
                query_clause_attrs,
            )
        return compact_mask(self.evaluate_mask(query_clause_attrs))

    def evaluate_subset(
        self,
        query_clause_attrs: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor:
        """Apply this filter only to ``candidate_ids: [B, P]`` (gather + broadcast equality, no
        full-N scan); reverse-clause and inactive-query (-1) semantics match ``evaluate_mask``."""
        gathered = self.item_clause_attrs[candidate_ids]  # [B, P, C, A_max]
        q = query_clause_attrs.unsqueeze(1).unsqueeze(-1)  # [B, 1, C, 1]
        match = gathered == q  # [B, P, C, A_max]
        clause_pass = match.any(dim=-1)  # [B, P, C]
        rev = self.clause_is_reverse.unsqueeze(0).unsqueeze(0)  # [1, 1, C]
        clause_pass = torch.where(rev, ~clause_pass, clause_pass)
        inactive = (query_clause_attrs == -1).unsqueeze(1)  # [B, 1, C]
        clause_pass = clause_pass | inactive
        return clause_pass.all(dim=-1)  # [B, P]
