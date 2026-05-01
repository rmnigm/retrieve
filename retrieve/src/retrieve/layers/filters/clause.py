from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import FilterModule


class ClauseIndex(FilterModule):
    """Standalone clause-attribute filter.

    Decoupled from any retrieval module — callers compose:

        ci = ClauseIndex(); ci.register_index(item_attrs)
        mask = ci.evaluate_mask(qa)           # for V1 (dense path)
        ids, cs = ci.evaluate_indices(qa)     # for V2 (sparse path)
    """

    item_clause_attrs: Tensor  # [N, C, A_max] int64
    clause_is_reverse: Tensor  # [C] bool

    def __init__(self) -> None:
        super().__init__()

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
        """Returns ``[B, N]`` bool. Pure-torch dense evaluation."""
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
        """Returns ``(positive_indices [B, P] int64, counts [B] int64)``.

        On CUDA: routes to the fused ``clause_compact`` Triton kernel — no
        ``[B, N]`` bool intermediate ever materialized. On CPU / fallback:
        ``compact_mask(self.evaluate_mask(qa))``. Output id order within a row
        is unspecified (atomics) — callers that care must sort.
        """
        if query_clause_attrs.is_cuda:
            from retrieve.kernels.triton.filters.clause_compact import (
                clause_compact,
            )

            return clause_compact(
                self.item_clause_attrs,
                self.clause_is_reverse,
                query_clause_attrs,
            )
        from retrieve.layers.utils.compact import compact_mask

        return compact_mask(self.evaluate_mask(query_clause_attrs))

    def evaluate_subset(
        self,
        query_clause_attrs: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor:
        """Apply this filter only to ``candidate_ids: [B, P]``.

        Gathers ``item_clause_attrs[candidate_ids]`` to ``[B, P, C, A_max]``
        and broadcasts equality with ``q[:, None, :, None]``. No full-N scan.
        Reverse-clause and inactive-query (-1) semantics match ``evaluate_mask``.
        """
        gathered = self.item_clause_attrs[candidate_ids]  # [B, P, C, A_max]
        q = query_clause_attrs.unsqueeze(1).unsqueeze(-1)  # [B, 1, C, 1]
        match = gathered == q  # [B, P, C, A_max]
        clause_pass = match.any(dim=-1)  # [B, P, C]
        rev = self.clause_is_reverse.unsqueeze(0).unsqueeze(0)  # [1, 1, C]
        clause_pass = torch.where(rev, ~clause_pass, clause_pass)
        inactive = (query_clause_attrs == -1).unsqueeze(1)  # [B, 1, C]
        clause_pass = clause_pass | inactive
        return clause_pass.all(dim=-1)  # [B, P]
