from __future__ import annotations

import abc
from typing import Literal

from torch import Tensor, nn

Backend = Literal["torch", "triton"]


class FilterModule(nn.Module, abc.ABC):
    """Boolean predicate over a registered item index; ``forward`` aliases ``evaluate_mask``.

    Three eval paths: ``evaluate_mask(q) -> [B, N] bool`` (dense), ``evaluate_indices(q) -> ([B,
    P] int64, [B] int64)`` (compact candidates), ``evaluate_subset(q, candidate_ids) -> [B, P]
    bool`` (check only the given ids)."""

    @abc.abstractmethod
    def register_index(
        self,
        item_clause_attrs: Tensor,
        item_embs: Tensor | None = None,
    ) -> None: ...

    @abc.abstractmethod
    def evaluate_mask(self, query_clause_attrs: Tensor) -> Tensor: ...

    def evaluate_indices(
        self,
        query_clause_attrs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        from retrieve.layers.utils.compact import compact_mask

        return compact_mask(self.evaluate_mask(query_clause_attrs))

    def evaluate_subset(
        self,
        query_clause_attrs: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor:
        mask = self.evaluate_mask(query_clause_attrs)
        return mask.gather(1, candidate_ids)

    def forward(self, query_clause_attrs: Tensor) -> Tensor:
        return self.evaluate_mask(query_clause_attrs)
