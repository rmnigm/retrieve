from __future__ import annotations

import abc

from torch import Tensor, nn


class FilterModule(nn.Module, abc.ABC):
    """Boolean predicate over an item index.

    Concrete filters expose three native paths over a registered item set,
    keyed by the consumer's preferred shape:

    - ``evaluate_mask(q) -> [B, N] bool`` — dense; consumed by V1 / V3 mask path,
      and by ``combine_masks`` for AND-of-masks composition.
    - ``evaluate_indices(q) -> ([B, P] int64, [B] int64)`` — compact; consumed by
      V2 / V3 candidate path. Default falls back to ``compact_mask(evaluate_mask)``;
      override when a fused compact kernel exists.
    - ``evaluate_subset(q, candidate_ids) -> [B, P] bool`` — apply this filter
      only to the given candidate ids. Default gathers columns of
      ``evaluate_mask``; override when a per-row check is much cheaper than full
      evaluation (e.g. P ≪ N).

    ``forward`` is an alias to ``evaluate_mask``.
    """

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


class RetrievalModule(nn.Module, abc.ABC):
    """Produces ``(ids[B, K], scores[B, K])`` from the full item pool.

    Concrete subclasses extend ``forward`` with whichever filter representation
    they consume (e.g. ``mask`` for V1's dense path, ``candidate_ids`` for V2,
    both for V3). The base contract is just ``forward(query)``.
    """

    @abc.abstractmethod
    def register_index(self, item_embs: Tensor) -> None: ...

    @abc.abstractmethod
    def forward(self, query: Tensor) -> tuple[Tensor, Tensor]: ...


class ScorerModule(nn.Module, abc.ABC):
    """Re-scores a set of candidate items: returns ``scores[B, K]``."""

    @abc.abstractmethod
    def register_index(self, item_embs: Tensor) -> None: ...

    @abc.abstractmethod
    def forward(
        self,
        query: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor: ...
