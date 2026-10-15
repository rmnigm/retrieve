"""The standalone ``FilterModule``: the exact clause filter. It resolves its backend's op namespace
through ``ops_for`` (``clause_mask`` / ``clause_mask_scores`` / ``clause_compact`` exist with one
signature in ``retrieve.ops.triton`` and ``retrieve.ops.reference``)."""

from __future__ import annotations

import torch
from torch import Tensor

from retrieve.functional import clause_subset_match
from retrieve.interfaces import FilterModule, LinrBackend, check_backend, ops_for


class ExactAttributeFilter(FilterModule):
    """Exact clause-attribute filter, decoupled from retrieval (compose via ``evaluate_mask`` /
    ``evaluate_indices``). ``backend="triton"`` (default) fuses the dense/compact paths via
    ``clause_mask``/``clause_compact``; ``backend="torch"`` runs broadcast equality over a
    ``[B, N, C, A_max]`` bool intermediate."""

    item_clause_attrs: Tensor  # [N, C, A_max] int64
    clause_is_reverse: Tensor  # [C] bool

    def __init__(self, backend: LinrBackend = "triton") -> None:
        super().__init__()
        check_backend(backend, LinrBackend)
        self.backend = backend
        ops_for(backend)

    def register_index(
        self,
        item_clause_attrs: Tensor,
        *,
        clause_is_reverse: Tensor | None = None,
    ) -> None:
        c = item_clause_attrs.shape[1]
        self.register_buffer("item_clause_attrs", item_clause_attrs.long())
        if clause_is_reverse is None:
            clause_is_reverse = torch.zeros(c, dtype=torch.bool, device=item_clause_attrs.device)
        self.register_buffer("clause_is_reverse", clause_is_reverse)

    def evaluate_mask(self, q: Tensor, start: int = 0, end: int | None = None) -> Tensor:
        """``[B, end - start]`` bool over items ``[start, end)``, computed on a row slice of the
        attribute table (a contiguous view), so no ``[B, N]`` exists."""
        return ops_for(self.backend).clause_mask(
            self.item_clause_attrs[start:end], self.clause_is_reverse, q
        )

    def mask_scores(self, scores: Tensor, q: Tensor) -> Tensor:
        """``scores`` [B, N] with the failing items at ``-inf``, one fused pass on triton."""
        return ops_for(self.backend).clause_mask_scores(
            scores, self.item_clause_attrs, self.clause_is_reverse, q
        )

    def evaluate_indices(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """``(positive_indices [B, N] int64, counts [B] int64)``; each row's ids are in
        ascending item order on both backends."""
        return ops_for(self.backend).clause_compact(
            self.item_clause_attrs, self.clause_is_reverse, q
        )

    def evaluate_subset(self, q: Tensor, candidate_ids: Tensor) -> Tensor:
        """The filter over ``candidate_ids: [B, P]`` only (gather + broadcast equality, no full-N
        scan); reverse-clause and inactive-query (-1) semantics as ``evaluate_mask``."""
        gathered = self.item_clause_attrs[candidate_ids]  # [B, P, C, A_max]
        return clause_subset_match(gathered, q, self.clause_is_reverse)
