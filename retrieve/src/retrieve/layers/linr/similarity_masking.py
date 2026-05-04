from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import RetrievalModule


class SimilarityMasking(RetrievalModule):
    """Pure-torch dense scoring + boolean mask + top-K.

    Computes the full ``query @ item_embs.T`` similarity matrix, applies an
    optional boolean mask via ``masked_fill(-inf)``, and selects the top-K.
    ``SimilarityMaskingTriton`` exists as a backend-dispatch alias but runs this
    same code — dense matmul + top-K has no real fusion benefit over
    cuBLAS + CUB. See docs/system/architecture.md.
    """

    item_embs_t: Tensor

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def register_index(self, item_embs: Tensor) -> None:
        # Pre-transpose to a contiguous D×N buffer so cuBLAS sees the same
        # operand layout as the brute-force oracle (`item_embs.t().contiguous()`).
        # A `.t()` view at call time dispatches a different kernel whose
        # accumulator order can flip K-th-place tiebreaks at the noise floor.
        self.register_buffer("item_embs_t", item_embs.t().contiguous())

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        scores = query @ self.item_embs_t
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        topk_scores, topk_ids = torch.topk(scores, self.k, dim=1)
        if mask is not None:
            topk_ids = torch.where(
                torch.isfinite(topk_scores),
                topk_ids,
                topk_ids.new_full((), -1),
            )
        return topk_ids, topk_scores
