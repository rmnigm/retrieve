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

    item_embs: Tensor

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def register_index(self, item_embs: Tensor) -> None:
        self.register_buffer("item_embs", item_embs)

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        scores = query @ self.item_embs.t()
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        topk_scores, topk_ids = torch.topk(scores, self.k, dim=1)
        return topk_ids, topk_scores
