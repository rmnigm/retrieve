from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import RetrievalModule


class LiNR_V1(RetrievalModule):
    """LiNR V1, pure-torch reference.

    Computes the full ``query @ item_embs.T`` similarity matrix, applies an
    optional boolean mask via ``masked_fill(-inf)``, and selects the top-K.
    The Triton-fused equivalent lives in ``linr_v1_triton.LiNR_V1_Triton``.
    See docs/system/architecture.md (formerly docs/ARCHITECTURE.md).
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


def build_linr_v1(item_embs: Tensor, k: int) -> LiNR_V1:
    module = LiNR_V1(k=k)
    module.register_index(item_embs)
    return module
