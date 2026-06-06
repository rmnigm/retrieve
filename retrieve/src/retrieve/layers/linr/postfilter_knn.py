from __future__ import annotations

import torch
from torch import Tensor, nn

from retrieve.interfaces import Backend


class PostfilterKNN(nn.Module):
    """Pure-torch dense scoring (``query @ item_embs.T``) + optional boolean mask + top-K; inputs
    are cast to fp16 (storage fp16, fp32 accumulate). The ``backend=`` flag is accepted for API
    symmetry but has no effect — cuBLAS + CUB already match a fused kernel here."""

    item_embs_t: Tensor

    def __init__(self, k: int, backend: Backend = "triton") -> None:
        super().__init__()
        self.k = k
        self.backend = backend

    def register_index(self, item_embs: Tensor) -> None:
        # Pre-transpose to a contiguous D×N buffer; a .t() view at call time dispatches a different
        # kernel whose accumulator order can flip K-th-place tiebreaks at the noise floor.
        self.register_buffer("item_embs_t", item_embs.to(torch.float16).t().contiguous())

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        scores = query.to(torch.float16) @ self.item_embs_t
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
