from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import LinrBackend, RetrievalModule, check_backend
from retrieve.layers.utils.topk import masked_topk


class PostfilterKNN(RetrievalModule):
    """Pure-torch dense scoring (``query @ item_embs.T``) + optional boolean mask + top-K; inputs
    are cast to fp16 (storage fp16, fp32 accumulate). The ``backend=`` flag is accepted for API
    symmetry but has no effect — cuBLAS + CUB already match a fused kernel here."""

    item_embs_t: Tensor

    def __init__(self, k: int, backend: LinrBackend = "triton") -> None:
        super().__init__()
        check_backend(backend, LinrBackend)
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
            return masked_topk(scores, self.k, valid=mask)
        return masked_topk(scores, self.k)
