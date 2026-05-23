from __future__ import annotations

import torch
from torch import Tensor, nn

from retrieve.interfaces import Backend


class PostfilterKNN(nn.Module):
    """Pure-torch dense scoring + boolean mask + top-K.

    Computes the full ``query @ item_embs.T`` similarity matrix, applies an
    optional boolean mask via ``masked_fill(-inf)``, and selects the top-K.

    **Precision.** ``item_embs`` and ``query`` may be fp32 or fp16; both are
    cast to fp16 internally (storage is fp16). The fp16 matmul on Ampere+
    GPUs uses tensor cores with a fp32 accumulator, so dot-product numerics
    are equivalent to a TF32 matmul on normalized embeddings — see the
    layer-package docstring for the full convention.

    The ``backend=`` flag is accepted for API symmetry with the other
    retrieval modules but has no effect here: the original
    ``fused_matmul_topk`` Triton kernel was removed because cuBLAS + CUB
    already deliver the same memory traffic and selection cost. Both
    ``backend="torch"`` and ``backend="triton"`` run this code.
    See ``docs/system/kernels.md`` for the historical rationale.
    """

    item_embs_t: Tensor

    def __init__(self, k: int, backend: Backend = "triton") -> None:
        super().__init__()
        self.k = k
        self.backend = backend

    def register_index(self, item_embs: Tensor) -> None:
        # Cast to fp16 at the API boundary (paper-faithful storage), then
        # pre-transpose to a contiguous D×N buffer so cuBLAS sees the same
        # operand layout as the brute-force oracle (`item_embs.t().contiguous()`).
        # A `.t()` view at call time dispatches a different kernel whose
        # accumulator order can flip K-th-place tiebreaks at the noise floor.
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
