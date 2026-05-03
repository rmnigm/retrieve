from __future__ import annotations

import torch
from torch import Tensor

from retrieve.kernels.triton.linr.fused_masked_knn_topk import fused_masked_knn_topk
from retrieve.layers.linr.v2 import LiNR_V2


class LiNR_V2_Triton(LiNR_V2):
    """LiNR V2 with the fused Triton prefilter kernel.

    Sparse path: ``fused_masked_knn_topk`` reads only the passing rows by
    indirect load, no dense ``[B, N]`` materialization. Without
    ``candidate_ids`` there's nothing to pre-filter, so the unmasked
    fallback runs the parent's pure-torch dense matmul + top-K (the same
    code as ``LiNR_V1``).
    """

    def forward(
        self,
        query: Tensor,
        candidate_ids: Tensor | None = None,
        counts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if candidate_ids is None:
            return super()._forward_full(query)

        b, p = candidate_ids.shape
        if p == 0:
            device = query.device
            return (
                torch.full((b, self.k), -1, dtype=torch.long, device=device),
                torch.full((b, self.k), float("-inf"), device=device),
            )
        if counts is None:
            counts = torch.full((b,), p, dtype=torch.long, device=query.device)
        return fused_masked_knn_topk(query, self.item_embs, candidate_ids, counts, self.k)


def build_linr_v2_triton(item_embs: Tensor, k: int) -> LiNR_V2_Triton:
    module = LiNR_V2_Triton(k=k)
    module.register_index(item_embs)
    return module
