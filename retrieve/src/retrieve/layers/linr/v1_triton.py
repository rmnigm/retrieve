from __future__ import annotations

from torch import Tensor

from retrieve.kernels.triton.linr.fused_matmul_topk import fused_matmul_topk
from retrieve.layers.linr.v1 import LiNR_V1


class LiNR_V1_Triton(LiNR_V1):
    """LiNR V1 (similarity-masking) with the fused Triton matmul-topk kernel.

    Per the paper's V1 design, the path is always dense: full
    ``query @ item_embs.T`` with the mask (if any) folded into the score
    inline, then top-K. ``fused_matmul_topk`` is tile-parallel over (B, N)
    and runs the matmul on tensor cores via ``tl.dot``.
    """

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return fused_matmul_topk(query, self.item_embs, self.k, mask=mask)


def build_linr_v1_triton(item_embs: Tensor, k: int) -> LiNR_V1_Triton:
    module = LiNR_V1_Triton(k=k)
    module.register_index(item_embs)
    return module
