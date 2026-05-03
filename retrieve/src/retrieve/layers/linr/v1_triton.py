from __future__ import annotations

from torch import Tensor

from retrieve.layers.linr.v1 import LiNR_V1


class LiNR_V1_Triton(LiNR_V1):
    """LiNR V1 backend-dispatch alias — pure torch under the hood.

    The original Triton wrapper executed a tile-parallel ``tl.dot`` and then
    ``torch.topk`` on the materialized ``[B, N]`` score buffer. That isn't
    real fusion: cuBLAS matmul + CUB top-K does the same thing with the
    same memory traffic, so the kernel was removed. The class is kept so
    ``build_linr_index(version=1, backend="triton")`` and the evaluation
    registry's ``triton_knn`` algorithm continue to work; the forward is
    inherited unchanged from ``LiNR_V1``.
    """


def build_linr_v1_triton(item_embs: Tensor, k: int) -> LiNR_V1_Triton:
    module = LiNR_V1_Triton(k=k)
    module.register_index(item_embs)
    return module
