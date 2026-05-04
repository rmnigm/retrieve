from __future__ import annotations

from retrieve.layers.linr.similarity_masking import SimilarityMasking


class SimilarityMaskingTriton(SimilarityMasking):
    """Backend-dispatch alias — pure torch under the hood.

    The original Triton wrapper executed a tile-parallel ``tl.dot`` and then
    ``torch.topk`` on the materialized ``[B, N]`` score buffer. That isn't
    real fusion: cuBLAS matmul + CUB top-K does the same thing with the
    same memory traffic, so the kernel was removed. The class is kept so the
    evaluation registry's ``triton_knn`` algorithm continues to work; the
    forward is inherited unchanged from ``SimilarityMasking``.
    """
