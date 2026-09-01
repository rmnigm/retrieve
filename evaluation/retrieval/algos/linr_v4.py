"""LiNR V4 — int8 dense matmul + optional boolean mask, then topk.

Single-stage twin of ``linr_v1_filter_mask`` backed by ``PostfilterKNNInt8``:
cuBLAS ``_int_mm`` (int8xint8 -> int32, IMMA on Ampere+) with per-row scale
recovery. No prefilter cascade — int8 preserves enough cosine signal to be the
final score (>=0.99 recall on unit-norm embeddings at D=128).

``backend`` is accepted for API symmetry but has no effect: cuBLAS LtGemm runs the
same code on every path.
"""

from __future__ import annotations

from torch import Tensor

from retrieve.interfaces import Backend, FilterModule
from retrieve.layers.linr.postfilter_knn_int8 import PostfilterKNNInt8

from ._helpers import AlgoBase
from .filter import make_mask


class LinrV4Algo(AlgoBase):
    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        filter_mod: FilterModule | None = None,
        backend: Backend = "triton",
    ) -> None:
        super().__init__()
        self.idx = PostfilterKNNInt8(k=k, backend=backend).to(item_embs.device)
        self.idx.register_index(item_embs)
        self._finalize(self.idx, filter_mod=filter_mod)

    def forward(self, q: Tensor, qa_narrow: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.idx(q, mask=make_mask(self.filter_mod, qa_narrow))
