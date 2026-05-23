"""LiNR V4 — int8 dense matmul + optional boolean mask, then topk.

Single-stage twin of ``linr_v1_filter_mask`` with int8 storage and int8
compute. Items and queries are int8-quantized symmetrically per row;
scoring is cuBLAS ``_int_mm`` (int8×int8 → int32, IMMA tensor cores on
Ampere+) followed by a per-row scale recovery to fp32. No quantized
prefilter cascade — int8 directly preserves cosine signal well enough
to be the final score (≥0.99 recall on unit-norm embeddings at D=128).

Backed by :class:`PostfilterKNNInt8`. ``backend`` is accepted for
API symmetry but has no effect: cuBLAS LtGemm runs the same code on
both paths.

The whole algo forward (filter mask build + index call) is wrapped
with ``torch.compile(dynamic=True, mode="reduce-overhead")`` in
``__init__`` regardless of backend — same pattern as ``linr_v1``.
"""

from __future__ import annotations

from torch import Tensor, nn

from retrieve.layers.linr.postfilter_knn_int8 import PostfilterKNNInt8
from retrieve.interfaces import Backend, FilterModule

from ._helpers import collect_modules
from .filter import make_mask


class LinrV4Algo(nn.Module):
    is_cpu = False

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
        self.filter_mod = filter_mod
        self.algo_modules = collect_modules(self.idx, filter_mod=filter_mod)
        self.compile(dynamic=True, mode="reduce-overhead")

    def forward(self, q: Tensor, qa_narrow: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.idx(q, mask=make_mask(self.filter_mod, qa_narrow))
