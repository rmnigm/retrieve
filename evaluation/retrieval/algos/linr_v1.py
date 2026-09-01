"""LiNR V1 — dense matmul + optional boolean mask, then topk.

Backed by ``PostfilterKNN``. Covers two registered names:
``linr_v1_filter_mask`` (canonical for filter cells) and ``triton_knn`` (historic
alias kept so the yambda YAML lineage keeps working).
"""

from __future__ import annotations

from torch import Tensor

from retrieve import PostfilterKNN
from retrieve.interfaces import Backend, FilterModule

from ._helpers import AlgoBase
from .filter import make_mask


class LinrV1Algo(AlgoBase):
    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        filter_mod: FilterModule | None = None,
        backend: Backend = "triton",
    ) -> None:
        super().__init__()
        # PostfilterKNN handles the fp32→fp16 cast for paper-faithful
        # storage; see retrieve.layers.linr package docstring.
        self.idx = PostfilterKNN(k=k, backend=backend).to(item_embs.device)
        self.idx.register_index(item_embs)
        self._finalize(self.idx, filter_mod=filter_mod)

    def forward(self, q: Tensor, qa_narrow: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.idx(q, mask=make_mask(self.filter_mod, qa_narrow))
