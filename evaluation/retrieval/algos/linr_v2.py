"""LiNR V2 — exact filtered top-K over the sparse path.

The candidate source IS the filter: ``filter_mod.evaluate_indices(qa)`` returns
``(ids [B, P], counts [B])`` with no ``[B, N]`` mask materialized, and
``PrefilterKNN`` rescores those P candidates at fp32. ``filter_mod`` is therefore
required — an unfiltered mode would defeat the point of the compact path.
"""

from __future__ import annotations

from torch import Tensor

from retrieve import PrefilterKNN
from retrieve.interfaces import Backend, FilterModule

from ._helpers import AlgoBase


class LinrV2Algo(AlgoBase):
    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        filter_mod: FilterModule,
        backend: Backend = "triton",
    ) -> None:
        super().__init__()
        # PrefilterKNN handles the fp32→fp16 cast; see retrieve.layers.linr
        # package docstring.
        self.idx = PrefilterKNN(k=k, backend=backend).to(item_embs.device)
        self.idx.register_index(item_embs)
        self._finalize(self.idx, filter_mod=filter_mod)

    def forward(self, q: Tensor, qa_narrow: Tensor) -> tuple[Tensor, Tensor]:
        cand, counts = self.filter_mod.evaluate_indices(qa_narrow)
        return self.idx(q, candidate_ids=cand, counts=counts)
