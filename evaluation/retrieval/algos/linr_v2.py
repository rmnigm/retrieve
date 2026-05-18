"""LiNR V2 + compact filter primitive — exact filtered top-K, sparse path.

The candidate source IS the filter: ``filter_mod.evaluate_indices(qa)``
returns ``(ids [B, P], counts [B])`` with no ``[B, N]`` mask
materialized. ``PrefilterKNN(backend="triton")`` then rescores those P
candidates at fp32. ``filter_mod`` is required (no unfiltered mode) —
that's the whole point of the compact path.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from retrieve import PrefilterKNN
from retrieve.interfaces import FilterModule


class LinrV2Algo:
    is_cpu = False

    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        filter_mod: FilterModule,
    ) -> None:
        self.idx = PrefilterKNN(k=k, backend="triton").to(item_embs.device)
        self.idx.register_index(item_embs)
        self.filter_mod = filter_mod
        self.modules: list[nn.Module] = [self.idx, filter_mod]

    def forward(self, q: Tensor, qa_narrow: Tensor) -> tuple[Tensor, Tensor]:
        cand, counts = self.filter_mod.evaluate_indices(qa_narrow)
        return self.idx(q, candidate_ids=cand, counts=counts)
