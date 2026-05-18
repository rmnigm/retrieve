"""LiNR V1 — dense matmul + optional boolean mask, then topk.

Covers two registered names: ``linr_v1_filter_mask`` (canonical for
filter cells) and ``triton_knn`` (historic alias used by yambda
configs). Same implementation; the alias exists so both YAML lineages
keep working without a rename. Backed by ``SimilarityMasking`` with
``backend="triton"``.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from retrieve import SimilarityMasking
from retrieve.interfaces import FilterModule

from .filter import make_mask


class LinrV1Algo:
    is_cpu = False

    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        filter_mod: FilterModule | None = None,
    ) -> None:
        self.idx = SimilarityMasking(k=k, backend="triton").to(item_embs.device)
        self.idx.register_index(item_embs)
        self.filter_mod = filter_mod
        self.modules: list[nn.Module] = [self.idx]
        if filter_mod is not None:
            self.modules.append(filter_mod)

    def forward(self, q: Tensor, qa_narrow: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.idx(q, mask=make_mask(self.filter_mod, qa_narrow))
