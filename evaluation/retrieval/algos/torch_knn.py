"""Dense top-K KNN baseline — exact via torch matmul + topk."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from retrieve import FullScanKNN
from retrieve.interfaces import FilterModule

from .filter import make_mask


class TorchKnnAlgo:
    is_cpu = False

    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        filter_mod: FilterModule | None = None,
    ) -> None:
        self.idx = FullScanKNN(k=k).to(item_embs.device)
        self.idx.register_index(item_embs)
        self.filter_mod = filter_mod
        self.modules: list[nn.Module] = [self.idx]
        if filter_mod is not None:
            self.modules.append(filter_mod)

    def forward(self, q: Tensor, qa_narrow: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.idx(q, mask=make_mask(self.filter_mod, qa_narrow))
