"""LiNR V3 → V2 cascade: 1-bit Hamming pre-filter, fp32 rerank.

Stage 1 (LiNR §4.3, Fig knn-v3): ``OneBitKNNTriton`` produces a
top-``candidate_pool`` list at 1-bit precision. Stage 2:
``PrefilterKNNTriton`` rescores those candidates at full precision.
The dense ``SimilarityMasking`` path is intentionally absent as stage-2 —
its full matmul does the same work as ``triton_knn`` alone, so a 1-bit
prefilter into a dense rescore is strictly slower than the unfiltered
baseline (measured ~1.22 ms vs 0.89 ms at 500M scale).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from retrieve import OneBitKNNTriton, PrefilterKNNTriton
from retrieve.interfaces import FilterModule

from .filter import make_mask


class LinrV3Algo:
    is_cpu = False

    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        candidate_pool: int = 5000,
        v3_seed: int = 0,
        filter_mod: FilterModule | None = None,
    ) -> None:
        device = item_embs.device
        self.stage1 = OneBitKNNTriton(k=candidate_pool, seed=v3_seed).to(device)
        self.stage1.register_index(item_embs)
        self.stage2 = PrefilterKNNTriton(k=k).to(device)
        self.stage2.register_index(item_embs)
        self.filter_mod = filter_mod
        self.modules: list[nn.Module] = [self.stage1, self.stage2]
        if filter_mod is not None:
            self.modules.append(filter_mod)

    def forward(self, q: Tensor, qa_narrow: Tensor | None = None) -> tuple[Tensor, Tensor]:
        mask = make_mask(self.filter_mod, qa_narrow)
        cand_ids, _ = self.stage1(q, mask=mask)
        # When the mask admits < candidate_pool items stage 1's trailing slots
        # are -1; the stage-2 fused_masked_knn_topk does indirect loads via raw
        # pointer arithmetic, so item_embs_ptr + (-1)*stride walks off the
        # buffer. Pass per-row counts when masked so stage 2 only scores the
        # valid prefix.
        if mask is not None:
            counts = (cand_ids >= 0).sum(dim=1)
            return self.stage2(q, candidate_ids=cand_ids, counts=counts)
        return self.stage2(q, candidate_ids=cand_ids)
