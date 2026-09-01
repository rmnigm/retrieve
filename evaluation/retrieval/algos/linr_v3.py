"""LiNR V3 -> V2 cascade: 1-bit Hamming pre-filter, fp32 rerank.

Stage 1 (LiNR §4.3) ``OneBitKNN`` produces a top-``candidate_pool`` list at 1-bit
precision; stage 2 ``PrefilterKNN`` rescores it at full precision. When filtered,
``filter_mod.evaluate_indices`` feeds stage 1's indirect-load path directly, so no
``[B, N]`` mask is materialized and the filter is consulted in exactly one place.

Both stages compile into **one** graph. Compiling them separately trips
``RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by
a subsequent run`` on the inter-stage ``cand_ids`` tensor.

Stage 2 is deliberately not ``PostfilterKNN``: a full matmul there does the same
work as ``triton_knn`` alone, making the cascade strictly slower than the
unfiltered baseline (~1.22 ms vs 0.89 ms at 500M).
"""

from __future__ import annotations

from torch import Tensor

from retrieve import OneBitKNN, PrefilterKNN
from retrieve.interfaces import Backend, FilterModule

from ._helpers import AlgoBase


class LinrV3Algo(AlgoBase):
    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        candidate_pool: int = 5000,
        v3_seed: int = 0,
        filter_mod: FilterModule | None = None,
        backend: Backend = "triton",
    ) -> None:
        super().__init__()
        device = item_embs.device
        # Stage 1 (OneBitKNN) is dtype-agnostic — sign-OPORP discards float
        # precision at bit-pack time. Stage 2 (PrefilterKNN) casts to fp16
        # internally for paper-faithful storage. See retrieve.layers.linr
        # package docstring for the full precision contract.
        self.stage1 = OneBitKNN(k=candidate_pool, seed=v3_seed, backend=backend).to(device)
        self.stage1.register_index(item_embs)
        self.stage2 = PrefilterKNN(k=k, backend=backend).to(device)
        self.stage2.register_index(item_embs)
        self._finalize(self.stage1, self.stage2, filter_mod=filter_mod)

    def forward(self, q: Tensor, qa_narrow: Tensor | None = None) -> tuple[Tensor, Tensor]:
        filtered = self.filter_mod is not None and qa_narrow is not None
        if filtered:
            # Single fused compact straight from the filter — no [B, N] bool
            # intermediate. On backend="triton" this is the clause_compact /
            # bloom_compact custom_op; on backend="torch" it falls back to
            # compact_mask(evaluate_mask(...)). Either way: one filter call,
            # one place. The retrieve library is 0-indexed over real items
            # (loaders.py drops the training-side padding row), so no item-0
            # fixup is needed.
            pos_idx, pcounts = self.filter_mod.evaluate_indices(qa_narrow)
            cand_ids, _ = self.stage1(q, candidate_ids=pos_idx, counts=pcounts)
        else:
            cand_ids, _ = self.stage1(q)
        # When the filter admits < candidate_pool items stage 1's trailing
        # slots are -1; the stage-2 fused_masked_knn_topk does indirect loads
        # via raw pointer arithmetic, so item_embs_ptr + (-1)*stride walks
        # off the buffer. Pass per-row counts when filtered so stage 2 only
        # scores the valid prefix.
        if filtered:
            counts = (cand_ids >= 0).sum(dim=1)
            return self.stage2(q, candidate_ids=cand_ids, counts=counts)
        return self.stage2(q, candidate_ids=cand_ids)
