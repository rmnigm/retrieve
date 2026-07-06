"""LiNR V2 + compact filter primitive — exact filtered top-K, sparse path.

The candidate source IS the filter: ``filter_mod.evaluate_indices(qa)``
returns ``(ids [B, P], counts [B])`` with no ``[B, N]`` mask
materialized. ``PrefilterKNN`` then rescores those P candidates at
fp32 (``backend="triton"`` default; ``"torch"`` available too).
The whole algo forward (filter compact-indices build + index call) is
wrapped with ``torch.compile(dynamic=True, mode="reduce-overhead")``
in ``__init__`` regardless of backend — Inductor fuses gather +
matmul + topk and cudagraph_trees replay collapses launch overhead.
``filter_mod`` is required (no unfiltered mode) — that's the whole
point of the compact path.
"""

from __future__ import annotations

from torch import Tensor, nn

from retrieve import PrefilterKNN
from retrieve.interfaces import Backend, FilterModule

from ._helpers import collect_modules


class LinrV2Algo(nn.Module):
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
        self.filter_mod = filter_mod
        self.algo_modules = collect_modules(self.idx, filter_mod=filter_mod)
        self.compile(dynamic=True, mode="reduce-overhead")

    def forward(self, q: Tensor, qa_narrow: Tensor) -> tuple[Tensor, Tensor]:
        cand, counts = self.filter_mod.evaluate_indices(qa_narrow)
        return self.idx(q, candidate_ids=cand, counts=counts)
