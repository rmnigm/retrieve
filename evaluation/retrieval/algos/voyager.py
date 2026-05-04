"""Voyager HNSW — Spotify's CPU baseline (multi-threaded).

The lone CPU algo. PyPI ``faiss-cpu`` was tried and dropped: its
bundled libgomp does not parallelize correctly here (>10× slowdown at
32+ threads vs 1), so faiss's single-thread numbers were not
meaningful as a "realistic CPU" baseline.

On filter cells, post-filters by gathering the per-row mask over the
returned ids (exact, not approximate — HNSW returns near-exact ids at
``ef_query=1000``).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from retrieval.voyager_baseline import VoyagerHNSW
from retrieve.interfaces import FilterModule

from .filter import make_mask


class VoyagerHNSWAlgo:
    is_cpu = True

    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        m: int = 16,
        ef_construction: int = 200,
        ef_query: int | None = None,
        num_threads: int = -1,
        seed: int = 0,
        filter_mod: FilterModule | None = None,
    ) -> None:
        self.idx = VoyagerHNSW(
            k=k,
            m=m,
            ef_construction=ef_construction,
            ef_query=ef_query,
            num_threads=num_threads,
            seed=seed,
        )
        self.idx.register_index(item_embs)
        self.filter_mod = filter_mod
        self.item_device = item_embs.device
        self.modules: list[nn.Module] = [self.idx]
        if filter_mod is not None:
            self.modules.append(filter_mod)

    def forward(self, q: Tensor, qa_narrow: Tensor | None = None) -> tuple[Tensor, Tensor]:
        ids, scores = self.idx(q)
        qa_dev = qa_narrow.to(self.item_device) if qa_narrow is not None else None
        mask = make_mask(self.filter_mod, qa_dev)
        if mask is not None:
            ids_dev = ids.to(mask.device)
            keep = mask.gather(1, ids_dev)
            ids = ids_dev.masked_fill(~keep, -1)
        return ids, scores
