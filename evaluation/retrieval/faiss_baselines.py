"""CPU faiss baselines for the yambda retrieval bench.

Two ``RetrievalModule`` wrappers around faiss-cpu indices, conforming to the
``register_index(item_embs)`` + ``forward(query) -> (ids, scores)`` contract
used by the rest of the bench. Single-thread (``faiss.omp_set_num_threads(1)``
at import time) so that the bs=1 latency is apples-to-apples vs the GPU
implementations — multi-thread can be added later behind a config knob.
"""

from __future__ import annotations

import faiss
import numpy as np
import torch
from torch import Tensor

from retrieve.interfaces import RetrievalModule

faiss.omp_set_num_threads(1)


def _to_np(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32, copy=False)


class FaissFlatIP(RetrievalModule):
    """Exhaustive inner-product search via ``faiss.IndexFlatIP``."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = int(k)
        self._index: faiss.Index | None = None

    def register_index(self, item_embs: Tensor) -> None:
        x = item_embs.detach().clone()
        x[0] = 0.0  # mirror the GPU pad treatment
        x_np = _to_np(x)
        d = x_np.shape[1]
        self._index = faiss.IndexFlatIP(d)
        self._index.add(x_np)

    def forward(self, query: Tensor) -> tuple[Tensor, Tensor]:
        assert self._index is not None, "register_index() not called"
        q_np = _to_np(query)
        scores_np, ids_np = self._index.search(q_np, self.k)
        ids = torch.from_numpy(ids_np.astype(np.int64)).to(query.device)
        scores = torch.from_numpy(scores_np).to(query.device)
        return ids, scores


class FaissIVFFlat(RetrievalModule):
    """Inverted-file IP search via ``faiss.IndexIVFFlat`` (CPU)."""

    def __init__(self, k: int, nlist: int = 2048, nprobe: int = 16, seed: int = 0) -> None:
        super().__init__()
        self.k = int(k)
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.seed = int(seed)
        self._index: faiss.IndexIVFFlat | None = None

    def register_index(self, item_embs: Tensor) -> None:
        x = item_embs.detach().clone()
        x[0] = 0.0
        x_np = _to_np(x)
        d = x_np.shape[1]
        # numpy seed makes faiss k-means init deterministic.
        np.random.seed(self.seed)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(x_np)
        index.add(x_np)
        index.nprobe = self.nprobe
        self._index = index

    def forward(self, query: Tensor) -> tuple[Tensor, Tensor]:
        assert self._index is not None, "register_index() not called"
        q_np = _to_np(query)
        scores_np, ids_np = self._index.search(q_np, self.k)
        ids = torch.from_numpy(ids_np.astype(np.int64)).to(query.device)
        scores = torch.from_numpy(scores_np).to(query.device)
        return ids, scores
