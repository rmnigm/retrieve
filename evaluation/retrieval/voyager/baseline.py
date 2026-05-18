"""Voyager (Spotify HNSW) CPU baseline for the yambda retrieval bench.

A ``RetrievalModule`` wrapper around ``voyager.Index`` (Space.InnerProduct),
conforming to the ``register_index(item_embs)`` + ``forward(query) -> (ids,
scores)`` contract. Voyager is HNSW under the hood, multi-threaded by default
(``num_threads=-1`` uses all cores), and is what Spotify ships in production
for their recommender stack — a more realistic "CPU ANN deployment" baseline
than single-thread faiss.

Distance convention: voyager's InnerProduct space returns ``1 - dot(a, b)``
(smaller = better). We convert back to a positive score (``-dist`` is enough
for ranking; the absolute value isn't used by the recall/ndcg metrics).

Build cache: HNSW build only depends on (item_embs, M, ef_construction, seed),
not on ``k`` or ``ef_query``. The bench creates a fresh wrapper per
``(algo, k)`` cell, which would rebuild twice per config — wasteful at 1.87M
items × M=32 (~10 min/build). A process-local cache keyed on those four
factors lets the second cell reuse the first cell's index; the harness's
allocator-snapshot bookkeeping is GPU-only and unaffected.
"""

from __future__ import annotations

import numpy as np
import torch
import voyager
from torch import Tensor

from retrieve.interfaces import RetrievalModule

_INDEX_CACHE: dict[tuple, voyager.Index] = {}


def _to_np(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32, copy=False)


class VoyagerHNSW(RetrievalModule):
    """HNSW IP search via ``voyager.Index(Space.InnerProduct, ...)``."""

    def __init__(
        self,
        k: int,
        m: int = 32,
        ef_construction: int = 400,
        ef_query: int | None = None,
        num_threads: int = -1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        # ef_query must be >= k for HNSW to return k results; default to max(2×k, 500)
        # which targets ~95% recall vs full-scan on D=64 gaussian-like embeddings.
        self.ef_query = int(ef_query) if ef_query is not None else max(2 * self.k, 500)
        self.num_threads = int(num_threads)
        self.seed = int(seed)
        self._index: voyager.Index | None = None

    def register_index(self, item_embs: Tensor) -> None:
        cache_key = (id(item_embs), self.m, self.ef_construction, self.seed)
        cached = _INDEX_CACHE.get(cache_key)
        if cached is not None:
            self._index = cached
            return

        x = item_embs.detach().clone()
        x[0] = 0.0  # mirror the GPU pad treatment
        x_np = _to_np(x)
        n, d = x_np.shape
        index = voyager.Index(
            voyager.Space.InnerProduct,
            num_dimensions=d,
            M=self.m,
            ef_construction=self.ef_construction,
            random_seed=self.seed,
            max_elements=n,
        )
        # Use original row indices as ids so query results are directly usable.
        index.add_items(x_np, ids=list(range(n)), num_threads=self.num_threads)
        self._index = index
        _INDEX_CACHE[cache_key] = index

    def forward(self, query: Tensor) -> tuple[Tensor, Tensor]:
        assert self._index is not None, "register_index() not called"
        q_np = _to_np(query)
        if q_np.ndim == 1:
            q_np = q_np[None, :]
        ids_np, dist_np = self._index.query(
            q_np,
            k=self.k,
            num_threads=self.num_threads,
            query_ef=self.ef_query,
        )
        ids = torch.from_numpy(ids_np.astype(np.int64)).to(query.device)
        # InnerProduct space: dist = 1 - dot(a, b); scores recovered as 1 - dist.
        scores = torch.from_numpy(1.0 - dist_np).to(query.device)
        return ids, scores
