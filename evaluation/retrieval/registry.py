"""Algorithm registry for the yambda retrieval benchmark.

Each entry returns a ``(forward, modules, is_cpu)`` triple: ``forward(query) ->
(ids, scores)``, the list of underlying ``nn.Module``\\s holding device buffers
(so callers can ``del`` them between cells to release memory), and a flag
indicating whether the index lives on CPU (``voyager_hnsw``) or GPU (everything
else). ``is_cpu`` lets the driver pick the right perf-measurement primitive
and zero out CUDA-only memory columns.

Cascade rationale (LinR §4.3, Fig knn-v3): the quantized 1-bit V3 produces a
top-T pre-filter list; V2 then reranks at full precision. V1 is intentionally
absent as a stage-2 — V1's dense matmul does the same work as ``triton_knn``
alone, so a V3→V1 cascade is strictly slower than the unfiltered baseline
(measured ~1.22 ms vs 0.89 ms at 500M scale).

``silvertorch`` here is ``SilverTorch`` with ``m_bits``/``k_hash`` left unset
(bloom disabled) since Yambda has no item attributes — running the bloom-fused
configuration against zero signatures would measure only the degenerate code
path. Build cost is dominated by k-means: expect ~30–90 s on a single A100/H100
for the 500M catalog (N≈1.87M, n_lists=1024, n_iter=10).

``voyager_hnsw`` is the lone CPU baseline — Spotify's HNSW (multi-threaded by
default, what production deployments actually run). The PyPI ``faiss-cpu``
wheel was tried and dropped: its bundled libgomp does not parallelize
correctly in this environment (>10× slowdown at 32+ threads vs 1), so its
single-thread numbers were not meaningful as a "realistic CPU" baseline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from retrieval.voyager_baseline import VoyagerHNSW
from retrieve import (
    FullScanKNN,
    LiNR_V1_Triton,
    LiNR_V2_Triton,
    LiNR_V3_Triton,
    SilverTorch,
)

ALGORITHMS = (
    "torch_fullscan",
    "triton_knn",
    "linr_v3_then_v2",
    "silvertorch",
    "voyager_hnsw",
)

_CPU_ALGOS = frozenset({"voyager_hnsw"})

ForwardFn = Callable[[Tensor], tuple[Tensor, Tensor]]


def build_algorithm(
    name: str,
    item_embs: Tensor,
    k: int,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[ForwardFn, list[torch.nn.Module], bool]:
    """Build an algorithm by name. Per-algo knobs live in ``params``.

    Returns ``(forward, modules, is_cpu)``.
    """
    p = params or {}
    device = item_embs.device
    is_cpu = name in _CPU_ALGOS

    if name == "torch_fullscan":
        idx = FullScanKNN(k=k).to(device)
        idx.register_index(item_embs)
        return (lambda q: idx(q)), [idx], is_cpu

    if name == "triton_knn":
        idx = LiNR_V1_Triton(k=k).to(device)
        idx.register_index(item_embs)
        return (lambda q: idx(q)), [idx], is_cpu

    if name == "linr_v3_then_v2":
        candidate_pool = int(p.get("candidate_pool", 5000))
        v3_seed = int(p.get("v3_seed", 0))
        v3 = LiNR_V3_Triton(k=candidate_pool, seed=v3_seed).to(device)
        v3.register_index(item_embs)
        stage2 = LiNR_V2_Triton(k=k).to(device)
        stage2.register_index(item_embs)

        # V2 takes candidate_ids directly — no mask round-trip, no scatter,
        # no [B, N+1] alloc, no compact_mask argsort.
        def forward(q: Tensor) -> tuple[Tensor, Tensor]:
            cand_ids, _ = v3(q)
            return stage2(q, candidate_ids=cand_ids)

        return forward, [v3, stage2], is_cpu

    if name == "silvertorch":
        n_lists = int(p.get("n_lists", 1024))
        n_probe = int(p.get("n_probe", 16))
        n_iter = int(p.get("n_iter", 10))
        seed = int(p.get("seed", 0))
        idx = SilverTorch(
            k=k,
            n_lists=n_lists,
            n_probe=n_probe,
            n_iter=n_iter,
            seed=seed,
        ).to(device)
        idx.register_index(item_embs)
        return (lambda q: idx(q)), [idx], is_cpu

    if name == "voyager_hnsw":
        m = int(p.get("m", 16))
        ef_construction = int(p.get("ef_construction", 200))
        ef_query = p.get("ef_query")
        ef_query = int(ef_query) if ef_query is not None else None
        num_threads = int(p.get("num_threads", -1))
        seed = int(p.get("seed", 0))
        idx = VoyagerHNSW(
            k=k,
            m=m,
            ef_construction=ef_construction,
            ef_query=ef_query,
            num_threads=num_threads,
            seed=seed,
        )
        idx.register_index(item_embs)
        return (lambda q: idx(q)), [idx], is_cpu

    raise ValueError(f"unknown algorithm: {name!r}")
