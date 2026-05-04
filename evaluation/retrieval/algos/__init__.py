"""Algorithm registry for the retrieval benchmark.

One class per algo, duck-typed protocol::

    algo.modules: list[nn.Module]               # for memory cleanup
    algo.is_cpu: bool                           # CPU baseline marker
    algo.forward(q, qa_narrow=None) -> (ids, scores)

``build_algorithm`` is a thin factory: it picks the class for ``name``,
unpacks ``params`` into its constructor, and forwards
``filter_kind``/``filter_mod``/``item_attrs_narrow`` so each class can
opt in to whichever inputs it actually consumes.

The only construction-time eligibility rule: ``linr_v2`` requires a
filter and raises on ``filter_kind="none"``. Silvertorch on
reverse-clause sweeps is skipped by the driver itself
(``evaluate.py``), not here.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from retrieve.interfaces import FilterModule

from .filter import build_filter, make_mask
from .linr_v1 import LinrV1Algo
from .linr_v2 import LinrV2Algo
from .linr_v3 import LinrV3Algo
from .silvertorch import SilvertorchAlgo
from .torch_knn import TorchKnnAlgo
from .voyager import VoyagerHNSWAlgo

ALGORITHMS = (
    "torch_knn",
    "triton_knn",
    "linr_v1_filter_mask",
    "linr_v3",
    "linr_v2",
    "silvertorch",
    "voyager_hnsw",
)


def build_algorithm(
    name: str,
    item_embs: Tensor,
    k: int,
    *,
    filter_kind: str = "none",
    filter_mod: FilterModule | None = None,
    item_attrs_narrow: Tensor | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    """Return an algo instance for one ``(name, filter_kind)`` cell.

    Raises ``ValueError`` when the algo is incompatible with
    ``filter_kind``; the driver catches and skips that cell.
    """
    p = params or {}

    if name == "torch_knn":
        return TorchKnnAlgo(item_embs, k, filter_mod=filter_mod)

    if name in ("triton_knn", "linr_v1_filter_mask"):
        return LinrV1Algo(item_embs, k, filter_mod=filter_mod)

    if name == "linr_v3":
        return LinrV3Algo(
            item_embs,
            k,
            candidate_pool=int(p.get("candidate_pool", 5000)),
            v3_seed=int(p.get("v3_seed", 0)),
            filter_mod=filter_mod,
        )

    if name == "linr_v2":
        if filter_mod is None:
            raise ValueError("linr_v2 requires a filter (clause or bloom)")
        return LinrV2Algo(item_embs, k, filter_mod=filter_mod)

    if name == "silvertorch":
        return SilvertorchAlgo(
            item_embs,
            k,
            filter_kind=filter_kind,
            item_attrs_narrow=item_attrs_narrow,
            n_lists=int(p.get("n_lists", 1024)),
            n_probe=int(p.get("n_probe", 24)),
            n_iter=int(p.get("n_iter", 10)),
            m_bits=int(p.get("m_bits", 1024)),
            k_hash=int(p.get("k_hash", 5)),
            seed=int(p.get("seed", 0)),
        )

    if name == "voyager_hnsw":
        ef_query = p.get("ef_query")
        return VoyagerHNSWAlgo(
            item_embs,
            k,
            m=int(p.get("m", 16)),
            ef_construction=int(p.get("ef_construction", 200)),
            ef_query=int(ef_query) if ef_query is not None else None,
            num_threads=int(p.get("num_threads", -1)),
            seed=int(p.get("seed", 0)),
            filter_mod=filter_mod,
        )

    raise ValueError(f"unknown algorithm: {name!r}")


__all__ = [
    "ALGORITHMS",
    "build_algorithm",
    "build_filter",
    "make_mask",
]
