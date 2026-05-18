"""Algorithm registry for the retrieval benchmark.

One ``nn.Module`` subclass per algo, with::

    algo.algo_modules: list[nn.Module]          # for memory cleanup
    algo.is_cpu: bool                           # CPU baseline marker
    algo(q, qa_narrow=None) -> (ids, scores)    # via Module.__call__

Each algo's ``__init__`` calls ``self.compile(dynamic=True,
mode="reduce-overhead")`` so the whole forward (filter + index +
cascade) becomes one cudagraph capture. ``algo_modules`` is a
separately maintained list (distinct from ``nn.Module.modules()``)
that the sweep driver iterates for explicit per-cell GPU-memory
cleanup.

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

from retrieve.interfaces import Backend, FilterModule

from ._helpers import collect_modules
from .filter import build_filter, make_mask
from .linr_v1 import LinrV1Algo
from .linr_v2 import LinrV2Algo
from .linr_v3 import LinrV3Algo
from .silvertorch import SilvertorchAlgo
from .torch_knn import TorchKnnAlgo

ALGORITHMS = (
    "torch_knn",
    "triton_knn",
    "linr_v1_filter_mask",
    "linr_v3",
    "linr_v2",
    "silvertorch",
)


# Algos that accept a `backend` parameter. Algos outside this set (currently
# only `torch_knn`) emit a single row regardless of `cfg.backends`.
BACKEND_CAPABLE_ALGOS = frozenset(
    {"triton_knn", "linr_v1_filter_mask", "linr_v2", "linr_v3", "silvertorch"}
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
    backend: Backend = "triton",
) -> Any:
    """Return an algo instance for one ``(name, filter_kind, backend)`` cell.

    Raises ``ValueError`` when the algo is incompatible with
    ``filter_kind``; the driver catches and skips that cell. ``backend``
    is forwarded to algos in ``BACKEND_CAPABLE_ALGOS`` (which thread it
    into the underlying ``RetrievalModule``); other algos ignore it.
    """
    p = params or {}

    if name == "torch_knn":
        return TorchKnnAlgo(item_embs, k, filter_mod=filter_mod)

    if name in ("triton_knn", "linr_v1_filter_mask"):
        return LinrV1Algo(item_embs, k, filter_mod=filter_mod, backend=backend)

    if name == "linr_v3":
        return LinrV3Algo(
            item_embs,
            k,
            candidate_pool=int(p.get("candidate_pool", 5000)),
            v3_seed=int(p.get("v3_seed", 0)),
            filter_mod=filter_mod,
            backend=backend,
        )

    if name == "linr_v2":
        if filter_mod is None:
            raise ValueError("linr_v2 requires a filter (clause or bloom)")
        return LinrV2Algo(item_embs, k, filter_mod=filter_mod, backend=backend)

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
            backend=backend,
        )

    raise ValueError(f"unknown algorithm: {name!r}")


__all__ = [
    "ALGORITHMS",
    "BACKEND_CAPABLE_ALGOS",
    "build_algorithm",
    "build_filter",
    "collect_modules",
    "make_mask",
]
