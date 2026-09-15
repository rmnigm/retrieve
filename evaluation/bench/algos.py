"""The algorithm table (library-harness-boundary.md §3): harness name → library class, the
filter kinds and backends, ``PATHS`` derived from ``retrieve.interfaces.DISPATCH``, and
``build`` — the one factory the cell loop calls.

``PATHS[(algo, filter_kind, backend)]`` is the code path that actually runs, or ``None`` when
there is no such cell: ``DISPATCH``'s label for the module's backend (``cublas`` where the flag
is a no-op, ``None`` where the constructor raises — ``official`` outside SilverTorch), suffixed
with the filter's backend on filter cells of the cuBLAS algos (``cublas+triton``: cuBLAS
scoring, Triton ``clause_mask``), and ``None`` on ``linr_v2 / none`` — its candidate source
*is* the filter. ``tests/bench/test_paths.py`` pins the derived table. The standalone filter
modules of ``official`` cells are Triton (O §6.2): ``filter_backend``.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from retrieve import (
    BloomFilter,
    ExactAttributeFilter,
    LiNRV1,
    LiNRV2,
    LiNRV3,
    LiNRV4,
    SilverTorch,
)
from retrieve.interfaces import DISPATCH, FilterModule

ALGOS: dict[str, type[nn.Module]] = {
    "linr_v1_filter_mask": LiNRV1,
    "linr_v2": LiNRV2,
    "linr_v3": LiNRV3,
    "linr_v4": LiNRV4,
    "silvertorch": SilverTorch,
}
FILTER_KINDS = ("none", "clause", "bloom")
BACKENDS = ("triton", "torch", "official")
FILTER_MODE = {"none": "none", "clause": "exact", "bloom": "bloom"}  # filter_kind → filter_mode
SILVERTORCH_DEFAULTS = {"n_lists": 1024, "n_probe": 24, "n_iter": 10}
BLOOM_DEFAULTS = {"m_bits": 1024, "k_hash": 5}


def filter_backend(backend: str) -> str:
    return "triton" if backend == "official" else backend


def _path(algo: str, filter_kind: str, backend: str) -> str | None:
    p = DISPATCH[ALGOS[algo].__name__][backend]
    if p is None or (algo == "linr_v2" and filter_kind == "none"):
        return None
    return f"{p}+{backend}" if p == "cublas" and filter_kind != "none" else p


PATHS: dict[tuple[str, str, str], str | None] = {
    (a, f, b): _path(a, f, b) for a in ALGOS for f in FILTER_KINDS for b in BACKENDS
}


def is_valid_combo(algo: str, params: dict[str, Any]) -> bool:
    """``n_probe <= n_lists`` — the one combo the library would reject at build."""
    p = {**SILVERTORCH_DEFAULTS, **params}
    return not (algo == "silvertorch" and p["n_probe"] > p["n_lists"])


def build_filter(
    filter_kind: str,
    item_attrs: Tensor | None,
    *,
    clause_is_reverse: Tensor | None = None,
    backend: str = "triton",
    m_bits: int = BLOOM_DEFAULTS["m_bits"],
    k_hash: int = BLOOM_DEFAULTS["k_hash"],
) -> FilterModule | None:
    """One standalone FilterModule per (filter_kind, filter backend); ``None`` for ``none``.
    Bloom is paper-strict forward-only, so it never sees ``clause_is_reverse``."""
    if filter_kind == "none":
        return None
    assert item_attrs is not None, f"filter_kind={filter_kind} needs item_attrs"
    if filter_kind == "clause":
        f: FilterModule = ExactAttributeFilter(backend=backend)
        f.register_index(item_attrs, clause_is_reverse=clause_is_reverse)
    elif filter_kind == "bloom":
        f = BloomFilter(m_bits=m_bits, k_hash=k_hash, backend=backend)
        f.register_index(item_attrs)
    else:
        raise ValueError(f"unknown filter_kind {filter_kind!r}")
    return f


def build(
    algo: str,
    item_embs: Tensor,
    *,
    k: int,
    backend: str,
    filter_kind: str = "none",
    filter_mod: FilterModule | None = None,
    item_attrs: Tensor | None = None,
    clause_is_reverse: Tensor | None = None,
    params: dict[str, Any] | None = None,
    seed: int = 0,
) -> nn.Module:
    """Construct and register one cell's module. ``params`` = build + query params merged
    (``n_lists``, ``n_probe``, ``n_iter``, ``m_bits``, ``k_hash``, ``candidate_pool``); ``seed``
    drives the k-means and the OPORP projection. SilverTorch fuses the predicate (the
    attribute buffers live inside it, ``filter_mod`` is unused); the LiNR modules take the
    standalone filter as ``self.filter``. Raises on cells ``PATHS`` marks ``None``."""
    p = dict(params or {})
    if PATHS[algo, filter_kind, backend] is None:
        raise ValueError(f"no code path for ({algo}, {filter_kind}, {backend})")
    if not is_valid_combo(algo, p):
        raise ValueError(f"invalid params for {algo}: {p}")
    if algo == "silvertorch":
        mode = FILTER_MODE[filter_kind]
        bloom = BLOOM_DEFAULTS if mode == "bloom" else {}
        module = SilverTorch(
            k=k, filter_mode=mode, seed=seed, backend=backend,
            **{**SILVERTORCH_DEFAULTS, **bloom, **p},
        )  # fmt: skip
        if mode == "none":
            module.register_index(item_embs)
        else:
            assert item_attrs is not None, f"silvertorch/{filter_kind} needs item_attrs"
            rev = clause_is_reverse if mode == "exact" else None
            module.register_index(item_embs, item_clause_attrs=item_attrs, clause_is_reverse=rev)
        return module
    if algo == "linr_v3":
        p["seed"] = seed
    module = ALGOS[algo](k, filter=filter_mod, backend=backend, **p)
    module.register_index(item_embs)
    return module


__all__ = [
    "ALGOS",
    "BACKENDS",
    "BLOOM_DEFAULTS",
    "FILTER_KINDS",
    "FILTER_MODE",
    "PATHS",
    "SILVERTORCH_DEFAULTS",
    "build",
    "build_filter",
    "filter_backend",
    "is_valid_combo",
]
