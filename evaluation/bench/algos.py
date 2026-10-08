"""The algorithm table (the retired library-harness-boundary plan; see
docs/system/architecture.md for the current module map): harness name → library class, the
filter kinds and backends, ``PATHS`` derived from ``retrieve.interfaces.DISPATCH``, and
``build`` — the one factory the cell loop calls.

``PATHS[(algo, filter_kind, backend)]`` is the code path that actually runs, or ``None`` when there
is no such cell: ``DISPATCH``'s label for the module's backend (``cublas`` where the flag is a
no-op, ``None`` where the constructor raises — ``official`` outside SilverTorch), suffixed with the
filter's backend on filter cells of the cuBLAS algos (``cublas+triton``: cuBLAS scoring, Triton
``clause_mask``), and ``None`` on ``linr_v2 / none`` — its candidate source *is* the filter.
``postfilter`` is the harness's own baseline (``bench.postfilter``), not a library module:
``DISPATCH`` gains its row here, torch only, filter cells only. ``tests/bench/test_paths.py`` pins
the derived table. The standalone filter modules of ``official`` cells are Triton (O §6.2):
``filter_backend``.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from bench.postfilter import Postfilter
from retrieve import (
    BloomFilter,
    ExactAttributeFilter,
    LiNRV1,
    LiNRV2,
    LiNRV3,
    OfficialConfig,
    SilverTorch,
)
from retrieve.interfaces import DISPATCH as LIBRARY_DISPATCH
from retrieve.interfaces import FilterModule

ALGOS: dict[str, type[nn.Module]] = {
    "linr_v1_filter_mask": LiNRV1,
    "linr_v2": LiNRV2,
    "linr_v3": LiNRV3,
    "silvertorch": SilverTorch,
    "postfilter": Postfilter,
}
DISPATCH = {
    **LIBRARY_DISPATCH,
    "Postfilter": {"triton": None, "torch": "cublas", "official": None},
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
    if p is None or (algo in ("linr_v2", "postfilter") and filter_kind == "none"):
        return None
    return f"{p}+{backend}" if p == "cublas" and filter_kind != "none" else p


PATHS: dict[tuple[str, str, str], str | None] = {
    (a, f, b): _path(a, f, b) for a in ALGOS for f in FILTER_KINDS for b in BACKENDS
}


def is_valid_combo(algo: str, params: dict[str, Any]) -> bool:
    """``n_probe <= n_lists`` — the one combo the library would reject at build."""
    p = {**SILVERTORCH_DEFAULTS, **params}
    return not (algo == "silvertorch" and p["n_probe"] > p["n_lists"])


def official_config(
    algo: str, filter_kind: str, backend: str, params: dict[str, Any]
) -> OfficialConfig | None:
    """The ``bloom_path`` build param (the S9 co-design ablation) as ``OfficialConfig``;
    ``None`` (the library default, ``bloom_path="partial"``) when the key is absent. It has a
    meaning on ``silvertorch / bloom / official`` only and raises anywhere else."""
    if "bloom_path" not in params:
        return None
    if (algo, filter_kind, backend) != ("silvertorch", "bloom", "official"):
        raise ValueError(
            f"bloom_path applies to silvertorch/bloom/official only, got {algo}/{filter_kind}/"
            f"{backend}"
        )
    return OfficialConfig(bloom_path=params["bloom_path"])


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
    if item_attrs is None:
        raise ValueError(f"filter_kind={filter_kind} needs item_attrs")
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
    (``n_lists``, ``n_probe``, ``n_iter``, ``m_bits``, ``k_hash``, ``candidate_pool``, ``alpha``,
    ``compile``); ``seed`` drives the k-means and the OPORP projection. SilverTorch fuses the
    predicate (the attribute buffers live inside it, ``filter_mod`` is unused); the LiNR modules
    and ``postfilter`` take the standalone filter as ``self.filter``. ``compile`` wraps the built
    module in place (``compile_module``). Raises on cells ``PATHS`` marks ``None``."""
    p = dict(params or {})
    compile_mode = p.pop("compile", None)
    if PATHS[algo, filter_kind, backend] is None:
        raise ValueError(f"no code path for ({algo}, {filter_kind}, {backend})")
    if not is_valid_combo(algo, p):
        raise ValueError(f"invalid params for {algo}: {p}")
    if algo == "silvertorch":
        mode = FILTER_MODE[filter_kind]
        bloom = BLOOM_DEFAULTS if mode == "bloom" else {}
        official = official_config(algo, filter_kind, backend, p)
        p.pop("bloom_path", None)
        module = SilverTorch(
            k=k, filter_mode=mode, seed=seed, backend=backend, official=official,
            **{**SILVERTORCH_DEFAULTS, **bloom, **p},
        )  # fmt: skip
        if mode == "none":
            module.register_index(item_embs)
        else:
            if item_attrs is None:
                raise ValueError(f"silvertorch/{filter_kind} needs item_attrs")
            rev = clause_is_reverse if mode == "exact" else None
            module.register_index(item_embs, item_clause_attrs=item_attrs, clause_is_reverse=rev)
    else:
        if algo == "linr_v3":
            p["seed"] = seed
        module = ALGOS[algo](k, filter=filter_mod, backend=backend, **p)
        module.register_index(item_embs)
    return module if compile_mode is None else compile_module(module, compile_mode)


def compile_module(module: nn.Module, mode: str) -> nn.Module:
    """``torch.compile(mode=mode)`` in place (``nn.Module.compile``), so ``k``,
    ``set_query_params`` and the buffers stay the module's own. Compilation is lazy: the first
    forward pays it. ``max-autotune`` already replays inductor's own CUDA graphs, so the module
    is marked uncapturable and ``graph`` mode records a null entry (``not_capturable``)."""
    module.compile(mode=mode)
    cls = type(module)  # SilverTorch's capturable is a read-only property: override per instance
    module.__class__ = type(cls.__name__, (cls,), {"capturable": False})
    return module


__all__ = [
    "ALGOS",
    "BACKENDS",
    "BLOOM_DEFAULTS",
    "DISPATCH",
    "FILTER_KINDS",
    "FILTER_MODE",
    "PATHS",
    "SILVERTORCH_DEFAULTS",
    "build",
    "build_filter",
    "compile_module",
    "filter_backend",
    "is_valid_combo",
    "official_config",
]
