"""The algorithm table (docs/system/evaluation.md § Algorithms and the `PATHS` table) and
``build``, the one factory the cell loop calls."""

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
    OneBitKNN,
    SilverTorch,
)
from retrieve.interfaces import DISPATCH as LIBRARY_DISPATCH
from retrieve.interfaces import OFFICIAL_BACKENDS, FilterModule

ALGOS: dict[str, type[nn.Module]] = {
    "linr_v1_filter_mask": LiNRV1,
    "linr_v2": LiNRV2,
    "linr_v3": LiNRV3,
    "silvertorch": SilverTorch,
    "postfilter": Postfilter,
}
DISPATCH = {
    **LIBRARY_DISPATCH,
    "Postfilter": {"triton": None, "torch": "cublas", "official": None, "official-fork": None},
}
FILTER_KINDS = ("none", "clause", "bloom")
BACKENDS = ("triton", "torch", *OFFICIAL_BACKENDS)
FILTER_MODE = {"none": "none", "clause": "exact", "bloom": "bloom"}  # filter_kind → filter_mode
SILVERTORCH_DEFAULTS = {"n_lists": 1024, "n_probe": 24, "n_iter": 10}
BLOOM_DEFAULTS = {"m_bits": 1024, "k_hash": 5}


def filter_backend(backend: str) -> str:
    return "triton" if backend in OFFICIAL_BACKENDS else backend


def _path(algo: str, filter_kind: str, backend: str) -> str | None:
    p = DISPATCH[ALGOS[algo].__name__][backend]
    if p is None or (algo in ("linr_v2", "postfilter") and filter_kind == "none"):
        return None
    if filter_kind == "bloom" and algo != "silvertorch":
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
    """``bloom_path`` / ``score_path`` / ``recheck`` as an ``OfficialConfig``, ``None`` when all
    are absent; each key raises on an arm where it means nothing (docs/system/evaluation.md
    § Algorithms and the `PATHS` table)."""
    kw = {k: params[k] for k in ("bloom_path", "score_path", "recheck") if k in params}
    if not kw:
        return None
    if "bloom_path" in kw and (
        (algo, filter_kind) != ("silvertorch", "bloom") or backend not in OFFICIAL_BACKENDS
    ):
        raise ValueError(
            f"bloom_path applies to silvertorch/bloom on the official backends only, got "
            f"{algo}/{filter_kind}/{backend}"
        )
    if "score_path" in kw and (algo != "silvertorch" or backend not in OFFICIAL_BACKENDS):
        raise ValueError(
            f"score_path applies to silvertorch on the official backends only, got {algo}/{backend}"
        )
    if "recheck" in kw and (
        (algo, filter_kind) != ("silvertorch", "bloom")
        or backend not in OFFICIAL_BACKENDS
        or (backend == "official" and kw["recheck"])
    ):
        raise ValueError(
            f"recheck applies to silvertorch/bloom on official-fork (false only on official), got "
            f"{algo}/{filter_kind}/{backend} recheck={kw['recheck']}"
        )
    return OfficialConfig(**kw)


def build_filter(
    filter_kind: str,
    item_attrs: Tensor | None,
    *,
    clause_is_reverse: Tensor | None = None,
    backend: str = "triton",
    m_bits: int = BLOOM_DEFAULTS["m_bits"],
    k_hash: int = BLOOM_DEFAULTS["k_hash"],
) -> FilterModule | None:
    """One standalone ``FilterModule``; ``None`` for ``none``. Bloom is forward-only, so it never
    sees ``clause_is_reverse``."""
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
    """Construct and register one cell's module from its merged build + query ``params``;
    raises on cells ``PATHS`` marks ``None``."""
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
        for key in ("bloom_path", "score_path", "recheck"):
            p.pop(key, None)
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
        k_bits = p.pop("k_bits", None) if algo == "linr_v3" else None
        if algo == "linr_v3":
            p["seed"] = seed
        module = ALGOS[algo](k, filter=filter_mod, backend=backend, **p)
        if k_bits is not None:  # LiNRV3 does not take k_bits: its stage 1 does
            module.stage1 = OneBitKNN(k=module.stage1.k, seed=seed, backend=backend, k_bits=k_bits)
        module.register_index(item_embs)
    return module if compile_mode is None else compile_module(module, compile_mode)


def compile_module(module: nn.Module, mode: str) -> nn.Module:
    """``nn.Module.compile`` in place; ``max-autotune`` replays inductor's own CUDA graphs, so
    the module is marked uncapturable (docs/system/evaluation.md § Compiled arms)."""
    module.compile(mode=mode)
    cls = type(module)  # SilverTorch's capturable is a read-only property: override per instance
    module.__class__ = type(cls.__name__, (cls,), {"capturable": False})
    return module
