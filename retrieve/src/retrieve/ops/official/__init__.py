"""Meta's SilverTorch ops, adapted (``torch.ops.st.*``; the workspace member ``silvertorch/``,
Meta's `meta-recsys/silvertorch <https://github.com/meta-recsys/silvertorch>`_ at ``21aa35e``
with the changes listed in its ``CHANGES.md``), behind one adapter: ``SilverTorch(backend="cuda")``
keeps our phase 1 (k-means, int8 quantization, probe selection) and runs phases 2+3 as one
``st::search`` call; the upstream-shaped ops (``fused_kmean_ann``, the bloom index build and
searches, the expression parser) are wrapped here for the gates and for Meta's own modules
(``retrieve.modules.official``). Loader, ``OfficialConfig`` and the upstream constants live in
this package; the op wrappers in ``adapter`` are re-exported below. ``retrieve.ops.official.st``
is ``torch.ops.st`` once loaded (docs/system/kernels.md § Meta's ops, "Ops called").

The adapted ops do not sync the host and have shape functions, so the ``cuda`` forward captures
into a CUDA graph and compiles (docs/system/kernels.md § cuda).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal

import torch

BitOrder = Literal["high_first", "low_first"]
ScorePath = Literal["int32", "fp16"]
BloomWidth = Literal["configured", "upstream"]

PACKAGE, NAMESPACE = "silvertorch", "st"

# The scorer reads doc ``d`` of a ``filtering_bit_mask`` / partial column mask at bit
# ``63 - d % 64`` of word ``d // 64`` (kernels.md § Meta's ops, "Bit order — measured, high-first").
MASK_BIT_ORDER: BitOrder = "high_first"

# ``bloom_index_search_batch(return_bool_mask=False)`` packs from bit 63 downwards
# (``bloom_index_search_cuda.cu:552-556``).
BLOOM_OUTPUT_BIT_ORDER: BitOrder = "high_first"

# ``MAX_K_V2`` (``bloom_index_util.h:45``): the search kernel keeps the resolved hash
# positions in a fixed ``std::array<…, MAX_K_V2>`` (``bloom_index_search_cuda.cu:254``) and
# never checks ``k`` against it — an unchecked hard limit we enforce host-side.
MAX_SEARCH_K = 10
DOCS_PER_BUNDLE = 2048  # C_BITS_IN_BLOOM_V2_COL_BUNDLE = 64 bits × 32 columns
WORDS_PER_BUNDLE = 32  # C_BLOOM_V2_COL_BUNDLE_SIZE
WARP = 32  # output rows are padded to a multiple of the warp size (fused_kmean_ann.cpp:107)
FP16_MAX = 65504.0

REQUIRED_OPS = (
    "fused_kmean_ann",
    "fused_kmean_ann_with_partial_masks",
    "bloom_index_build",
    "parse_expression_query_batch",
    "bloom_index_search_batch",
    "bloom_index_search_batch_return_partial_response",
    "search",
)


class OfficialMissing(ImportError):
    """The ``silvertorch`` package cannot run here: not installed, or no CUDA device. Distinct
    from a plain ``ImportError``, which means it *is* installed and its extension is
    broken — tests skip on the former and fail on the latter (``require_official``)."""


@dataclass(frozen=True)
class OfficialConfig:
    """Knobs of ``SilverTorch`` that have no counterpart on the other modules.

    - ``score_path``: ``"int32"`` → ``divisor_for_int8=-1``, the raw int32 dot, epilogue
      ``(dot·q_scale)·global_scale`` **bit-identical** to the torch oracle (the parity path);
      ``"fp16"`` → ``divisor_for_int8 = divisor`` (a power of two chosen so ``127²·D / divisor ≤
      65504``), the kernel writes ``fp16(dot / divisor)`` — the instantiation Meta ships for int8
      serving, hence the timed path.
    - ``divisor``: ``None`` → :func:`default_divisor` of the index width.
    - ``b_multiplier``: bloom width per bundle = ``max_terms_per_doc · k · b_multiplier``
      bits per doc (``bloom_indexer.cpp:52-66``); must be ``> 1.0``. Read when ``bloom_width`` is
      ``"upstream"``.
    - ``n_stored_hashes``: raw murmur hashes precomputed per query term at parse time (the
      ops' ``hash_k`` argument — renamed here because the library-wide ``k_hash`` is the
      *search* ``k``, ``SilverTorch.k_hash``); the search ANDs the first ``k`` *distinct*
      positions among them, so it must exceed ``k`` by a margin (README uses 7 for k=3).
    - ``build_k``: hash positions set per term at index build; ``None`` → the search
      ``k`` (README's usage; the upstream module builder passes ``hash_k`` instead, which
      sets extra bits per term).
    - ``max_sub_queries``: parser fan-out bound per AND/OR node (semantics unchanged).
    - ``fast_build``: the CUDA builder's single-pass signature-free build (ignored by the
      CPU builder).
    - ``bloom_width``: ``"configured"`` (default) → ``m_bits`` bits per doc in every bundle
      (CHANGES.md C12; the layer's ``m_bits`` is then required); ``"upstream"`` → Meta's width,
      each 2048-doc bundle ``max_terms · k · b_multiplier`` bits per doc (``m_bits`` ignored).
    - ``recheck``: test the bloom's survivors exactly (CHANGES.md C13; default on): bloom ∧ exact
      is the exact filter, so no false positive is returned and the bloom only spares the exact
      test its failing docs; the layer then keeps the exact attrs beside the bloom index. Off:
      Meta's bloom semantics, false positives included.

    The expression parse is query preparation: ``SilverTorch.prepare_queries`` runs it once
    per batch, outside ``forward``."""

    score_path: ScorePath = "fp16"
    divisor: int | None = None
    b_multiplier: float = 10.0
    build_k: int | None = None
    max_sub_queries: int = 5
    fast_build: bool = False
    n_stored_hashes: int = 7
    bloom_width: BloomWidth = "configured"
    recheck: bool = True

    def __post_init__(self) -> None:
        if self.score_path not in ("int32", "fp16"):
            raise ValueError(f"score_path must be 'int32' or 'fp16', got {self.score_path!r}")
        if self.divisor is not None and (
            self.divisor <= 0 or (self.divisor & (self.divisor - 1)) != 0
        ):
            raise ValueError(f"divisor must be a positive power of two, got {self.divisor}")
        if not self.b_multiplier > 1.0:
            raise ValueError(
                f"b_multiplier must be > 1.0 (upstream TORCH_CHECK), got {self.b_multiplier}"
            )
        if self.n_stored_hashes <= 0:
            raise ValueError(f"n_stored_hashes must be positive, got {self.n_stored_hashes}")
        if self.build_k is not None and not (0 < self.build_k <= MAX_SEARCH_K):
            raise ValueError(f"build_k must be in [1, {MAX_SEARCH_K}], got {self.build_k}")
        if self.max_sub_queries <= 0:
            raise ValueError(f"max_sub_queries must be positive, got {self.max_sub_queries}")
        if self.bloom_width not in ("configured", "upstream"):
            raise ValueError(
                f"bloom_width must be 'configured' or 'upstream', got {self.bloom_width!r}"
            )


DEFAULT_CONFIG = OfficialConfig()


_LOAD_MEMO: list = []


def _try_load():
    """One import attempt; returns the op namespace or the exception to memoize."""
    if not torch.cuda.is_available():
        return OfficialMissing(
            "torch reports no CUDA device; SilverTorch runs its scorer on CUDA only (the "
            "extension's CPU reference ops are not oracles)."
        )
    try:
        importlib.import_module(f"{PACKAGE}.ops._load_ops")  # registers torch.ops.st.*
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == PACKAGE:
            return OfficialMissing(
                f"{PACKAGE} is not installed; sync the workspace member (`uv sync "
                f"--all-packages`). Underlying error: {e}"
            )
        failure = ImportError(f"{PACKAGE} import failed on a dependency:\n{e}")
        failure.__cause__ = e
        return failure
    except ImportError as e:
        # The package is importable but its `_C` is not: the extension did not build. That
        # is a real failure, not an absent optional dependency.
        failure = ImportError(
            f"{PACKAGE} is installed but its C++/CUDA extension failed to load "
            f"(build against the running torch with `--no-build-isolation`):\n{e}"
        )
        failure.__cause__ = e
        return failure
    ops = getattr(torch.ops, NAMESPACE)
    missing = [name for name in REQUIRED_OPS if not hasattr(ops, name)]
    if missing:
        return ImportError(
            f"{PACKAGE} loaded but torch.ops.{NAMESPACE} lacks {missing}; the source does "
            "not match the one this adapter was written against (silvertorch/CHANGES.md)."
        )
    return ops


def ensure_loaded():
    """Load the ops (memoized) and return their namespace, ``torch.ops.st``.

    Raises :class:`OfficialMissing` when the package cannot run here and a plain
    ``ImportError`` when it is present but broken."""
    if not _LOAD_MEMO:
        _LOAD_MEMO.append(_try_load())
    ops = _LOAD_MEMO[0]
    if isinstance(ops, BaseException):
        raise ops
    return ops


def is_available() -> bool:
    """True iff the ops load here (flattens "missing" and "broken")."""
    try:
        ensure_loaded()
    except ImportError:
        return False
    return True


def __getattr__(name: str):
    if name == "st":
        return ensure_loaded()
    raise AttributeError(f"module 'retrieve.ops.official' has no attribute {name!r}")


from retrieve.ops.official.adapter import (  # noqa: E402  (needs the loader above)
    DevicePlans,
    ForkBloom,
    attrs_to_features,
    bloom_filtering_mask,
    bloom_full_mask,
    bloom_index_docs,
    bloom_partial_masks,
    build_bloom_index,
    default_divisor,
    dequantize_scores,
    fork_clause_attrs,
    fork_query_attrs,
    fork_search,
    fused_scores,
    max_partial_columns,
    official_probe_score,
    official_scores_full,
    pack_mask,
    padded_rows,
    parse_plans,
    plan_capacity_bytes,
    prepare_plans,
    queries_to_expressions,
    reverse_bits64,
    unpack_mask,
    unpack_partial_mask,
)

__all__ = [
    "BLOOM_OUTPUT_BIT_ORDER",
    "DevicePlans",
    "ForkBloom",
    "DEFAULT_CONFIG",
    "MASK_BIT_ORDER",
    "MAX_SEARCH_K",
    "NAMESPACE",
    "PACKAGE",
    "OfficialConfig",
    "OfficialMissing",
    "attrs_to_features",
    "bloom_filtering_mask",
    "bloom_full_mask",
    "bloom_index_docs",
    "bloom_partial_masks",
    "build_bloom_index",
    "default_divisor",
    "dequantize_scores",
    "ensure_loaded",
    "fork_clause_attrs",
    "fork_query_attrs",
    "fork_search",
    "fused_scores",
    "max_partial_columns",
    "is_available",
    "official_probe_score",
    "official_scores_full",
    "pack_mask",
    "padded_rows",
    "parse_plans",
    "plan_capacity_bytes",
    "prepare_plans",
    "queries_to_expressions",
    "reverse_bits64",
    "unpack_mask",
    "unpack_partial_mask",
]
