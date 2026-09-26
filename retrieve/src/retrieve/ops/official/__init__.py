"""Meta's official SilverTorch ops (``torch.ops.st.*``) as the reference backend.

``SilverTorch(backend="official")`` keeps our phase 1 (k-means, int8 quantization, probe
selection) and routes Algorithm 1 phases 2+3 to the ops of
`meta-recsys/silvertorch <https://github.com/meta-recsys/silvertorch>`_ (pinned at
``21aa35e``; docs/system/kernels.md § official). This package is the
whole adapter: the availability probe, ``OfficialConfig`` and the upstream constants here;
attribute → feature / expression mapping, mask packing and the eager scoring wrappers whose
outputs go through the same ``masked_topk`` epilogue as every other backend in ``adapter``
(re-exported below). ``retrieve.ops.official.st`` is ``torch.ops.st`` once loaded.

Op schemas matched against the upstream registrations (``TORCH_LIBRARY_FRAGMENT(st,
…)`` at 21aa35e):

- ``fused_kmean_ann(Tensor cluster_offsets, Tensor cluster_ids, Tensor cluster_length,
  Tensor embeddings, Tensor queries, int max_tensor_size_per_row,
  Tensor? filtering_bit_mask=None, int invalid_index_value=-1, int divisor_for_int8=-1,
  Tensor? filtering_bit_index=None, Tensor? per_embedding_scale=None) -> (Tensor,
  Tensor)`` — ``fused_kmean_ann.cpp:400-413``. ``cluster_offsets`` int64 ``[n_lists+1]``,
  ``cluster_ids`` / ``cluster_length`` int64 ``[B, P]``, ``embeddings`` int8 ``[N, D]``
  **cluster-sorted**, ``queries`` int8 ``[B, D]``; returns ``scores`` (int32 when
  ``divisor_for_int8 == -1``, fp16 otherwise) and ``indices`` int32, both ``[B,
  round_up(max_tensor_size_per_row, 32)]``, pads at the dtype minimum /
  ``invalid_index_value``.
- ``fused_kmean_ann_with_partial_masks(…same five tensors…, int max_tensor_size_per_row,
  Tensor partial_mask_column_counts_cumsum, Tensor partial_mask_first_item_offset_in_column,
  Tensor partial_mask_column_results, int invalid_index_value=-1, int divisor_for_int8=-1,
  Tensor? filtering_bit_index=None, Tensor? per_embedding_scale=None, Tensor?
  cluster_warp_size=None, Tensor? cluster_warp_rounded_length_cumsum=None, Tensor?
  cluster_remaining_length_cumsum=None, Tensor? cluster_warp_size_cumsum=None, int
  total_cluster_rounded_warps=0, int total_cluster_remaining_warps=0) -> (Tensor,
  Tensor)`` — ``fused_kmean_ann.cpp:415-436``.
- ``bloom_index_build(Tensor feature_ids, Tensor feature_offsets, Tensor feature_values,
  float b_multiplier, int k, bool fast_build=False) -> (Tensor, Tensor)`` —
  ``bloom_indexer.cpp:115-125``; int32 ``[F]``, int64 ``[N·F+1]``, int64 ``[nnz]``,
  ``b_multiplier > 1.0``; returns ``bloom_index`` int64 ``[W]`` and
  ``bundle_b_offsets`` int64 ``[n_bundles+1]``.
- ``parse_expression_query_batch(str[] expressions, Tensor silvertorch_ks, int
  bloom_hash_k, bool return_query_plan=True, int max_sub_queries=5) -> (int, Tensor[])``
  — ``expression_query_parser.cpp:456-465``; CPU-only (the CUDA registration forwards
  to it); ``silvertorch_ks`` is unused upstream; returns ``(max_stack_size,
  [plans_data int8, plans_offsets int64])``.
- ``bloom_index_search_batch(Tensor bloom_index, Tensor bloom_bundle_b_offsets, Tensor
  bloom_query_plans_data, Tensor bloom_query_plans_offsets, int k, int hash_k, bool
  return_bool_mask=True) -> Tensor`` — ``bloom_index_search.cpp:549-558``; bool ``[B,
  n_bundles·2048]`` or packed int64 ``[B, n_bundles·32]``.
- ``bloom_index_search_batch_return_partial_response(…same four…, Tensor
  selected_cluster_offsets, Tensor selected_cluster_lengths, int k, int hash_k, Tensor?
  query_plan_index=None) -> (Tensor, Tensor, Tensor)`` — ``bloom_index_search.cpp:560-571``;
  int64 ``[B, P]`` doc-start / length per probe; returns ``column_counts_cumsum`` int32
  ``[B·P]``, ``first_item_offset_in_column`` int8 ``[B·P]``, ``column_mask_response``
  int64 ``[Σ columns]``.

Everything here is **eager only**: each op syncs the host, so there is no
``torch.compile`` / CUDA-graph path; the layer refuses to compile this backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

BitOrder = Literal["high_first", "low_first"]
ScorePath = Literal["int32", "fp16"]
BloomPath = Literal["partial", "full"]

# Bit order the official *scorer* uses when it reads a ``filtering_bit_mask`` word (and the
# partial column masks): doc ``d`` is bit ``63 - d % 64`` of word ``d // 64`` — "lower doc id
# put at higher bits", ``bloom_index_util.cuh`` ``get_bit_64_bit_mask`` /
# ``get_next_32_bit_mask`` (lines 125-180 at 21aa35e). Measured three independent ways, all
# HIGH-first (kernels.md, "Bit order — measured, high-first").
# ``tests/parity/test_official.py`` T3 pins it (``OFFICIAL_BIT_ORDER``); nothing else in
# the adapter hard-codes it.
MASK_BIT_ORDER: BitOrder = "high_first"

# Order of the packed output of ``bloom_index_search_batch(return_bool_mask=False)``:
# ``store<bool>`` unpacks the raw word from bit 63 downwards
# (``bloom_index_search_cuda.cu:552-556``), so the raw int64 word is high-first. This one is
# unambiguous in the source; T3 round-trips it anyway.
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
)


class OfficialMissing(ImportError):
    """The official package cannot run here: not installed, or no CUDA device. Distinct
    from a plain ``ImportError``, which means it *is* installed and its extension is
    broken — tests skip on the former and fail on the latter (``require_official``)."""


@dataclass(frozen=True)
class OfficialConfig:
    """Knobs of the official backend that have no counterpart on the other backends.

    - ``score_path``: ``"int32"`` → ``divisor_for_int8=-1``, the raw int32 dot, host
      epilogue ``(dot·q_scale)·global_scale`` **bit-identical** to Triton / torch (the
      parity path); ``"fp16"`` → ``divisor_for_int8 = divisor`` (a power of two
      chosen so ``127²·D / divisor ≤ 65504``), the kernel writes ``fp16(dot / divisor)``
      — the instantiation Meta ships for int8 serving, hence the timed path.
    - ``divisor``: ``None`` → :func:`default_divisor` of the index width.
    - ``bloom_path``: ``"partial"`` → ``bloom_index_search_batch_return_partial_response``
      over the probed clusters + ``fused_kmean_ann_with_partial_masks`` (the paper's
      co-design, default); ``"full"`` → ``bloom_index_search_batch(return_bool_mask=False)``
      over all ``N`` + ``fused_kmean_ann(filtering_bit_mask=…)`` (the S9 ablation).
    - ``b_multiplier``: bloom width per bundle = ``max_terms_per_doc · k · b_multiplier``
      bits per doc (``bloom_indexer.cpp:52-66``); must be ``> 1.0``. Matched-memory /
      matched-FPR calibration against our ``m_bits`` is roadmap D3.
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
    - ``cache_plans``: memoise :func:`parse_plans` on the expression tuple (default). The
      CPU expression parse costs ≈ 59 µs per call at B=16 —
      10–20 % of an eager bloom forward — and a benchmark that replays one fixed batch
      would hide it behind the cache after the first call. **Timing must use
      ``cache_plans=False``** (every forward pays the parse, as a serving path with fresh
      queries does) **or report both, labelled.** Results are identical either way.
    """

    score_path: ScorePath = "fp16"
    divisor: int | None = None
    bloom_path: BloomPath = "partial"
    b_multiplier: float = 10.0
    build_k: int | None = None
    max_sub_queries: int = 5
    fast_build: bool = False
    cache_plans: bool = True
    n_stored_hashes: int = 7

    def __post_init__(self) -> None:
        if self.score_path not in ("int32", "fp16"):
            raise ValueError(f"score_path must be 'int32' or 'fp16', got {self.score_path!r}")
        if self.bloom_path not in ("partial", "full"):
            raise ValueError(f"bloom_path must be 'partial' or 'full', got {self.bloom_path!r}")
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


DEFAULT_CONFIG = OfficialConfig()


# --- availability ---------------------------------------------------------------------

_LOAD_MEMO: object | None = None


def _try_load():
    """One import attempt; returns ``torch.ops.st`` or the exception to memoize."""
    if not torch.cuda.is_available():
        return OfficialMissing(
            "torch reports no CUDA device; the official SilverTorch backend runs its "
            "scorer on CUDA only (the CPU reference ops are not oracles)."
        )
    try:
        import silvertorch.ops._load_ops  # noqa: F401  (registers torch.ops.st.*)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "silvertorch":
            return OfficialMissing(
                "meta-recsys/silvertorch is not installed; sync the `official` extra "
                f"(`uv sync --extra official`). Underlying error: {e}"
            )
        failure = ImportError(f"silvertorch import failed on a dependency:\n{e}")
        failure.__cause__ = e
        return failure
    except ImportError as e:
        # `silvertorch` is importable but `silvertorch._C` is not: the extension did
        # not build. That is a real failure, not an absent optional dependency.
        failure = ImportError(
            "silvertorch is installed but its C++/CUDA extension failed to load "
            f"(build against the running torch with `--no-build-isolation`):\n{e}"
        )
        failure.__cause__ = e
        return failure
    st = torch.ops.st
    missing = [name for name in REQUIRED_OPS if not hasattr(st, name)]
    if missing:
        return ImportError(
            f"silvertorch loaded but torch.ops.st lacks {missing}; the pinned sha "
            "does not match the one this adapter was written against (21aa35e)."
        )
    return st


def ensure_loaded():
    """Load the official ops (memoized) and return the ``torch.ops.st`` namespace.

    Raises :class:`OfficialMissing` when the package cannot run here and a plain
    ``ImportError`` when it is present but broken."""
    global _LOAD_MEMO
    if _LOAD_MEMO is None:
        _LOAD_MEMO = _try_load()
    if isinstance(_LOAD_MEMO, BaseException):
        raise _LOAD_MEMO
    return _LOAD_MEMO


def is_available() -> bool:
    """True iff the official ops load here (flattens "missing" and "broken")."""
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
    attrs_to_features,
    bloom_filtering_mask,
    bloom_full_mask,
    bloom_index_docs,
    bloom_partial_masks,
    build_bloom_index,
    default_divisor,
    dequantize_scores,
    fused_scores,
    official_probe_score,
    official_scores_full,
    pack_mask,
    padded_rows,
    parse_plans,
    queries_to_expressions,
    reverse_bits64,
    unpack_mask,
    unpack_partial_mask,
)

__all__ = [
    "BLOOM_OUTPUT_BIT_ORDER",
    "DEFAULT_CONFIG",
    "MASK_BIT_ORDER",
    "MAX_SEARCH_K",
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
    "fused_scores",
    "is_available",
    "official_probe_score",
    "official_scores_full",
    "pack_mask",
    "padded_rows",
    "parse_plans",
    "queries_to_expressions",
    "reverse_bits64",
    "unpack_mask",
    "unpack_partial_mask",
]
