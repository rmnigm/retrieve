"""Meta's official SilverTorch ops (``torch.ops.st.*``) as the reference backend.

``SilverTorch(backend="official")`` keeps our phase 1 (k-means, int8 quantization, probe
selection) and routes Algorithm 1 phases 2+3 to the ops of
`meta-recsys/silvertorch <https://github.com/meta-recsys/silvertorch>`_ (pinned at
``21aa35e``; see ``docs/plans/silvertorch-official-integration.md``). This module is the
whole adapter: availability probe, layout conversion (padded IVF → the official CSR
contract), attribute → feature / expression mapping, mask packing, and the eager
scoring wrappers whose outputs go through the same ``masked_topk`` epilogue as every
other backend.

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

Everything here is **eager only** (plan D7): each op syncs the host, so there is no
``torch.compile`` / CUDA-graph path; the layer refuses to compile this backend.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from retrieve.layers.utils.quantize import quantize_int8
from retrieve.layers.utils.topk import masked_topk

BitOrder = Literal["high_first", "low_first"]
ScorePath = Literal["int32", "fp16"]
BloomPath = Literal["partial", "full"]

# Bit order the official *scorer* uses when it reads a ``filtering_bit_mask`` word (and the
# partial column masks): doc ``d`` is bit ``63 - d % 64`` of word ``d // 64`` — "lower doc id
# put at higher bits", ``bloom_index_util.cuh`` ``get_bit_64_bit_mask`` /
# ``get_next_32_bit_mask`` (lines 125-180 at 21aa35e). Read from the source, not yet
# measured: roadmap A3 (plan §10 WP-1) probes it on the A100 and
# ``tests/parity/test_official.py`` T3 pins it against the GPU. If A3 finds the other
# order, flip this constant — nothing else in the adapter hard-codes it.
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
      parity path, plan D5); ``"fp16"`` → ``divisor_for_int8 = divisor`` (a power of two
      chosen so ``127²·D / divisor ≤ 65504``), the kernel writes ``fp16(dot / divisor)``
      — the instantiation Meta ships for int8 serving, hence the timed path (plan §4.2).
    - ``divisor``: ``None`` → :func:`default_divisor` of the index width.
    - ``bloom_path``: ``"partial"`` → ``bloom_index_search_batch_return_partial_response``
      over the probed clusters + ``fused_kmean_ann_with_partial_masks`` (the paper's
      co-design, default); ``"full"`` → ``bloom_index_search_batch(return_bool_mask=False)``
      over all ``N`` + ``fused_kmean_ann(filtering_bit_mask=…)`` (the S9 ablation).
    - ``b_multiplier``: bloom width per bundle = ``max_terms_per_doc · k · b_multiplier``
      bits per doc (``bloom_indexer.cpp:52-66``); must be ``> 1.0``. Matched-memory /
      matched-FPR calibration against our ``m_bits`` is plan §4.3.
    - ``hash_k``: raw murmur hashes precomputed per query term at parse time; the search
      ANDs the first ``k`` (our ``k_hash``) *distinct* positions among them, so it must
      exceed ``k`` by a margin (README uses 7 for k=3).
    - ``build_k``: hash positions set per term at index build; ``None`` → the search
      ``k`` (README's usage; the upstream module builder passes ``hash_k`` instead, which
      sets extra bits per term — plan §1.1).
    - ``max_sub_queries``: parser fan-out bound per AND/OR node (semantics unchanged).
    - ``fast_build``: the CUDA builder's single-pass signature-free build (ignored by the
      CPU builder).
    """

    score_path: ScorePath = "fp16"
    divisor: int | None = None
    bloom_path: BloomPath = "partial"
    b_multiplier: float = 10.0
    hash_k: int = 7
    build_k: int | None = None
    max_sub_queries: int = 5
    fast_build: bool = False

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
        if self.hash_k <= 0:
            raise ValueError(f"hash_k must be positive, got {self.hash_k}")
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
            "scorer on CUDA only (the CPU reference ops are not oracles, plan §1.1)."
        )
    try:
        import silvertorch.ops._load_ops  # noqa: F401  (registers torch.ops.st.*)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "silvertorch":
            return OfficialMissing(
                "meta-recsys/silvertorch is not installed; sync the `official` extra "
                f"(`uv sync --extra official`, roadmap A2). Underlying error: {e}"
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


# --- layout: padded IVF → official CSR ----------------------------------------------------


def csr_from_assignments(
    assignments: Tensor, n_lists: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cluster assignments ``[N]`` → ``(sort_perm, inv_perm, cluster_offsets, cluster_sizes)``.

    ``sort_perm[j]`` is the original id of the item at cluster-sorted position ``j``
    (``argsort(assignments)``, stable so a cluster's items keep id order); ``inv_perm`` is
    its inverse; ``cluster_offsets`` is the int64 ``[n_lists+1]`` CSR the official
    scorer indexes with ``cluster_ids``; ``cluster_sizes`` int64 ``[n_lists]``."""
    cluster_sizes = torch.bincount(assignments, minlength=n_lists)
    sort_perm = torch.argsort(assignments, stable=True)
    inv_perm = torch.empty_like(sort_perm)
    inv_perm[sort_perm] = torch.arange(sort_perm.numel(), device=sort_perm.device)
    offsets = torch.zeros(n_lists + 1, dtype=torch.int64, device=assignments.device)
    offsets[1:] = cluster_sizes.cumsum(0)
    return sort_perm, inv_perm, offsets, cluster_sizes


def default_divisor(d: int) -> int:
    """Smallest power of two ``v`` with ``127² · d / v ≤ 65504`` — the fp16 score path
    divides the int32 dot by ``v`` before the cast, so this is the smallest divisor at
    which a full-range int8 dot cannot overflow fp16 (plan §4.2: 16 at D=64, 32 at
    D=128, 64 at D=256). A power of two keeps the division exact."""
    need = (127 * 127 * d) / FP16_MAX
    return 1 << max(0, math.ceil(math.log2(need)))


def padded_rows(max_tensor_size_per_row: int) -> int:
    """Output width the scorer allocates: ``max_tensor_size_per_row`` rounded up to the warp."""
    return (max_tensor_size_per_row + WARP - 1) // WARP * WARP


# --- attributes → features / expressions ------------------------------------------------


def attrs_to_features(attrs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``[N, C, A]`` int64 narrow attrs (``-1`` pad) → the official jagged feature layout:
    ``feature_ids`` int32 ``[C]`` (the clause index is the feature id — the same
    ``(clause, value)`` keying as our salt), ``feature_offsets`` int64 ``[N·C+1]``
    (CSR over ``(doc, clause)`` slots), ``feature_values`` int64 ``[nnz]`` in ``(doc,
    clause, slot)`` order. Pass **cluster-sorted** attrs so the index's doc space is the
    scorer's."""
    if attrs.dim() != 3:
        raise ValueError(f"attrs must be [N, C, A_max], got shape {tuple(attrs.shape)}")
    n, c, _ = attrs.shape
    attrs = attrs.long()
    valid = attrs != -1
    counts = valid.sum(dim=-1).reshape(-1)  # [N·C]
    offsets = torch.zeros(n * c + 1, dtype=torch.int64, device=attrs.device)
    offsets[1:] = counts.cumsum(0)
    values = attrs.reshape(-1)[valid.reshape(-1)]  # row-major → (doc, clause, slot) order
    feature_ids = torch.arange(c, dtype=torch.int32, device=attrs.device)
    return feature_ids, offsets.contiguous(), values.contiguous()


def build_bloom_index(
    attrs: Tensor,
    *,
    b_multiplier: float,
    build_k: int,
    fast_build: bool = False,
) -> tuple[Tensor, Tensor]:
    """``bloom_index_build`` over cluster-sorted attrs → ``(bloom_index int64 [W],
    bundle_b_offsets int64 [n_bundles+1])`` on ``attrs.device`` (the CUDA builder needs
    CUDA inputs; the CPU builder is a single-threaded loop). ``W = Σ_bundles B_bundle ·
    32`` words, i.e. ``B_bundle / 8`` bytes per doc."""
    st = ensure_loaded()
    if not b_multiplier > 1.0:
        raise ValueError(f"b_multiplier must be > 1.0, got {b_multiplier}")
    if not 0 < build_k <= MAX_SEARCH_K:
        raise ValueError(f"build_k must be in [1, {MAX_SEARCH_K}], got {build_k}")
    feature_ids, feature_offsets, feature_values = attrs_to_features(attrs)
    bloom_index, bundle_b_offsets = st.bloom_index_build(
        feature_ids, feature_offsets, feature_values, float(b_multiplier), int(build_k), fast_build
    )
    return bloom_index.contiguous(), bundle_b_offsets.contiguous()


def bloom_index_docs(bundle_b_offsets: Tensor) -> int:
    """Doc space of an official bloom index: ``n_bundles · 2048`` (≥ N)."""
    return (int(bundle_b_offsets.numel()) - 1) * DOCS_PER_BUNDLE


def queries_to_expressions(
    query_clause_attrs: Tensor, clause_is_reverse: Tensor | None = None
) -> list[str]:
    """``[B, C]`` int64 query attrs (``-1`` inactive) → one expression per row in the
    official DSL: active clauses joined with ``AND`` as ``"{c}:{v}"``, reverse clauses as
    ``"NOT {c}:{v}"``, no active clause → ``""`` (``EMPTY`` = match all,
    ``expression_query_parser.cpp:400-405``). Host-side by nature (a string parser)."""
    if query_clause_attrs.dim() != 2:
        raise ValueError(
            f"query_clause_attrs must be [B, C], got {tuple(query_clause_attrs.shape)}"
        )
    c = query_clause_attrs.shape[1]
    rows = query_clause_attrs.long().tolist()
    rev = (
        [False] * c if clause_is_reverse is None else [bool(x) for x in clause_is_reverse.tolist()]
    )
    if len(rev) != c:
        raise ValueError(f"clause_is_reverse must have {c} entries, got {len(rev)}")
    out = []
    for row in rows:
        terms = []
        for ci, v in enumerate(row):
            if v == -1:
                continue
            if v < 0:
                raise ValueError(f"attribute values must be >= 0 (or -1 inactive), got {v}")
            terms.append(f"NOT {ci}:{v}" if rev[ci] else f"{ci}:{v}")
        out.append(" AND ".join(terms))
    return out


@functools.lru_cache(maxsize=4096)
def _parse_plans_cached(
    expressions: tuple[str, ...], hash_k: int, max_sub_queries: int
) -> tuple[Tensor, Tensor]:
    st = ensure_loaded()
    # `silvertorch_ks` is accepted and ignored upstream (expression_query_parser.cpp:395).
    ks = torch.ones(len(expressions), dtype=torch.int64)
    _max_stack, plans = st.parse_expression_query_batch(
        list(expressions), ks, int(hash_k), True, int(max_sub_queries)
    )
    data, offsets = plans
    return data.contiguous(), offsets.contiguous()


def parse_plans(
    expressions: list[str], hash_k: int, max_sub_queries: int = 5
) -> tuple[Tensor, Tensor]:
    """``(plans_data int8, plans_offsets int64)`` on **CPU** for a batch of expressions,
    LRU-cached on the string tuple (plans depend only on ``(strings, hash_k,
    max_sub_queries)``; the search ops decode them host-side per call and would copy
    CUDA-resident plans back to the CPU first, so keeping them on CPU is the cheaper
    choice — plan §3/§4.4)."""
    return _parse_plans_cached(tuple(expressions), int(hash_k), int(max_sub_queries))


# --- masks ------------------------------------------------------------------------------


def _shifts(bit_order: BitOrder, device: torch.device) -> Tensor:
    if bit_order == "high_first":
        return 63 - torch.arange(64, dtype=torch.int64, device=device)
    if bit_order == "low_first":
        return torch.arange(64, dtype=torch.int64, device=device)
    raise ValueError(f"bit_order must be 'high_first' or 'low_first', got {bit_order!r}")


def pack_mask(mask: Tensor, bit_order: BitOrder = MASK_BIT_ORDER) -> Tensor:
    """``[B, N]`` bool → ``[B, ceil(N/64)]`` int64 with doc ``d`` at bit ``63 - d % 64``
    (``"high_first"``) or ``d % 64`` (``"low_first"``) of word ``d // 64``. The default is
    the scorer's read order, so the result is a ``filtering_bit_mask``."""
    if mask.dim() != 2 or mask.dtype != torch.bool:
        raise ValueError(f"mask must be [B, N] bool, got {tuple(mask.shape)} {mask.dtype}")
    b, n = mask.shape
    w = (n + 63) // 64
    padded = torch.zeros(b, w * 64, dtype=torch.bool, device=mask.device)
    padded[:, :n] = mask
    bits = padded.view(b, w, 64).to(torch.int64)
    # Disjoint bit patterns add without carries, so the sum is the OR (bit 63 included:
    # two's-complement wrap-around is exactly the int64 bit pattern we want).
    return (bits << _shifts(bit_order, mask.device)).sum(dim=-1)


def pack_mask_high_first(mask: Tensor) -> Tensor:
    """``pack_mask(mask, "high_first")`` — the name plan §5.1 uses."""
    return pack_mask(mask, "high_first")


def unpack_mask(words: Tensor, n: int, bit_order: BitOrder = MASK_BIT_ORDER) -> Tensor:
    """Inverse of :func:`pack_mask`: ``[B, W]`` int64 → ``[B, n]`` bool."""
    b, w = words.shape
    if n > w * 64:
        raise ValueError(f"n={n} exceeds {w} words × 64 bits")
    bits = (words.unsqueeze(-1) >> _shifts(bit_order, words.device)) & 1
    return bits.view(b, w * 64)[:, :n].bool()


def _i64(x: int) -> int:
    return x - (1 << 64) if x >= (1 << 63) else x


def reverse_bits64(x: Tensor) -> Tensor:
    """Bit ``i`` ↔ bit ``63 - i`` within each int64 (converts a high-first word into a
    low-first one and back). Arithmetic right shifts pull in sign bits; every mask
    below has its top bits clear, so they are dropped."""
    if x.dtype != torch.int64:
        raise TypeError(f"reverse_bits64 expects int64, got {x.dtype}")
    x = ((x >> 1) & _i64(0x5555555555555555)) | ((x & _i64(0x5555555555555555)) << 1)
    x = ((x >> 2) & _i64(0x3333333333333333)) | ((x & _i64(0x3333333333333333)) << 2)
    x = ((x >> 4) & _i64(0x0F0F0F0F0F0F0F0F)) | ((x & _i64(0x0F0F0F0F0F0F0F0F)) << 4)
    x = ((x >> 8) & _i64(0x00FF00FF00FF00FF)) | ((x & _i64(0x00FF00FF00FF00FF)) << 8)
    x = ((x >> 16) & _i64(0x0000FFFF0000FFFF)) | ((x & _i64(0x0000FFFF0000FFFF)) << 16)
    x = ((x >> 32) & _i64(0x00000000FFFFFFFF)) | ((x & _i64(0x00000000FFFFFFFF)) << 32)
    return x


def unpack_partial_mask(
    column_counts_cumsum: Tensor,
    first_item_offset_in_column: Tensor,
    column_mask_response: Tensor,
    selected_cluster_lengths: Tensor,
    max_size: int,
    bit_order: BitOrder = BLOOM_OUTPUT_BIT_ORDER,
) -> Tensor:
    """Decode a ``_return_partial_response`` triple into ``[B, P·max_size]`` bool in the
    padded-IVF slot order (``slot = p · max_size + j``, ``j`` the doc's position within
    its probed cluster; slots past the cluster length are ``False``). For the ``(query,
    probe)`` range ``idx``, its words are ``column_mask_response[cumsum[idx-1] :
    cumsum[idx]]`` and doc ``j`` is bit position ``first_offset[idx] + j``
    (``fused_kmean_ann_cuda.cu:661-673``). Used by the tests (bloom ⊇ exact, FPR) — the
    scorer consumes the packed form directly."""
    b, p = selected_cluster_lengths.shape
    device = column_mask_response.device
    cumsum = column_counts_cumsum.to(device=device, dtype=torch.int64)
    first = first_item_offset_in_column.to(device=device, dtype=torch.int64)
    lengths = selected_cluster_lengths.reshape(-1).to(device=device, dtype=torch.int64)
    start = torch.cat([cumsum.new_zeros(1), cumsum[:-1]])  # [B·P]
    j = torch.arange(max_size, dtype=torch.int64, device=device)  # [max_size]
    pos = first[:, None] + j[None, :]  # [B·P, max_size]
    in_range = j[None, :] < lengths[:, None]
    word_idx = (start[:, None] + pos // 64).clamp_(0, max(int(column_mask_response.numel()) - 1, 0))
    words = (
        column_mask_response[word_idx]
        if column_mask_response.numel() > 0
        else torch.zeros_like(word_idx)
    )
    shift = (63 - pos % 64) if bit_order == "high_first" else (pos % 64)
    bits = ((words >> shift) & 1).bool() & in_range
    return bits.view(b, p * max_size)


# --- scoring ------------------------------------------------------------------------------


def fused_scores(
    q_codes: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    cluster_sizes: Tensor,
    item_codes_sorted: Tensor,
    max_tensor_size_per_row: int,
    *,
    divisor: int = -1,
    filtering_bit_mask: Tensor | None = None,
    partial: tuple[Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Raw call into the official scorer: ``(scores, indices)`` as the op returns them —
    int32 or fp16 ``[B, padded_rows]``, int32 sorted-table positions with ``-1`` pads and
    filtered-out slots. ``partial`` selects ``fused_kmean_ann_with_partial_masks``;
    otherwise ``fused_kmean_ann`` with an optional full-``N`` ``filtering_bit_mask``."""
    st = ensure_loaded()
    if q_codes.dtype != torch.int8 or item_codes_sorted.dtype != torch.int8:
        raise TypeError("official scorer takes int8 queries and int8 embeddings")
    cluster_ids = probe_ids.to(torch.int64).contiguous()
    cluster_length = cluster_sizes[cluster_ids].to(torch.int64).contiguous()
    common = (
        cluster_offsets.to(torch.int64).contiguous(),
        cluster_ids,
        cluster_length,
        item_codes_sorted.contiguous(),
        q_codes.contiguous(),
        int(max_tensor_size_per_row),
    )
    if partial is not None:
        if filtering_bit_mask is not None:
            raise ValueError("pass either partial masks or a filtering_bit_mask, not both")
        counts_cumsum, first_offset, column_results = partial
        return st.fused_kmean_ann_with_partial_masks(
            *common,
            counts_cumsum.contiguous(),
            first_offset.contiguous(),
            column_results.contiguous(),
            invalid_index_value=-1,
            divisor_for_int8=int(divisor),
        )
    return st.fused_kmean_ann(
        *common,
        filtering_bit_mask=None if filtering_bit_mask is None else filtering_bit_mask.contiguous(),
        invalid_index_value=-1,
        divisor_for_int8=int(divisor),
    )


def dequantize_scores(
    raw: Tensor,
    indices: Tensor,
    q_scales: Tensor,
    global_scale: float | Tensor,
    sort_perm: Tensor,
    divisor: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Host epilogue: ``(scores fp32 [B, M], ids int64 [B, M], valid bool [B, M])``.

    int32 path (``divisor == -1``): ``(dot.float() · q_scale[b]) · global_scale`` — the
    same two left-associated fp32 multiplies as the Triton kernel and the torch path, so
    scores are bit-identical (plan D5). fp16 path: the kernel wrote ``fp16(dot /
    divisor)``; ``score = raw.float() · (divisor · q_scale[b] · global_scale)``. Slots the
    op did not write (pads, filtered docs) carry ``indices == -1``; the caller masks them
    to ``-inf`` / ``-1`` through ``masked_topk``."""
    valid = indices >= 0
    ids = torch.where(
        valid, sort_perm[indices.long().clamp_min(0)], indices.new_full((), -1).long()
    )
    if raw.dtype == torch.int32:
        if divisor != -1:
            raise ValueError("int32 scores imply divisor_for_int8 == -1")
        scores = raw.to(torch.float32) * q_scales.unsqueeze(1) * global_scale
    else:
        scores = raw.to(torch.float32) * (divisor * q_scales * global_scale).unsqueeze(1)
    return scores, ids, valid


def official_scores_full(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    cluster_sizes: Tensor,
    item_codes_sorted: Tensor,
    sort_perm: Tensor,
    global_scale: float | Tensor,
    max_tensor_size_per_row: int,
    *,
    score_path: ScorePath = "int32",
    divisor: int | None = None,
    filtering_bit_mask: Tensor | None = None,
    partial: tuple[Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Phases 2+3 on the official ops with our quantization and epilogue, **before** the
    top-k: ``(scores fp32 [B, M], ids int64 [B, M], valid bool [B, M])`` with ``M =
    padded_rows(max_tensor_size_per_row)``. Slot order is the op's (warp-aligned segments
    per probe, then remainders) — fine for a top-k, not a concatenation."""
    q_codes, q_scales = quantize_int8(query)
    div = (
        -1
        if score_path == "int32"
        else (default_divisor(item_codes_sorted.shape[1]) if divisor is None else int(divisor))
    )
    raw, idx = fused_scores(
        q_codes,
        probe_ids,
        cluster_offsets,
        cluster_sizes,
        item_codes_sorted,
        max_tensor_size_per_row,
        divisor=div,
        filtering_bit_mask=filtering_bit_mask,
        partial=partial,
    )
    return dequantize_scores(raw, idx, q_scales, global_scale, sort_perm, div)


def official_probe_score(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    cluster_sizes: Tensor,
    item_codes_sorted: Tensor,
    sort_perm: Tensor,
    global_scale: float | Tensor,
    k: int,
    max_tensor_size_per_row: int,
    *,
    score_path: ScorePath = "int32",
    divisor: int | None = None,
    filtering_bit_mask: Tensor | None = None,
    partial: tuple[Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """:func:`official_scores_full` + the shared ``masked_topk`` epilogue → ``(ids int64
    [B, k], scores fp32 [B, k])`` with ``-1`` / ``-inf`` pads, like every other backend."""
    scores, ids, valid = official_scores_full(
        query,
        probe_ids,
        cluster_offsets,
        cluster_sizes,
        item_codes_sorted,
        sort_perm,
        global_scale,
        max_tensor_size_per_row,
        score_path=score_path,
        divisor=divisor,
        filtering_bit_mask=filtering_bit_mask,
        partial=partial,
    )
    return masked_topk(scores, k, valid=valid, gather_ids=ids)


# --- bloom search ---------------------------------------------------------------------


def bloom_partial_masks(
    bloom_index: Tensor,
    bundle_b_offsets: Tensor,
    plans: tuple[Tensor, Tensor],
    selected_cluster_offsets: Tensor,
    selected_cluster_lengths: Tensor,
    k: int,
    hash_k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Phase 2 of the paper's co-design: the official bloom evaluated **only over the
    probed clusters**, returned as the ``(column_counts_cumsum int32 [B·P],
    first_item_offset_in_column int8 [B·P], column_mask_response int64 [Σ])`` triple
    ``fused_kmean_ann_with_partial_masks`` consumes. ``selected_cluster_offsets`` /
    ``lengths`` are int64 ``[B, P]`` (``cluster_offsets[probe_ids]``,
    ``cluster_sizes[probe_ids]``)."""
    st = ensure_loaded()
    if not 0 < k <= MAX_SEARCH_K:
        raise ValueError(
            f"k must be in [1, {MAX_SEARCH_K}] (MAX_K_V2, unchecked upstream), got {k}"
        )
    data, offsets = plans
    return st.bloom_index_search_batch_return_partial_response(
        bloom_index,
        bundle_b_offsets,
        data,
        offsets,
        selected_cluster_offsets.to(torch.int64).contiguous(),
        selected_cluster_lengths.to(torch.int64).contiguous(),
        int(k),
        int(hash_k),
    )


def bloom_full_mask(
    bloom_index: Tensor,
    bundle_b_offsets: Tensor,
    plans: tuple[Tensor, Tensor],
    k: int,
    hash_k: int,
    *,
    return_bool_mask: bool = False,
) -> Tensor:
    """``bloom_index_search_batch`` over the whole index: bool ``[B, n_bundles·2048]`` or
    the packed int64 ``[B, n_bundles·32]`` (high-first words)."""
    st = ensure_loaded()
    if not 0 < k <= MAX_SEARCH_K:
        raise ValueError(
            f"k must be in [1, {MAX_SEARCH_K}] (MAX_K_V2, unchecked upstream), got {k}"
        )
    data, offsets = plans
    return st.bloom_index_search_batch(
        bloom_index, bundle_b_offsets, data, offsets, int(k), int(hash_k), bool(return_bool_mask)
    )


def bloom_filtering_mask(
    bloom_index: Tensor,
    bundle_b_offsets: Tensor,
    plans: tuple[Tensor, Tensor],
    k: int,
    hash_k: int,
) -> Tensor:
    """The full-``N`` official bloom mask in the scorer's ``filtering_bit_mask`` order —
    the packed search output, bit-reversed per word only if :data:`MASK_BIT_ORDER` ever
    differs from :data:`BLOOM_OUTPUT_BIT_ORDER` (both high-first at 21aa35e)."""
    packed = bloom_full_mask(
        bloom_index, bundle_b_offsets, plans, k, hash_k, return_bool_mask=False
    )
    if MASK_BIT_ORDER != BLOOM_OUTPUT_BIT_ORDER:
        packed = reverse_bits64(packed)
    return packed.contiguous()


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
    "csr_from_assignments",
    "default_divisor",
    "dequantize_scores",
    "ensure_loaded",
    "fused_scores",
    "is_available",
    "official_probe_score",
    "official_scores_full",
    "pack_mask",
    "pack_mask_high_first",
    "padded_rows",
    "parse_plans",
    "queries_to_expressions",
    "reverse_bits64",
    "unpack_mask",
    "unpack_partial_mask",
]
