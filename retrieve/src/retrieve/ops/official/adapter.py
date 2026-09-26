"""The official-backend adapter proper: attributes → features / expressions, mask packing, the
bloom searches and the scoring wrappers over ``torch.ops.st.*`` (loader, ``OfficialConfig`` and
the upstream constants live in the package ``__init__``). Everything is eager only.
"""

from __future__ import annotations

import functools
import math

import torch
from torch import Tensor

from retrieve.functional import masked_topk
from retrieve.indexing.quantize import quantize_int8
from retrieve.ops.official import (
    BLOOM_OUTPUT_BIT_ORDER,
    DOCS_PER_BUNDLE,
    FP16_MAX,
    MASK_BIT_ORDER,
    MAX_SEARCH_K,
    WARP,
    BitOrder,
    ScorePath,
    ensure_loaded,
)

# --- layout: padded IVF → official CSR ----------------------------------------------------
#
# The cluster-sorted CSR (``sort_perm`` / ``inv_perm`` / ``cluster_offsets`` /
# ``cluster_sizes``) is ``retrieve.indexing.csr_layout`` of the same assignment as the padded
# layout, so both arms share one slot order; nothing here re-derives it.


def default_divisor(d: int) -> int:
    """Smallest power of two ``v`` with ``127² · d / v ≤ 65504`` — the fp16 score path
    divides the int32 dot by ``v`` before the cast, so this is the smallest divisor at
    which a full-range int8 dot cannot overflow fp16. A power of two keeps the division exact."""
    need = (127 * 127 * d) / FP16_MAX
    return 1 << max(0, math.ceil(math.log2(need)))


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


def queries_to_expressions(
    query_clause_attrs: Tensor, clause_is_reverse: Tensor | None = None
) -> list[str]:
    """``[B, C]`` int64 query attrs (``-1`` inactive) → one expression per row in the
    official DSL: active clauses joined with ``AND`` as ``"{c}:{v}"``, reverse clauses as
    ``"NOT {c}:{v}"``, no active clause → ``""`` (``EMPTY`` = match all,
    ``expression_query_parser.cpp:400-405``). Host-side by nature (a string parser).
    ``clause_is_reverse`` is a test seam: the library never passes it (bloom mode rejects
    reverse clauses at ``register_index``); ``NOT`` is exercised by T4 only."""
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


def _parse_plans(
    expressions: tuple[str, ...], hash_k: int, max_sub_queries: int
) -> tuple[Tensor, Tensor]:
    """One call into the CPU parser — the uncached body of :func:`parse_plans`."""
    st = ensure_loaded()
    # `silvertorch_ks` is accepted and ignored upstream (expression_query_parser.cpp:395).
    ks = torch.ones(len(expressions), dtype=torch.int64)
    _max_stack, plans = st.parse_expression_query_batch(
        list(expressions), ks, int(hash_k), True, int(max_sub_queries)
    )
    data, offsets = plans
    return data.contiguous(), offsets.contiguous()


_parse_plans_cached = functools.lru_cache(maxsize=4096)(_parse_plans)


def parse_plans(
    expressions: list[str], hash_k: int, max_sub_queries: int = 5, *, cache: bool = True
) -> tuple[Tensor, Tensor]:
    """``(plans_data int8, plans_offsets int64)`` on **CPU** for a batch of expressions.
    Plans depend only on ``(strings, hash_k, max_sub_queries)``; the search ops decode them
    host-side per call and would copy CUDA-resident plans back to the CPU first, so keeping
    them on CPU is the cheaper choice. ``cache=True`` memoises on the string
    tuple (LRU, 4096 batches); ``cache=False`` parses every call — ≈ 59 µs at B=16 (plan
    §13.2) — which is what a timing run must use so the parse is not hidden behind a
    replayed batch (``OfficialConfig.cache_plans``)."""
    if not cache:
        return _parse_plans(tuple(expressions), int(hash_k), int(max_sub_queries))
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
    scores are bit-identical. fp16 path: the kernel wrote ``fp16(dot /
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


# --- test support: probes used by tests/parity/test_official.py, not by the adapter ---------


def padded_rows(max_tensor_size_per_row: int) -> int:
    """Output width the scorer allocates: ``max_tensor_size_per_row`` rounded up to the warp."""
    return (max_tensor_size_per_row + WARP - 1) // WARP * WARP


def bloom_index_docs(bundle_b_offsets: Tensor) -> int:
    """Doc space of an official bloom index: ``n_bundles · 2048`` (≥ N)."""
    return (int(bundle_b_offsets.numel()) - 1) * DOCS_PER_BUNDLE


def unpack_mask(words: Tensor, n: int, bit_order: BitOrder = MASK_BIT_ORDER) -> Tensor:
    """Inverse of :func:`pack_mask`: ``[B, W]`` int64 → ``[B, n]`` bool."""
    b, w = words.shape
    if n > w * 64:
        raise ValueError(f"n={n} exceeds {w} words × 64 bits")
    bits = (words.unsqueeze(-1) >> _shifts(bit_order, words.device)) & 1
    return bits.view(b, w * 64)[:, :n].bool()


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
