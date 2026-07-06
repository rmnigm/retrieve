"""Shared ``@triton.jit`` building blocks for the filter / silvertorch / linr kernels.

Every helper is a pure function of already-loaded tiles or takes fully-resolved
addressing from the caller — no helper decides its own tile shape, launch grid,
or masking policy. Callers keep their own grids, loads, and epilogue policy.

Contract (the signatures below are the Wave-1 rewrite contract):

- ``or_combine(a, b) -> a | b`` — combine_fn for ``tl.reduce`` OR-reductions.
- ``popcount_int64(x) -> int32`` — SWAR popcount over int64 lanes.
  Call site: ``linr/oporp_1bit_match_topk`` (replaces its local ``_popcount_int64``).
- ``bloom_subset_pass(qb, sigs) -> [BLOCK] int1`` — bloom subset test over
  loaded tiles ``qb`` [W] int64, ``sigs`` [BLOCK, W] int64.
  Call sites: ``silvertorch/bloom_match``, ``filters/bloom_compact``,
  ``silvertorch/codesigned_probe_score`` (standardizes on the ``qb & ~sig``
  OR-reduce form; boolean-identical to the equality + min-reduce form).
- ``clause_pass(item_attrs_ptr, is_reverse_ptr, query_attrs_ptr, ids,
  load_mask, bid, stride_in, stride_ic, stride_ia, stride_qb, stride_qc,
  C, A_MAX) -> [BLOCK] int1`` — AND-of-OR exact clause predicate, already
  ANDed with ``load_mask``.
  Call sites: ``filters/clause_mask`` / ``filters/clause_compact``
  (``ids=n_offsets``, ``load_mask=n_valid``) and
  ``silvertorch/codesigned_probe_score_exact`` (``ids=safe_ids``,
  ``load_mask=valid``).
- ``compact_store(pass_mask, ids, counts_ptr, out_ptr, bid, stride_ob,
  stride_on)`` — stream-compaction epilogue (cumsum intra-tile offsets +
  atomic_add row base + masked store).
  Call sites: ``filters/clause_compact``, ``filters/bloom_compact``.
"""

import triton
import triton.language as tl


@triton.jit
def or_combine(a, b):
    return a | b


@triton.jit
def popcount_int64(x):
    """SWAR popcount over int64 lanes → int32 (no libdevice dependency).

    torch twin: ``retrieve.layers.utils.quantize.popcount_int64`` — the
    bit-exact pairing is load-bearing (torch reference and Triton kernel must
    agree bit-for-bit; see kernels.md → Numerics)."""
    M1 = 0x5555555555555555
    M2 = 0x3333333333333333
    M4 = 0x0F0F0F0F0F0F0F0F
    H01 = 0x0101010101010101
    x = x - ((x >> 1) & M1)
    x = (x & M2) + ((x >> 2) & M2)
    x = (x + (x >> 4)) & M4
    return ((x * H01) >> 56).to(tl.int32)


@triton.jit
def bloom_subset_pass(qb, sigs):
    """``(qb & sig) == qb`` per word ⇔ ``qb & ~sig == 0``; OR-reduce over W.

    Operates on already-loaded tiles: ``qb`` [W] int64, ``sigs`` [BLOCK, W]
    int64 → [BLOCK] int1. The caller owns the sigs load mask; lanes loaded
    with ``other=0`` pass iff ``qb == 0``, identical in both algebraic forms,
    so callers must still AND the result with their own validity mask."""
    diff = qb[None, :] & ~sigs
    return tl.reduce(diff, axis=1, combine_fn=or_combine) == 0


@triton.jit
def clause_pass(
    item_attrs_ptr,
    is_reverse_ptr,
    query_attrs_ptr,
    ids,
    load_mask,
    bid,
    stride_in,
    stride_ic,
    stride_ia,
    stride_qb,
    stride_qc,
    C: tl.constexpr,
    A_MAX: tl.constexpr,
):
    """AND-of-OR exact clause predicate over a tile of item rows → [BLOCK] int1.

    ``ids`` = item row indices ([BLOCK], contiguous ``n_offsets`` or gathered
    ``safe_ids``; int32 or int64 both fine for pointer arithmetic);
    ``load_mask`` gates the attr loads (``other=-1``) and seeds ``keep``, so
    the result is already ANDed with it. ``is_reverse_ptr`` must point at int8
    storage (host wrappers do ``.to(torch.int8)``; Triton can't load native
    torch.bool) — loaded per clause and cast ``.to(tl.int1)``. ``q_c == -1``
    marks an inactive clause: always passes, overriding reverse."""
    keep = load_mask
    for c in tl.static_range(C):
        q_c = tl.load(query_attrs_ptr + bid * stride_qb + c * stride_qc)
        rev_c = tl.load(is_reverse_ptr + c).to(tl.int1)
        clause_match = tl.zeros(ids.shape, tl.int1)
        for a in tl.static_range(A_MAX):
            ia = tl.load(
                item_attrs_ptr + ids * stride_in + c * stride_ic + a * stride_ia,
                mask=load_mask,
                other=-1,
            )
            clause_match = clause_match | (ia == q_c)
        clause_match = clause_match ^ rev_c
        clause_match = clause_match | (q_c == -1)  # inactive clause always passes
        keep = keep & clause_match
    return keep


@triton.jit
def compact_store(pass_mask, ids, counts_ptr, out_ptr, bid, stride_ob, stride_on):
    """Stream compaction: cumsum intra-tile offsets + atomic_add row base.

    ``counts_ptr`` must be int64 zero-initialized; ``out_ptr`` rows must be
    host-prefilled with ``-1`` (only ``[base, base + tile_sum)`` is written,
    positions past ``counts[bid]`` keep the sentinel). ``ids`` are cast to
    int64 on store. Within-row order is unspecified (atomics across tiles)."""
    pass_int = tl.where(pass_mask, 1, 0).to(tl.int32)
    intra = tl.cumsum(pass_int, axis=0) - 1
    tile_sum = tl.sum(pass_int)
    base = tl.atomic_add(counts_ptr + bid, tile_sum.to(tl.int64))
    tl.store(
        out_ptr + bid * stride_ob + (base + intra.to(tl.int64)) * stride_on,
        ids.to(tl.int64),
        mask=pass_mask,
    )
