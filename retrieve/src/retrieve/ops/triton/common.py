"""Shared ``@triton.jit`` building blocks for the filter / silvertorch / linr kernels.

Every helper is a pure function of already-loaded tiles, or takes fully-resolved
addressing from the caller. **No helper decides its own tile shape, launch grid, or
masking policy** — callers keep their own grids, loads, and epilogue policy. That
invariant is what lets one helper serve kernels with very different launch shapes.

Helpers: ``row_base``, ``tile_rows``, ``or_combine``, ``popcount_int64``, ``bloom_subset_pass``,
``clause_pass``, ``compact_store``, ``compact_stash``; plus ``compact_scatter_kernel``, the one
launched kernel here (the predicate-free second phase both compaction ops share, driven by
``_host.compact_finish``). Per-helper semantics and the call-site map live in
docs/system/kernels.md § Shared kernel helpers.
"""

import triton
import triton.language as tl


@triton.jit
def row_base(ptr, row, stride, WIDE: tl.constexpr):
    """``ptr + row * stride``, the product in int64 when ``WIDE`` (kernels.md § Addressing). The
    assumes make both operands zero-extensions, so the product lowers to one ``mul.wide.u32``
    instead of a full 64-bit multiply."""
    if WIDE:
        tl.assume(row >= 0)
        tl.assume(stride >= 0)
        return ptr + row.to(tl.int64) * stride
    return ptr + row * stride


@triton.jit
def tile_rows(ptr, row0, lane, stride, WIDE: tl.constexpr):
    """``(base, ids)`` with item row ``row0 + lane`` at ``base + ids * stride``: the table pointer
    and absolute int32 ids when narrow (the baseline addressing, fastest at ``B = 1``), the
    pointer moved to ``row0`` in int64 and lane ids when ``WIDE`` (kernels.md § Addressing)."""
    if WIDE:
        return row_base(ptr, row0, stride, WIDE), lane
    return ptr, row0 + lane


@triton.jit
def or_combine(a, b):
    return a | b


@triton.jit
def popcount_int64(x):
    """SWAR popcount over int64 lanes → int32 (no libdevice dependency).

    torch twin: ``retrieve.functional.popcount_int64`` — the
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

    ``ids`` = item rows relative to ``item_attrs_ptr`` ([BLOCK]: int32 lanes over a pointer the
    caller moved to its tile with ``row_base``, or absolute int64 gathered ``safe_ids``;
    kernels.md § Addressing);
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
def compact_store(pass_mask, ids, base, out_ptr, bid, stride_ob, stride_on, WIDE: tl.constexpr):
    """Stream-compaction store at a caller-supplied row base: ``tl.cumsum`` intra-tile ranks and
    a masked store of ``ids`` (cast to the pointee type) at ``base + rank``. Ascending ``ids``
    in, ascending out."""
    pass_int = tl.where(pass_mask, 1, 0).to(tl.int32)
    intra = tl.cumsum(pass_int, axis=0) - 1
    tl.store(
        row_base(out_ptr, bid, stride_ob, WIDE) + (base + intra) * stride_on,
        ids.to(out_ptr.dtype.element_ty),
        mask=pass_mask,
    )


@triton.jit
def compact_stash(
    pass_mask,
    ids,
    tile_counts_ptr,
    scratch_ptr,
    bid,
    tile_id,
    stride_tb,
    stride_sb,
    BLOCK_N: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Phase 1 of the two-phase compaction: the tile's survivor count to ``tile_counts[bid,
    tile_id]`` and its surviving ``ids`` compacted into the tile's own slot range
    ``scratch[bid, tile_id * BLOCK_N : +count]`` (int32), for ``compact_scatter_kernel`` to move
    to the row offset once the counts are scanned. Same epilogue as the pre-L3 kernel with the
    ``atomic_add`` row base replaced by the fixed tile-local base — the shape that keeps the
    predicate kernel at its one-pass speed (plan §7: a count-only epilogue is 1.8× slower at
    B=1, a packed bitmask 1.7×)."""
    # The scan first: Triton lays the tile out for the first reduction it meets, and the layout
    # it picks for a bare tl.sum makes the predicate's loads 1.8x slower at B=1 (plan §7).
    compact_store(pass_mask, ids, tile_id * BLOCK_N, scratch_ptr, bid, stride_sb, 1, WIDE)
    tile_sum = tl.sum(tl.where(pass_mask, 1, 0).to(tl.int32))
    tl.store(tile_counts_ptr + bid * stride_tb + tile_id, tile_sum.to(tl.int64))


@triton.jit
def compact_scatter_kernel(
    scratch_ptr,  # [B, T * BLOCK_N] int32, compact_stash's tile-local id runs
    tile_counts_ptr,  # [B, T] int64
    offsets_ptr,  # [B, T] int64 exclusive scan of the tile counts
    counts_ptr,  # [B] int64 row totals
    out_ptr,  # [B, N] int64, uninitialised
    N,
    tiles_y,
    stride_sb,
    stride_tb,
    stride_ob,
    stride_on,
    BLOCK_N: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Phase 3 of the two-phase compaction, one program per ``(row, tile)`` on the same 3-D grid
    as the predicate kernel: move the tile's ``count`` stashed ids to ``out[bid, offset :]``, and
    write ``-1`` over the tile's own slice of the row's tail ``[counts[bid], N)``. The runs cover
    ``[0, counts[bid])``, so the two writes never overlap. Launched by
    ``_host.compact_finish``; the only ``@triton.jit`` here that is a kernel rather than a
    callee."""
    bid = tl.program_id(0)
    tile_id = tl.program_id(2) * tiles_y + tl.program_id(1)
    lane = tl.arange(0, BLOCK_N)
    count = tl.load(tile_counts_ptr + bid * stride_tb + tile_id)
    base = tl.load(offsets_ptr + bid * stride_tb + tile_id)
    run = row_base(scratch_ptr, bid, stride_sb, WIDE) + tile_id * BLOCK_N + lane
    ids = tl.load(run, mask=lane < count)
    out_row = row_base(out_ptr, bid, stride_ob, WIDE)
    tl.store(out_row + (base + lane) * stride_on, ids.to(tl.int64), mask=lane < count)
    slot = tile_id * BLOCK_N + lane
    tail = (slot >= tl.load(counts_ptr + bid)) & (slot < N)
    tl.store(out_row + slot * stride_on, tl.full([BLOCK_N], -1, tl.int64), mask=tail)
