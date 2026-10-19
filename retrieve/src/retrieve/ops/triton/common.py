"""Shared ``@triton.jit`` building blocks of the clause and LiNR kernels. Every helper is a pure
function of already-loaded tiles, or takes fully-resolved addressing from the caller: no helper
decides its own tile shape, launch grid or masking policy. Per-helper semantics and the call-site
map: docs/system/kernels.md § Shared kernel helpers."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


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
def popcount_int64(x):
    """Popcount over int64 lanes → int32: the hardware ``POPC`` (``__nv_popcll``). Exact, so it
    equals the torch twin ``retrieve.functional.popcount_int64`` bit for bit."""
    return libdevice.popc(x)


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
    """AND-of-OR exact clause predicate over a tile of item rows → [BLOCK] int1. ``ids`` are item
    rows relative to ``item_attrs_ptr`` (kernels.md § Addressing); ``load_mask`` gates the attr
    loads (``other=-1``) and seeds the result. ``is_reverse_ptr`` points at int8 storage (Triton
    cannot load torch.bool). ``q_c == -1`` marks an inactive clause: it always passes, so a
    uniform branch skips its loads (kernels.md § Shared kernel helpers, "Inactive clauses")."""
    keep = load_mask
    for c in tl.static_range(C):
        q_c = tl.load(query_attrs_ptr + bid * stride_qb + c * stride_qc)
        if q_c != -1:
            rev_c = tl.load(is_reverse_ptr + c).to(tl.int1)
            clause_match = tl.zeros(ids.shape, tl.int1)
            for a in tl.static_range(A_MAX):
                ia = tl.load(
                    item_attrs_ptr + ids * stride_in + c * stride_ic + a * stride_ia,
                    mask=load_mask,
                    other=-1,
                )
                clause_match = clause_match | (ia == q_c)
            keep = keep & (clause_match ^ rev_c)
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
    to the row offset once the counts are scanned (kernels.md § ``clause_compact``)."""
    # The scan first: Triton lays the tile out for the first reduction it meets, and the layout
    # it picks for a bare tl.sum makes the predicate's loads 1.8x slower at B=1.
    compact_store(pass_mask, ids, tile_id * BLOCK_N, scratch_ptr, bid, stride_sb, 1, WIDE)
    tile_sum = tl.sum(tl.where(pass_mask, 1, 0).to(tl.int32))
    tl.store(tile_counts_ptr + bid * stride_tb + tile_id, tile_sum.to(tl.int64))


@triton.jit
def compact_scatter_kernel(
    scratch_ptr,  # [B, T * BLOCK_N] int32, compact_stash's tile-local id runs
    tile_counts_ptr,  # [B, T] int64
    offsets_ptr,  # [B, T] int64 exclusive scan of the tile counts
    out_ptr,  # [B, N] int64, uninitialised
    tiles_y,
    stride_sb,
    stride_tb,
    stride_ob,
    stride_on,
    BLOCK_N: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Phase 3 of the two-phase compaction, one program per ``(row, tile)`` on the predicate
    kernel's 3-D grid: move the tile's ``count`` stashed ids to ``out[bid, offset :]``. Nothing
    past ``counts[bid]`` is written (kernels.md § ``clause_compact``). Launched by
    ``_host.compact_finish``; the only ``@triton.jit`` here that is a kernel, not a callee."""
    bid = tl.program_id(0)
    tile_id = tl.program_id(2) * tiles_y + tl.program_id(1)
    lane = tl.arange(0, BLOCK_N)
    count = tl.load(tile_counts_ptr + bid * stride_tb + tile_id)
    base = tl.load(offsets_ptr + bid * stride_tb + tile_id)
    run = row_base(scratch_ptr, bid, stride_sb, WIDE) + tile_id * BLOCK_N + lane
    ids = tl.load(run, mask=lane < count)
    out_row = row_base(out_ptr, bid, stride_ob, WIDE)
    tl.store(out_row + (base + lane) * stride_on, ids.to(tl.int64), mask=lane < count)
