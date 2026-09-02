"""CuTe DSL port of the CUDA C++ SilverTorch backend (Algorithm 1, ``partial_bloom``,
phases 2+3) — a one-to-one translation of ``../cuda/codesigned_probe_score.cu``.

Three kernels: two phase-2 filters that write the SAME 1-bit-per-item mask layout —
``cps_bloom_mask_kernel`` evaluates bloom against a transposed, cluster-major bit
matrix, ``cps_clause_mask_kernel`` evaluates the exact AND-of-OR clause predicate over
the probed ids — and ``cps_score_kernel``, which scores int8 code rows through ``dp4a``
gated by whichever mask it is handed. The scorer is filter-agnostic on purpose: a
filter is a bit-vector. Lane layout, id prefetch, predicated loads, carry loop, ballot
packing and launch geometry are the ``.cu``'s; its template parameters are
``Constexpr`` arguments here, and ``compile_*`` memoizes one ``cute.compile`` per
constexpr tuple. Runtime shapes are ``Int64``/``Int32`` scalars, so a new shape never
recompiles.

Design, layout contract, perf model and the full constraint list (shared with the C++
backend): docs/system/kernels.md § codesigned_probe_score_cuda.

BIT-EXACTNESS: results must be identical to the C++ and Triton kernels; tests/parity
asserts it with ``torch.equal``. Same ``dp4a`` order over a lane's four words, same
exact integer segment reduction (``redux.sync.add`` — sm_80+ only; the C++ sub-sm_80
butterfly is not ported), same two left-associated fp32 multiplies, same ``-inf``. Do
not reassociate the epilogue and pass no fast-math options to ``cute.compile``. UNROLL
changes only how many items a segment keeps in flight, never the arithmetic.

Importing this module imports ``cutlass``; ``codesigned_probe_score_cute.py`` (the
host side) imports it lazily and owns all shape / dtype validation — there are no
device-side checks, and ids are trusted to be in range exactly as in the ``.cu``.
"""

import ctypes
import math
import threading
import warnings

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Constexpr, Float32, Int32, Int64, Uint8
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cute.runtime import make_ptr

MASK_THREADS = 256  # kMaskKernelThreads: block size of both phase-2 kernels

# --- device helpers (straight-line only: plain functions are not preprocessed) -------


def dp4a(a: Int32, b: Int32, c: Int32) -> Int32:
    """``__dp4a(a, b, c)``: four int8 MACs into an int32 (one ``IDP.4A.S8.S8``)."""
    return cute.arch.inline_ptx(
        "dp4a.s32.s32 {$w0}, {$r0}, {$r1}, {$r2};",
        write_only_types=[Int32],
        read_only_args=[a, b, c],
    )


def load_int4(ptr: cute.Pointer, cop: str | None) -> list[Int32]:
    """One 16-byte vector load through an ``Int32`` pointer, as four words. ``cop="cs"``
    is ``__ldcs`` (evict-first, ``LDG.E.EF.128``) for the once-touched code rows."""
    v = cute.arch.load(ptr, ir.VectorType.get([4], Int32.mlir_type), cop=cop)
    return [Int32(llvm.extractelement(v, Int32(i).ir_value())) for i in range(4)]


def cttz64(x: Int64) -> Int32:
    """``__ffsll(x) - 1`` for ``x != 0``: index of the lowest set bit."""
    return Int32(Int64(mlir_math.cttz(x.ir_value())))


def seg_reduce_add(v: Int32, seg_mask: Int32) -> Int32:
    """``__reduce_add_sync(mask, v)``: exact int32 sum over the lanes of ``seg_mask``."""
    return cute.arch.warp_redux_sync(v, "add", mask_and_clamp=seg_mask)


def ld(ptr: cute.Pointer, off):
    """Scalar ``ptr[off]``. The offset goes on the pointer, not on a size-1 layout
    (indexing that with a dynamic coordinate is not a pointer offset)."""
    return cute.make_tensor(ptr + off, cute.make_layout(1))[0]


def st(ptr: cute.Pointer, off, value) -> None:
    """Scalar ``ptr[off] = value``."""
    cute.make_tensor(ptr + off, cute.make_layout(1))[0] = value


# --- phase 2: bloom mask --------------------------------------------------------------


@cute.kernel
def cps_bloom_mask_kernel(
    qb: cute.Pointer,  # [B, W] int64 query signature
    sigs_t: cute.Pointer,  # [W*64, n_lists*wpc] int64 transposed index
    probe_ids: cute.Pointer,  # [B, n_probe] int64 probed cluster ids
    mask: cute.Pointer,  # [B, n_probe*wpc] int64 out
    sig_row_words: Int64,  # n_lists * wpc
    mask_words: Int64,  # n_probe * wpc
    n_probe: Int64,
    wpc: Int32,
    W: Int32,
):
    """One thread produces one output mask word (= the verdict for 64 items);
    consecutive threads walk consecutive words of one cluster span, so the ``sigs_t``
    row reads coalesce. Iterates ONLY the set bits of the query signature.

    Unlike the clause kernel, the pad tail of a span's last word is NOT forced to 0:
    an empty query signature short-circuits to the ``~0`` identity. Harmless — phase 3
    only ever addresses ``slot < max_size`` — but do not start relying on a zero tail."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    b = Int64(bidy)
    idx = Int64(bidx) * bdim + tidx
    if idx < mask_words:
        pi = idx // wpc  # probe ordinal
        w_in_c = idx - pi * wpc  # word within the cluster span
        cluster = ld(probe_ids, b * n_probe + pi)
        col = cluster * wpc + w_in_c
        result = ~Int64(0)  # AND identity: no set query bits => all pass
        for qw in cutlass.range(W):
            bits = ld(qb, b * W + qw)
            while bits != 0:  # set-bit walk: __ffsll + bits &= bits - 1
                j = cttz64(bits)
                bits = bits & (bits - 1)
                m = Int64(qw) * 64 + j
                result = result & ld(sigs_t, m * sig_row_words + col)  # 64 items per AND
        st(mask, b * mask_words + idx, result)


# --- phase 2 (exact): clause mask -----------------------------------------------------


@cute.kernel
def cps_clause_mask_kernel(
    flat_items: cute.Pointer,  # [B, P] int64, -1 = pad
    item_attrs: cute.Pointer,  # [N, C, A_MAX] int64
    is_reverse: cute.Pointer,  # [C] uint8 view of torch.bool storage
    query_attrs: cute.Pointer,  # [B, C] int64, -1 = inactive
    mask: cute.Pointer,  # [B, n_probe*wpc] int64 out
    P: Int64,
    max_size: Int64,  # items per cluster span (P = n_probe * max_size)
    mask_words: Int64,
    wpc: Int32,
    n_clauses: Int32,
    a_max: Int32,
    C: Constexpr[int],
    A: Constexpr[int],
):
    """Same output layout as the bloom kernel. One thread owns one slot, one warp one
    32-bit half of an output word: a ballot packs the 32 verdicts and lane 0 stores the
    half through a 32-bit view of the int64 word (little-endian: low half at 2*word).
    Ids come from ``flat_items``, so a half reads 32 consecutive ids in one run.

    ``(C, A)`` constexpr: all ``C*A`` attribute words of a slot load into registers
    before the first compare; ``(0, 0)`` is the runtime-bound fallback. The predicate
    is bit-identical to ``common.clause_pass``: ``keep = id >= 0 AND over clauses of
    [(OR over A values of attr == q_c) XOR rev_c OR q_c == -1]``; padding slots and
    the pad tail of a span's last word get bit 0."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    lane = cute.arch.lane_idx()
    warp = tidx // 32
    # Warp-uniform (block and warp ordinal only), so the whole warp either skips the
    # word or reaches the ballot below together.
    half_idx = Int64(bidx) * (bdim // 32) + warp
    word_idx = half_idx >> 1
    if word_idx < mask_words:
        h = Int32(half_idx & 1)  # low / high half of the word
        b = Int64(bidy)
        pi = word_idx // wpc  # probe ordinal
        w_in_c = word_idx - pi * wpc  # word within the cluster span
        slot = w_in_c * 64 + h * 32 + lane
        span = b * P + pi * max_size  # flat_items base of the span
        q_row = b * n_clauses  # query_attrs base of this query
        bit = Boolean(False)
        if cutlass.const_expr(C > 0):
            # Query side first: independent of the id and warp-uniform (broadcast).
            q = [ld(query_attrs, q_row + c) for c in range(C)]
            rev = [ld(is_reverse, c) != 0 for c in range(C)]
            if slot < max_size:  # pad tail of the last word stays 0
                idv = ld(flat_items, span + slot)
                if idv >= 0:  # cluster padding stays 0
                    v = [ld(item_attrs, idv * (C * A) + i) for i in range(C * A)]
                    keep = Boolean(True)
                    for c in cutlass.range_constexpr(C):
                        match = Boolean(False)
                        for a in cutlass.range_constexpr(A):
                            match = match | (v[c * A + a] == q[c])
                        # XOR the reverse flag first, then let the inactive sentinel
                        # override it — the order clause_pass uses.
                        keep = keep & ((match != rev[c]) | (q[c] == -1))
                    bit = keep
        else:
            if slot < max_size:  # pad tail of the last word stays 0
                idv = ld(flat_items, span + slot)
                if idv >= 0:  # cluster padding stays 0
                    keep = Boolean(True)
                    for c in cutlass.range(n_clauses):
                        q_c = ld(query_attrs, q_row + c)
                        rev_c = ld(is_reverse, c) != 0
                        attr = (idv * n_clauses + c) * a_max
                        match = Boolean(False)
                        for a in cutlass.range(a_max):
                            match = match | (ld(item_attrs, attr + a) == q_c)
                        keep = keep & ((match != rev_c) | (q_c == -1))
                    bit = keep
        # Outside every lane-divergent branch: all 32 lanes reach the ballot.
        packed = cute.arch.vote_ballot_sync(bit)
        if lane == 0:
            mask32 = cute.recast_ptr(mask, dtype=Int32)
            st(mask32, (b * mask_words + word_idx) * 2 + h, packed)


# --- phase 3: masked dp4a scoring -----------------------------------------------------


@cute.jit
def _ld_id(flat_items: cute.Pointer, row: Int64, p_u: Int64, warp_end: Int64) -> Int64:
    """Segment-uniform predicated id load: a dead tail item (``p_u >= warp_end``) is
    never addressed and reads as pad (-1)."""
    idv = Int64(-1)
    if p_u < warp_end:
        idv = ld(flat_items, row + p_u)
    return idv


@cute.jit
def _mask_bit(mask: cute.Pointer, mask_row: Int64, cl: Int64, slot: Int64, wpc: Int32):
    """Mask bit of slot ``slot`` in cluster span ``cl``: word ``cl*wpc + slot//64``,
    bit ``slot%64``. The word covers 64 consecutive slots, so it stays L1-resident
    across a segment's iterations."""
    word = ld(mask, mask_row + cl * wpc + (slot >> 6))
    return ((word >> (slot & 63)) & 1) != 0


@cute.jit
def _mask_keep(
    mask: cute.Pointer,
    mask_row: Int64,
    keep: Boolean,
    slot_u: Int64,
    cl_u: Int64,
    max_size: Int64,
    wpc: Int32,
) -> Boolean:
    """Phase (a) mask test for one item: ``slot_u`` is the segment's base slot plus
    ``u*SPW`` (< STEP), so its ``(cluster, slot)`` is the base's after a handful of
    carry subtractions — no per-item division."""
    if keep:  # segment-uniform
        while slot_u >= max_size:
            slot_u -= max_size
            cl_u += 1
        keep = _mask_bit(mask, mask_row, cl_u, slot_u, wpc)
    return keep


@cute.jit
def _ld_row(item_codes: cute.Pointer, keep: Boolean, idv: Int64, row_words, chunk) -> list[Int32]:
    """Predicated 16-byte row-chunk load: a rejected item costs no row traffic, and its
    id is clamped to row 0 so the (unused) address stays inside the table."""
    row = cutlass.select_(keep, idv, Int64(0))
    rw = [Int32(0)] * 4
    if keep:  # segment-uniform
        rw = load_int4(item_codes + (row * row_words + chunk), cop="cs")
    return rw


@cute.kernel
def cps_score_kernel(
    q_codes: cute.Pointer,  # [B, D] int8 as Int32 words, 16 B-aligned rows
    q_scales: cute.Pointer,  # [B] fp32
    mask: cute.Pointer,  # [B, mask_words] int64 (HAS_MASK only)
    flat_items: cute.Pointer,  # [B, P] int64, -1 = pad
    item_codes: cute.Pointer,  # [N, D] int8 as Int32 words, 16 B-aligned rows
    out: cute.Pointer,  # [B, P] fp32
    global_scale: Float32,
    P: Int64,
    max_size: Int64,  # items per cluster span (P = n_probe * max_size)
    mask_words: Int64,
    wpc: Int32,
    items_per_warp: Int32,
    SEG: Constexpr[int],
    HAS_MASK: Constexpr[bool],
    UNROLL: Constexpr[int],
):
    """Each lane owns one 16-byte chunk of a row, so a segment of ``SEG = D/16`` lanes
    reads a row as one coalesced int4 request and keeps its four query words in
    registers; no shared memory. ``SPW = 32/SEG`` items advance per warp-instruction
    and UNROLL items per segment stay in flight per iteration. D=64 → SEG=4,
    D=128 → SEG=8, D=256 → SEG=16."""
    D = SEG * 16
    SPW = 32 // SEG  # segments (items) per warp-instruction
    STEP = SPW * UNROLL  # items one warp covers per iteration

    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    lane = cute.arch.lane_idx()
    warp = tidx // 32
    seg = lane // SEG  # segment index within the warp
    sl = lane % SEG  # lane index within the segment = 16 B chunk
    seg_mask = ((1 << SEG) - 1) << (seg * SEG)
    b = Int64(bidy)

    # Block-lifetime registers: this lane's four query words + the row scale.
    qv = load_int4(q_codes + (b * (D // 4) + sl * 4), cop=None)
    q_scale = ld(q_scales, b)
    # `if constexpr`: without a mask `mask` is a 1x1 dummy, so even forming (never
    # mind reading) mask + b*mask_words would be out of range.
    mask_row = Int64(0)
    if cutlass.const_expr(HAS_MASK):
        mask_row = b * mask_words

    # The warp owns a contiguous item range; segments interleave inside it so the
    # SPW ids/stores in flight per instruction stay adjacent (one sector).
    warp_start = Int64(bidx) * (items_per_warp * (bdim // 32)) + Int64(warp) * items_per_warp
    warp_end = cutlass.min(warp_start + items_per_warp, P)
    p_first = warp_start + seg
    ids_row = b * P

    # (cluster, slot) of the segment's base item, tracked incrementally so the hot
    # loop carries no 64-bit division: one divide here, then bounded carry loops of
    # at most STEP subtractions. Correct for any max_size >= 1, including
    # max_size < STEP, where one step crosses several cluster spans.
    cl = Int64(0)
    slot = Int64(0)
    if cutlass.const_expr(HAS_MASK):
        cl = p_first // max_size
        slot = p_first - cl * max_size

    # Ids are prefetched one iteration ahead so the id -> row chain overlaps the
    # current row gathers.
    ids_next = [_ld_id(flat_items, ids_row, p_first + u * SPW, warp_end) for u in range(UNROLL)]

    for p0 in range(p_first, warp_end, STEP):
        # (a) ids and keep verdicts, then the next iteration's id prefetch, then the
        # mask test. -1 covers both cluster padding and the dead tail.
        ids = ids_next
        keep = [idv >= 0 for idv in ids]
        ids_next = [
            _ld_id(flat_items, ids_row, p0 + STEP + u * SPW, warp_end) for u in range(UNROLL)
        ]
        if cutlass.const_expr(HAS_MASK):
            keep = [
                _mask_keep(mask, mask_row, keep[u], slot + u * SPW, cl, max_size, wpc)
                for u in range(UNROLL)
            ]
        # (b) all UNROLL row gathers, issued back to back before the first dot so
        # their latencies overlap; predicated on the segment-uniform keep flag.
        rw = [_ld_row(item_codes, keep[u], ids[u], D // 4, sl * 4) for u in range(UNROLL)]
        # (c) one dot + epilogue + store per item, in item order. Identical arithmetic
        # at every UNROLL, which is what keeps parity config-free.
        for u in cutlass.range_constexpr(UNROLL):
            p_u = p0 + u * SPW
            if p_u < warp_end:  # segment-uniform: the whole segment skips a dead tail
                score = Float32(-math.inf)
                if keep[u]:
                    acc = dp4a(rw[u][0], qv[0], Int32(0))
                    acc = dp4a(rw[u][1], qv[1], acc)
                    acc = dp4a(rw[u][2], qv[2], acc)
                    acc = dp4a(rw[u][3], qv[3], acc)
                    acc = seg_reduce_add(acc, seg_mask)
                    # Two fp32 multiplies, left-associated — the Triton epilogue.
                    score = Float32(acc) * q_scale * global_scale
                if sl == 0:
                    st(out, ids_row + p_u, score)
        if cutlass.const_expr(HAS_MASK):
            slot += STEP
            while slot >= max_size:
                slot -= max_size
                cl += 1


@cute.kernel
def cps_score_kernel_generic(
    q_codes: cute.Pointer,  # [B, D] int8 as Int32 words, 4 B-aligned
    q_scales: cute.Pointer,
    mask: cute.Pointer,
    flat_items: cute.Pointer,
    item_codes: cute.Pointer,  # [N, D] int8 as Int32 words, 4 B-aligned
    out: cute.Pointer,
    global_scale: Float32,
    P: Int64,
    max_size: Int64,
    mask_words: Int64,
    wpc: Int32,
    items_per_warp: Int32,
    d_words: Int32,  # D / 4
    HAS_MASK: Constexpr[bool],
):
    """Fallback for any ``D % 4 == 0`` outside {64, 128, 256} or misaligned row bases:
    full-warp segments with a runtime word loop; query words reload per item. No
    UNROLL parameter — a runtime word loop already has D/128 loads in flight per
    lane."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    lane = cute.arch.lane_idx()
    warp = tidx // 32
    b = Int64(bidy)

    q_row = b * d_words
    q_scale = ld(q_scales, b)
    mask_row = Int64(0)
    if cutlass.const_expr(HAS_MASK):
        mask_row = b * mask_words

    warp_start = Int64(bidx) * (items_per_warp * (bdim // 32)) + Int64(warp) * items_per_warp
    warp_end = cutlass.min(warp_start + items_per_warp, P)
    ids_row = b * P

    for p in range(warp_start, warp_end):
        idv = ld(flat_items, ids_row + p)
        keep = idv >= 0
        # `if constexpr` so neither the mask_row arithmetic nor the p / max_size divide
        # exists in the no-filter specialization (where max_size is 0).
        if cutlass.const_expr(HAS_MASK):
            if keep:
                cl = p // max_size  # non-negative operands: floor == C truncation
                keep = _mask_bit(mask, mask_row, cl, p - cl * max_size, wpc)
        score = Float32(-math.inf)
        if keep:
            row = idv * d_words
            acc = Int32(0)
            for w in range(lane, d_words, 32):
                acc = dp4a(ld(item_codes, row + w), ld(q_codes, q_row + w), acc)
            acc = cute.arch.warp_redux_sync(acc, "add")  # full warp
            score = Float32(acc) * q_scale * global_scale
        if lane == 0:
            st(out, ids_row + p, score)


# --- host launchers (the .cu's <<<grid, block>>> lines) and the compile cache ---------


@cute.jit
def _launch_bloom_mask(
    qb: cute.Pointer,
    sigs_t: cute.Pointer,
    probe_ids: cute.Pointer,
    mask: cute.Pointer,
    sig_row_words: Int64,
    mask_words: Int64,
    n_probe: Int64,
    wpc: Int32,
    W: Int32,
    B: Int32,
    stream: cuda.CUstream,
):
    cps_bloom_mask_kernel(
        qb, sigs_t, probe_ids, mask, sig_row_words, mask_words, n_probe, wpc, W
    ).launch(
        grid=[(mask_words + MASK_THREADS - 1) // MASK_THREADS, B, 1],
        block=[MASK_THREADS, 1, 1],
        stream=stream,
    )


@cute.jit
def _launch_clause_mask(
    flat_items: cute.Pointer,
    item_attrs: cute.Pointer,
    is_reverse: cute.Pointer,
    query_attrs: cute.Pointer,
    mask: cute.Pointer,
    P: Int64,
    max_size: Int64,
    mask_words: Int64,
    wpc: Int32,
    n_clauses: Int32,
    a_max: Int32,
    B: Int32,
    C: Constexpr[int],
    A: Constexpr[int],
    stream: cuda.CUstream,
):
    words_per_block = MASK_THREADS // 64  # one warp per 32-bit half word
    cps_clause_mask_kernel(
        flat_items,
        item_attrs,
        is_reverse,
        query_attrs,
        mask,
        P,
        max_size,
        mask_words,
        wpc,
        n_clauses,
        a_max,
        C,
        A,
    ).launch(
        grid=[(mask_words + words_per_block - 1) // words_per_block, B, 1],
        block=[MASK_THREADS, 1, 1],
        stream=stream,
    )


@cute.jit
def _launch_score(
    q_codes: cute.Pointer,
    q_scales: cute.Pointer,
    mask: cute.Pointer,
    flat_items: cute.Pointer,
    item_codes: cute.Pointer,
    out: cute.Pointer,
    global_scale: Float32,
    P: Int64,
    max_size: Int64,
    mask_words: Int64,
    wpc: Int32,
    B: Int32,
    block_p: Int32,
    num_warps: Int32,
    SEG: Constexpr[int],
    HAS_MASK: Constexpr[bool],
    UNROLL: Constexpr[int],
    stream: cuda.CUstream,
):
    items_per_warp = block_p // num_warps
    cps_score_kernel(
        q_codes,
        q_scales,
        mask,
        flat_items,
        item_codes,
        out,
        global_scale,
        P,
        max_size,
        mask_words,
        wpc,
        items_per_warp,
        SEG,
        HAS_MASK,
        UNROLL,
    ).launch(grid=[(P + block_p - 1) // block_p, B, 1], block=[num_warps * 32, 1, 1], stream=stream)


@cute.jit
def _launch_score_generic(
    q_codes: cute.Pointer,
    q_scales: cute.Pointer,
    mask: cute.Pointer,
    flat_items: cute.Pointer,
    item_codes: cute.Pointer,
    out: cute.Pointer,
    global_scale: Float32,
    P: Int64,
    max_size: Int64,
    mask_words: Int64,
    wpc: Int32,
    B: Int32,
    block_p: Int32,
    num_warps: Int32,
    d_words: Int32,
    HAS_MASK: Constexpr[bool],
    stream: cuda.CUstream,
):
    items_per_warp = block_p // num_warps
    cps_score_kernel_generic(
        q_codes,
        q_scales,
        mask,
        flat_items,
        item_codes,
        out,
        global_scale,
        P,
        max_size,
        mask_words,
        wpc,
        items_per_warp,
        d_words,
        HAS_MASK,
    ).launch(grid=[(P + block_p - 1) // block_p, B, 1], block=[num_warps * 32, 1, 1], stream=stream)


def gmem_ptr(dtype, address: int, align: int) -> cute.Pointer:
    """Global-memory pointer for a launch argument. ``align`` is a promise the caller
    must keep (16 for the int4 row loads)."""
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


_streams: dict[int, cuda.CUstream] = {}


def cu_stream(handle: int) -> cuda.CUstream:
    """``cuda.CUstream`` wrapper for a torch stream handle, cached (4.5 µs to build);
    only the ``_Launch`` fallback path needs one."""
    stream = _streams.get(handle)
    if stream is None:
        stream = _streams[handle] = cuda.CUstream(handle)
    return stream


# One cute.compile per constexpr tuple, memoized. The example arguments bake in only
# *types* — pointer dtype / address space / alignment and the scalar widths — so one
# compiled callable serves every shape and every buffer. Constexpr parameters are
# stripped from its signature: call it with the runtime arguments + stream only, in
# the launcher's order.
#
# The compiled function itself is device-agnostic, but calling it directly would bind
# it to whichever device is current at its *first* call (its lazily created default
# executor holds one device context). So the callables handed out are per-device
# executors (`JitCompiledFunction.to(device)`, wrapped in `_Launch`), keyed on
# `(key, device)`: one compile still serves every GPU, and a launch on `cuda:1` never
# reuses `cuda:0`'s context.
_compiled: dict[tuple, object] = {}
_executors: dict[tuple, object] = {}
_I64P, _F32P = (Int64, 8), (Float32, 4)
_SCORE_SCALARS = [Float32, Int64, Int64, Int64, Int32, Int32, Int32, Int32]


def _current_device() -> int:
    """Ordinal of the device whose context is current (the host guards launches with
    ``torch.cuda.device(...)``, so this is the tensors' device)."""
    err, dev = cuda.cuCtxGetDevice()
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxGetDevice failed: {err}")
    return int(dev)


_CTYPES = {Float32: ctypes.c_float, Int64: ctypes.c_int64, Int32: ctypes.c_int32}


class _Launch:
    """A compiled specialization bound to one device, callable with plain Python values:
    pointer *addresses* (ints), scalars, and the CUDA stream's integer handle, in
    launcher order.

    ``JitExecutor.__call__`` re-adapts every argument on every call — ``typing.cast``,
    an owning ctypes cell per value, adapter lookups — which is ~25 µs for the scorer's
    15 arguments against ~4 µs for the whole C++ launcher. The compiled host function's
    ABI is the MLIR C-interface one (an array of pointers to each argument's storage),
    so this keeps one set of ctypes cells per thread, writes the values in place and
    calls ``run_compiled_program`` directly. The packing is verified once against the
    DSL's own ``generate_execution_args`` on the compile-time example arguments; if a
    DSL release ever packs differently the wrapper falls back to the ordinary call path
    (same semantics, slower)."""

    __slots__ = ("_executor", "_ptrs", "_scalar_ctypes", "_tls", "_fallback")

    def __init__(
        self, executor, example_args: list, ptrs: list[tuple[type, int]], scalars: list[type]
    ):
        self._executor = executor
        self._ptrs = ptrs
        self._scalar_ctypes = [_CTYPES[t] for t in scalars]
        self._tls = threading.local()
        self._fallback = not self._packing_matches(example_args)

    def _packing_matches(self, example_args: list) -> bool:
        n_ptr, n_sc = len(self._ptrs), len(self._scalar_ctypes)
        try:
            exe, _keepalive = self._executor.generate_execution_args(*example_args)
        except Exception:
            return False
        if len(exe) != n_ptr + n_sc + 1:
            return False
        addr = lambda x: x if isinstance(x, int) else x.value
        # Pointer and stream slots point at a cell holding the address/handle; scalar
        # slots point at the scalar's own cell. The examples are address 256, value 1
        # and the null stream (`_compile`).
        for i in range(n_ptr):
            if ctypes.c_void_p.from_address(addr(exe[i])).value != 256:
                return False
        for i, ct in enumerate(self._scalar_ctypes):
            if ct.from_address(addr(exe[n_ptr + i])).value != 1:
                return False
        return (ctypes.c_void_p.from_address(addr(exe[-1])).value or 0) == 0

    def _cells(self):
        tls = self._tls
        cells = getattr(tls, "cells", None)
        if cells is None:
            cells = [ctypes.c_void_p(0) for _ in self._ptrs]
            cells += [ct(0) for ct in self._scalar_ctypes]
            cells.append(ctypes.c_void_p(0))
            tls.cells = cells
            tls.exe_args = [ctypes.addressof(c) for c in cells]
        return cells, tls.exe_args

    def __call__(self, *args) -> None:
        if self._fallback:
            n = len(self._ptrs)
            converted = [
                gmem_ptr(dt, a, al) for (dt, al), a in zip(self._ptrs, args[:n], strict=True)
            ]
            converted += args[n:-1]
            self._executor(*converted, cu_stream(args[-1]))
            return
        cells, exe_args = self._cells()
        if len(args) != len(cells):
            raise TypeError(f"expected {len(cells)} launch arguments, got {len(args)}")
        for cell, value in zip(cells, args, strict=True):
            cell.value = value
        self._executor.run_compiled_program(exe_args)


def _compile(
    key: tuple,
    launcher,
    ptrs: list[tuple[type, int]],
    scalars: list[type],
    *cexprs,
    device: int | None = None,
) -> _Launch:
    if device is None:
        device = _current_device()
    launch = _executors.get((key, device))
    if launch is None:
        args = [gmem_ptr(dt, 256, al) for dt, al in ptrs] + [t(1) for t in scalars]
        fn = _compiled.get(key)
        if fn is None:
            with warnings.catch_warnings():
                # The block size is a runtime argument (num_warps * 32, as in the C++
                # launcher), which the DSL warns it cannot turn into a `reqntid` hint.
                warnings.filterwarnings("ignore", message="Dynamic variable in block size")
                fn = _compiled[key] = cute.compile(launcher, *args, *cexprs, cuda.CUstream())
        executor = fn.to(device)
        launch = _executors[(key, device)] = _Launch(
            executor, args + [cuda.CUstream()], ptrs, scalars
        )
    return launch


def _score_ptrs(align: int) -> list[tuple[type, int]]:
    return [(Int32, align), _F32P, _I64P, _I64P, (Int32, align), _F32P]


def compile_bloom_mask(device: int | None = None):
    """``(qb, sigs_t, probe_ids, mask, sig_row_words, mask_words, n_probe, wpc, W, B,
    stream)`` — pointer *addresses*, scalars and the stream *handle*, all plain ints
    (`_Launch`). ``device`` is the CUDA ordinal the executor is bound to (default: the
    device whose context is current); same for the other ``compile_*``."""
    scalars = [Int64] * 3 + [Int32] * 3
    return _compile(("bloom",), _launch_bloom_mask, [_I64P] * 4, scalars, device=device)


def compile_clause_mask(C: int, A: int, device: int | None = None):
    """``(flat_items, item_attrs, is_reverse, query_attrs, mask, P, max_size, mask_words,
    wpc, n_clauses, a_max, B, stream)``; ``(C, A)`` register fast path, ``(0, 0)`` the
    runtime-bound loop."""
    ptrs = [_I64P, _I64P, (Uint8, 1), _I64P, _I64P]
    scalars = [Int64] * 3 + [Int32] * 4
    return _compile(("clause", C, A), _launch_clause_mask, ptrs, scalars, C, A, device=device)


def compile_score(SEG: int, HAS_MASK: bool, UNROLL: int, device: int | None = None):
    """``(q_codes, q_scales, mask, flat_items, item_codes, out, global_scale, P, max_size,
    mask_words, wpc, B, block_p, num_warps, stream)``; the code pointers are ``Int32``
    views with 16 B-aligned rows."""
    key = ("score", SEG, HAS_MASK, UNROLL)
    ptrs, cexprs = _score_ptrs(16), (SEG, HAS_MASK, UNROLL)
    return _compile(key, _launch_score, ptrs, _SCORE_SCALARS, *cexprs, device=device)


def compile_score_generic(HAS_MASK: bool, device: int | None = None):
    """``compile_score``'s arguments plus ``d_words`` before the stream; the code
    pointers are ``Int32`` views with 4 B-aligned rows."""
    key, scalars = ("generic", HAS_MASK), _SCORE_SCALARS + [Int32]
    return _compile(key, _launch_score_generic, _score_ptrs(4), scalars, HAS_MASK, device=device)
