"""End-to-end: a SEG-lane-per-item int8 scorer combining every primitive (int4 .cs load, 4x dp4a, redux over a
segment mask, predicated loads, fp32 epilogue, -inf for rejected) -- bit-exact against a torch reference."""
import math, torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
import cuda.bindings.driver as cuda

def dp4a(a, b, c):
    return cute.arch.inline_ptx("dp4a.s32.s32 {$w0}, {$r0}, {$r1}, {$r2};",
                                write_only_types=[cutlass.Int32], read_only_args=[a, b, c])

def ld_int4(ptr, cs: bool):
    vt = ir.VectorType.get([4], cutlass.Int32.mlir_type)
    v = cute.arch.load(ptr, vt, cop="cs" if cs else None)
    return [cutlass.Int32(llvm.extractelement(v, cutlass.Int32(i).ir_value())) for i in range(4)]

@cute.kernel
def score_kernel(q_codes: cute.Pointer, q_scales: cute.Pointer, flat_items: cute.Pointer, item_codes: cute.Pointer,
                 out: cute.Pointer, global_scale: cutlass.Float32, P: cutlass.Int64, items_per_warp: cutlass.Int32,
                 SEG: cutlass.Constexpr[int]):
    D = SEG * 16
    SPW = 32 // SEG
    tidx, _, _ = cute.arch.thread_idx()
    bidx, b, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    seg = lane // SEG
    sl = lane % SEG
    seg_mask = ((1 << SEG) - 1) << (seg * SEG)
    b64 = cutlass.Int64(b)
    # query words for this lane (4 x Int32 = 16 bytes at chunk sl of row b)
    q_ptr = cute.recast_ptr(q_codes, dtype=cutlass.Int32) + (b64 * (D // 4) + sl * 4)
    qv = ld_int4(q_ptr, cs=False)
    q_scale = cute.make_tensor(q_scales + b64, cute.make_layout(1))[0]   # scalar load: offset the POINTER, then index 0
    # (indexing a size-1 layout with coordinate b is NOT an offset of b -- it read the wrong element)
    ids_t = cute.make_tensor(flat_items + b64 * P, cute.make_layout(P))
    out_t = cute.make_tensor(out + b64 * P, cute.make_layout(P))
    codes32 = cute.recast_ptr(item_codes, dtype=cutlass.Int32)
    warp_start = cutlass.Int64(bidx) * (items_per_warp * (bdim // 32)) + cutlass.Int64(warp) * items_per_warp
    warp_end = cutlass.min(warp_start + items_per_warp, P)
    for p in range(warp_start + seg, warp_end, SPW):        # dynamic-trip-count loop, python step constant
        idv = ids_t[p]
        keep = idv >= 0
        row = cutlass.select_(keep, idv, cutlass.Int64(0))
        rw = [cutlass.Int32(0)] * 4
        if keep:                                             # segment-uniform predicate
            rw = ld_int4(codes32 + (row * (D // 4) + sl * 4), cs=True)
        score = cutlass.Float32(-math.inf)
        if keep:
            acc = dp4a(rw[0], qv[0], cutlass.Int32(0))
            acc = dp4a(rw[1], qv[1], acc)
            acc = dp4a(rw[2], qv[2], acc)
            acc = dp4a(rw[3], qv[3], acc)
            acc = cute.arch.warp_redux_sync(acc, "add", mask_and_clamp=seg_mask)
            score = cutlass.Float32(acc) * q_scale * global_scale
        if sl == 0:
            out_t[p] = score

@cute.jit
def score_host(q_codes: cute.Pointer, q_scales: cute.Pointer, flat_items: cute.Pointer, item_codes: cute.Pointer,
               out: cute.Pointer, global_scale: cutlass.Float32, B: cutlass.Int32, P: cutlass.Int64,
               items_per_warp: cutlass.Int32, SEG: cutlass.Constexpr[int], stream: cuda.CUstream):
    threads = 128
    per_block = items_per_warp * (threads // 32)
    score_kernel(q_codes, q_scales, flat_items, item_codes, out, global_scale, P, items_per_warp, SEG).launch(
        grid=[(P + per_block - 1) // per_block, B, 1], block=[threads, 1, 1], stream=stream)

def ref_score(q_codes, q_scales, flat_items, item_codes, g):
    keep = flat_items >= 0
    rows = item_codes[flat_items.clamp(min=0)].to(torch.int32)           # [B,P,D]
    acc = (rows * q_codes.to(torch.int32)[:, None, :]).sum(-1)            # exact int32
    s = acc.to(torch.float32) * q_scales[:, None] * torch.tensor(g, dtype=torch.float32, device="cuda")
    return torch.where(keep, s, torch.tensor(-math.inf, device="cuda"))

torch.manual_seed(0)
st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
ptr = lambda t, ty, al=16: make_ptr(ty, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=al)
for D, B, P, N in [(64, 3, 1000, 5000), (128, 2, 4097, 20000), (256, 4, 777, 3000)]:
    SEG = D // 16
    q_codes = torch.randint(-128, 128, (B, D), device="cuda", dtype=torch.int8)
    q_scales = torch.rand(B, device="cuda") * 0.01
    item_codes = torch.randint(-128, 128, (N, D), device="cuda", dtype=torch.int8)
    flat_items = torch.randint(0, N, (B, P), device="cuda", dtype=torch.int64); flat_items[torch.rand(B, P, device="cuda") < 0.1] = -1
    out = torch.full((B, P), 12345.0, device="cuda")
    g = 0.0371
    compiled = cute.compile(score_host, ptr(q_codes, cutlass.Int8), ptr(q_scales, cutlass.Float32, 4), ptr(flat_items, cutlass.Int64, 8),
                            ptr(item_codes, cutlass.Int8), ptr(out, cutlass.Float32, 4), cutlass.Float32(g), cutlass.Int32(B),
                            cutlass.Int64(P), cutlass.Int32(64), SEG, st)
    call_args = [ptr(q_codes, cutlass.Int8), ptr(q_scales, cutlass.Float32, 4), ptr(flat_items, cutlass.Int64, 8),
                 ptr(item_codes, cutlass.Int8), ptr(out, cutlass.Float32, 4), g, B, P, 64]
    # NOTE: Constexpr params are stripped from the compiled callable's signature: call WITHOUT `SEG`.
    # (Passing it shifts the positionals so the int lands in the CUstream slot -> segfault, no validation.)
    compiled(*call_args, st)
    torch.cuda.synchronize()
    ref = ref_score(q_codes, q_scales, flat_items, item_codes, g)
    print(f"D={D} SEG={SEG} B={B} P={P}: bit-exact = {torch.equal(out, ref)}  (-inf: {int(torch.isinf(out).sum())}/{int((flat_items<0).sum())})")
    if not torch.equal(out, ref):
        bad = (out != ref) & ~(torch.isinf(out) & torch.isinf(ref))
        print("   mismatches:", int(bad.sum()), "| first bad idx:", bad.nonzero()[:4].tolist(), "| out:", out[bad][:4].tolist(), "ref:", ref[bad][:4].tolist(), "| untouched(12345):", int((out == 12345.0).sum()), "| zeros:", int((out == 0).sum()))
        print("   out[0,:6]", out[0,:6].tolist(), "ref[0,:6]", ref[0,:6].tolist())
    if D == 128:
        sass = compiled.__sass__ or ""   # property; populated only when CUTE_DSL_KEEP includes sass
        import collections
        c = collections.Counter(tok for l in sass.splitlines() for tok in l.split() if tok.startswith(("LDG", "IDP", "REDUX", "FMUL", "FFMA", "I2F", "STG")))
        print("   SASS ops:", dict(c))
