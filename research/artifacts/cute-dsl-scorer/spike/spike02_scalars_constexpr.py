"""Item 2: scalar kernel args of every type + Constexpr specialization + static unrolling."""
import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

@cute.kernel
def k(out: cute.Tensor, a64: cutlass.Int64, a32: cutlass.Int32, f: cutlass.Float32, flag: cutlass.Boolean,
      SEG: cutlass.Constexpr[int], HAS_MASK: cutlass.Constexpr[bool]):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        acc = cutlass.Int64(0)
        # static unroll: plain python range over a Constexpr -> fully unrolled at trace time
        for u in range(SEG):
            acc = acc + a64 * (u + 1)
        # explicit static unroll
        for u in cutlass.range_constexpr(SEG):
            acc = acc + a32
        # if constexpr equivalent
        if cutlass.const_expr(HAS_MASK):
            acc = acc + 1000000
        # dynamic branch on a Boolean arg
        if flag:
            acc = acc * 2
        out[0] = acc
        out[1] = cutlass.Int64(cutlass.Float32(acc) * f)  # float epilogue then trunc (fp32 rounding!)

@cute.jit
def host(out: cute.Tensor, a64: cutlass.Int64, a32: cutlass.Int32, f: cutlass.Float32, flag: cutlass.Boolean,
         SEG: cutlass.Constexpr[int], HAS_MASK: cutlass.Constexpr[bool], stream: cuda.CUstream):
    k(out, a64, a32, f, flag, SEG, HAS_MASK).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
out = torch.zeros(2, device="cuda", dtype=torch.int64)
for SEG, HAS_MASK, flag in [(4, True, True), (8, False, False), (16, True, False)]:
    a64, a32, f = 3_000_000_000, 7, 0.5
    host(from_dlpack(out), cutlass.Int64(a64), cutlass.Int32(a32), cutlass.Float32(f), cutlass.Boolean(flag), SEG, HAS_MASK, st)
    torch.cuda.synchronize()
    ref = a64 * SEG * (SEG + 1) // 2 + a32 * SEG + (1000000 if HAS_MASK else 0)
    if flag: ref *= 2
    # out[1] goes through fp32: emulate Int64 -> fp32 (rn) -> * f -> trunc to Int64
    ref1 = int((torch.tensor(ref, dtype=torch.int64).to(torch.float32) * torch.tensor(f, dtype=torch.float32)).to(torch.int64))
    print(f"SEG={SEG} HAS_MASK={HAS_MASK} flag={flag}: got {out.tolist()} ref [{ref}, {ref1}] ->", out[0].item() == ref and out[1].item() == ref1)
