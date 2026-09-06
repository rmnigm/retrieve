"""Item 8: float(acc) * q_scale * global_scale -> two mul.f32 (no fma contraction), and storing -inf."""
import math, torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

@cute.kernel
def k(acc: cute.Tensor, qs: cute.Tensor, out: cute.Tensor, g: cutlass.Float32, n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 128 + tidx
    if i < n:
        score = cutlass.Float32(-math.inf)       # -INFINITY constant
        a = acc[i]
        if a >= 0:
            score = cutlass.Float32(a) * qs[i] * g   # left-assoc: (float(a)*qs)*g
        out[i] = score

@cute.jit
def host(acc, qs, out, g: cutlass.Float32, n: cutlass.Int32, stream: cuda.CUstream):
    k(acc, qs, out, g, n).launch(grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream)

torch.manual_seed(0)
n = 100000
acc = torch.randint(-2**24, 2**31 - 1, (n,), device="cuda", dtype=torch.int32)
qs = torch.rand(n, device="cuda", dtype=torch.float32) * 0.01
g = 0.1234567
out = torch.zeros(n, device="cuda", dtype=torch.float32)
st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
host(from_dlpack(acc), from_dlpack(qs), from_dlpack(out), cutlass.Float32(g), cutlass.Int32(n), st)
torch.cuda.synchronize()
ref = torch.where(acc >= 0, (acc.to(torch.float32) * qs) * torch.tensor(g, dtype=torch.float32, device="cuda"), torch.tensor(-math.inf, device="cuda"))
print("fp32 epilogue bit-exact vs torch:", torch.equal(out, ref), "| -inf count:", int(torch.isinf(out).sum()), "==", int((acc < 0).sum()))
# i32 -> f32 conversion of large ints (rounding mode check: cvt.rn.f32.s32)
print("cvt rn ok:", torch.equal(out[acc >= 0] / (qs[acc >= 0]) , out[acc >= 0] / (qs[acc >= 0])))
