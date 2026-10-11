"""Item 3: dp4a.s32.s32 via cute.arch.inline_ptx; verified vs torch int8 dot."""
import os, torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

def dp4a(a: cutlass.Int32, b: cutlass.Int32, c: cutlass.Int32) -> cutlass.Int32:
    return cute.arch.inline_ptx("dp4a.s32.s32 {$w0}, {$r0}, {$r1}, {$r2};",
                                write_only_types=[cutlass.Int32], read_only_args=[a, b, c])

@cute.kernel
def k(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 128 + tidx
    if i < n:
        # a, b are Int32 views over packed int8x4; 4 words per row
        acc = cutlass.Int32(0)
        for w in range(4):
            acc = dp4a(a[i, w], b[w], acc)
        out[i] = acc

@cute.jit
def host(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
    k(a, b, out, n).launch(grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream)

torch.manual_seed(0)
n = 4096
A8 = torch.randint(-128, 128, (n, 16), device="cuda", dtype=torch.int8)
B8 = torch.randint(-128, 128, (16,), device="cuda", dtype=torch.int8)
out = torch.zeros(n, device="cuda", dtype=torch.int32)
st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
host(from_dlpack(A8.view(torch.int32)), from_dlpack(B8.view(torch.int32)), from_dlpack(out), n, st)
torch.cuda.synchronize()
ref = (A8.to(torch.int32) * B8.to(torch.int32)).sum(1)
print("dp4a ok:", torch.equal(out, ref))
