"""Item 10c: cold-process compile time + does the on-disk cache (CUTE_DSL_CACHE_DIR) hit across processes?"""
import time; T0 = time.perf_counter()
import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import cuda.bindings.driver as cuda
T1 = time.perf_counter()

@cute.kernel
def k(x: cute.Pointer, n: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx()
    t = cute.make_tensor(x, cute.make_layout(n))
    if tidx < n:
        t[tidx] = cutlass.Float32(tidx) * 0.5
@cute.jit
def host(x: cute.Pointer, n: cutlass.Int64, stream: cuda.CUstream):
    k(x, n).launch(grid=[1, 1, 1], block=[128, 1, 1], stream=stream)

st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
x = torch.zeros(128, device="cuda")
xp = make_ptr(cutlass.Float32, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
t0 = time.perf_counter(); c = cute.compile(host, xp, cutlass.Int64(128), st); t_compile = time.perf_counter() - t0
t0 = time.perf_counter(); c2 = cute.compile(host, xp, cutlass.Int64(128), st); t_compile2 = time.perf_counter() - t0
t0 = time.perf_counter(); host(xp, cutlass.Int64(128), st); t_direct = time.perf_counter() - t0
torch.cuda.synchronize()
print(f"imports {1e3*(T1-T0):.0f} ms | cute.compile #1 {1e3*t_compile:.0f} ms | cute.compile #2 same args {1e3*t_compile2:.0f} ms | @cute.jit direct 1st call {1e3*t_direct:.0f} ms | ok={torch.equal(x, torch.arange(128, device='cuda')*0.5)}")
