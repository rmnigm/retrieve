"""Are Python ints passed to Int64-annotated @cute.jit params dynamic, or baked in (recompiled per value)?"""
import time, torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import cuda.bindings.driver as cuda

@cute.kernel
def k(yp: cute.Pointer, n: cutlass.Int64, stride: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 256 + tidx
    yt = cute.make_tensor(yp, cute.make_layout(n, stride=stride))
    if i < n:
        yt[i] = cutlass.Int32(i)

@cute.jit
def host(yp: cute.Pointer, n: cutlass.Int64, stride: cutlass.Int64, stream: cuda.CUstream):
    k(yp, n, stride).launch(grid=[(n + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream)

y = torch.zeros(4096, device="cuda", dtype=torch.int32)
yp = make_ptr(cutlass.Int32, y.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
for n, s in [(1000, 1), (1000, 1), (1001, 1), (1002, 2), (777, 3)]:
    t0 = time.perf_counter(); host(yp, n, s, st); torch.cuda.synchronize()
    print(f"py int  n={n} s={s}: {1e3*(time.perf_counter()-t0):8.2f} ms")
for n, s in [(1000, 1), (1001, 1), (1002, 2), (777, 3)]:
    t0 = time.perf_counter(); host(yp, cutlass.Int64(n), cutlass.Int64(s), st); torch.cuda.synchronize()
    print(f"Int64() n={n} s={s}: {1e3*(time.perf_counter()-t0):8.2f} ms")
print("ok:", torch.equal(y[:777:3], torch.arange(0, 259, device='cuda', dtype=torch.int32)))
# now cute.compile: does the compiled callable accept different n?
c = cute.compile(host, yp, cutlass.Int64(1000), cutlass.Int64(1), st)
y.zero_()
c(yp, 500, 2, st); torch.cuda.synchronize()
print("compiled callable with different n/stride ok:", torch.equal(y[:1000:2], torch.arange(500, device='cuda', dtype=torch.int32)), bool((y[1000:]==0).all()))
c2 = cute.compile(host, yp, 1000, 1, st)
y.zero_()
c2(yp, 500, 2, st); torch.cuda.synchronize()
print("compiled-with-python-int callable with different n/stride ok:", torch.equal(y[:1000:2], torch.arange(500, device='cuda', dtype=torch.int32)))
