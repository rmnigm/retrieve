"""Item 10: cold compile time, per-call host overhead (from_dlpack vs make_ptr), persistent cache."""
import os, sys, time, torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
import cuda.bindings.driver as cuda

@cute.kernel
def k_empty_t(x: cute.Tensor, n: cutlass.Int64):
    pass
@cute.jit
def host_t(x: cute.Tensor, n: cutlass.Int64, stream: cuda.CUstream):
    k_empty_t(x, n).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

@cute.kernel
def k_empty_p(x: cute.Pointer, n: cutlass.Int64, m: cutlass.Int64):
    pass
@cute.jit
def host_p(x: cute.Pointer, n: cutlass.Int64, m: cutlass.Int64, stream: cuda.CUstream):
    k_empty_p(x, n, m).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
x = torch.zeros(1024, device="cuda", dtype=torch.float32)
mk = lambda t: from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=0)
mp = lambda t: make_ptr(cutlass.Float32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

t0 = time.perf_counter(); ct = cute.compile(host_t, mk(x), cutlass.Int64(1024), st); t_ct = time.perf_counter() - t0
t0 = time.perf_counter(); cp = cute.compile(host_p, mp(x), cutlass.Int64(1024), cutlass.Int64(1), st); t_cp = time.perf_counter() - t0
print(f"cute.compile (trivial kernel): from_dlpack {t_ct*1e3:.0f} ms, make_ptr {t_cp*1e3:.0f} ms  [first in process includes DSL warmup]")

def bench(fn, iters=2000):
    for _ in range(50): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    t = (time.perf_counter() - t0) / iters
    torch.cuda.synchronize()
    return t * 1e6

n = 1024
print(f"per-call host overhead (us, {2000} launches, no sync):")
print(f"  compiled, from_dlpack (incl. from_dlpack+mark per call): {bench(lambda: ct(mk(x), n, st)):.1f}")
xt = mk(x)
print(f"  compiled, from_dlpack (tensor wrapper pre-built):        {bench(lambda: ct(xt, n, st)):.1f}")
print(f"  compiled, make_ptr   (incl. make_ptr per call):          {bench(lambda: cp(mp(x), n, 1, st)):.1f}")
xp = mp(x)
print(f"  compiled, make_ptr   (ptr pre-built):                    {bench(lambda: cp(xp, n, 1, st)):.1f}")
print(f"  @cute.jit direct call, make_ptr, Int64() args:          {bench(lambda: host_p(xp, cutlass.Int64(n), cutlass.Int64(1), st), 300):.1f}")
print(f"  @cute.jit direct call, from_dlpack:                     {bench(lambda: host_t(xt, cutlass.Int64(n), st), 300):.1f}")
# torch baseline: an empty-ish torch op launch
print(f"  baseline torch x.add_(0) launch:                         {bench(lambda: x.add_(0.0)):.1f}")
print(f"  baseline from_dlpack+mark only (no launch):              {bench(lambda: mk(x)):.1f}")
print(f"  baseline make_ptr only (no launch):                      {bench(lambda: mp(x)):.1f}")
print(f"  baseline cuda.CUstream(torch stream) only:               {bench(lambda: cuda.CUstream(torch.cuda.current_stream().cuda_stream)):.1f}")
