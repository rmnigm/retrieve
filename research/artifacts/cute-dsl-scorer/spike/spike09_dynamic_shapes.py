"""Item 9: dynamic shapes with from_dlpack marking so a cute.compile'd callable is reused across shapes;
make_ptr route needs no marking."""
import time, torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
import cuda.bindings.driver as cuda

@cute.kernel
def k2d(x: cute.Tensor, y: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    R, C = x.shape          # dynamic Int32 values after marking
    c = bidx * 128 + tidx
    if c < C:
        y[bidy, c] = x[bidy, c] * 2

@cute.jit
def host2d(x: cute.Tensor, y: cute.Tensor, stream: cuda.CUstream):
    R, C = x.shape
    k2d(x, y).launch(grid=[(C + 127) // 128, R, 1], block=[128, 1, 1], stream=stream)

st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
def mk(t):
    # row-major 2D: mode 1 is contiguous (stride 1); shapes AND leading stride dynamic
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)

x0 = torch.randint(-100, 100, (4, 1000), device="cuda", dtype=torch.int32); y0 = torch.zeros_like(x0)
t0 = time.perf_counter()
compiled = cute.compile(host2d, mk(x0), mk(y0), st)
print(f"compile: {time.perf_counter()-t0:.2f}s")
for shape in [(4, 1000), (7, 333), (1, 64), (16, 4096)]:
    x = torch.randint(-100, 100, shape, device="cuda", dtype=torch.int32); y = torch.zeros_like(x)
    t0 = time.perf_counter()
    compiled(mk(x), mk(y), st)
    torch.cuda.synchronize()
    print(f"  shape {shape}: ok={torch.equal(y, x*2)} call+sync {1e3*(time.perf_counter()-t0):.2f} ms")
# non-contiguous rows (padded stride) also fine since leading stride is dynamic:
xb = torch.randint(-100, 100, (5, 512), device="cuda", dtype=torch.int32); x = xb[:, :300]; y = torch.zeros_like(xb)[:, :300]
compiled(mk(x), mk(y), st); torch.cuda.synchronize(); print("  strided rows ok:", torch.equal(y, x*2))
# What happens WITHOUT marking? (static shape baked into the compiled function)
compiled_static = cute.compile(host2d, from_dlpack(x0, assumed_align=16), from_dlpack(y0, assumed_align=16), st)
try:
    x = torch.randint(-100, 100, (7, 333), device="cuda", dtype=torch.int32); y = torch.zeros_like(x)
    compiled_static(from_dlpack(x, assumed_align=16), from_dlpack(y, assumed_align=16), st); torch.cuda.synchronize()
    print("  static-compiled with different shape: NO ERROR, result ok =", torch.equal(y, x*2))
except Exception as e:
    print("  static-compiled with different shape raised:", type(e).__name__, str(e).splitlines()[0][:150])
# alternative marking: mark_compact_shape_dynamic
def mk2(t): return from_dlpack(t, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=t.dim_order(), divisibility=1).mark_compact_shape_dynamic(mode=1, stride_order=t.dim_order(), divisibility=8)
c3 = cute.compile(host2d, mk2(x0), mk2(y0), st)
x = torch.randint(-100, 100, (3, 2048), device="cuda", dtype=torch.int32); y = torch.zeros_like(x)
c3(mk2(x), mk2(y), st); torch.cuda.synchronize(); print("  mark_compact_shape_dynamic ok:", torch.equal(y, x*2))

# make_ptr route: shapes are plain Int64 scalars, nothing to mark
@cute.kernel
def kp(xp: cute.Pointer, yp: cute.Pointer, R: cutlass.Int64, C: cutlass.Int64, ld: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx(); bidx, bidy, _ = cute.arch.block_idx()
    x = cute.make_tensor(xp, cute.make_layout((R, C), stride=(ld, 1)))
    y = cute.make_tensor(yp, cute.make_layout((R, C), stride=(ld, 1)))
    c = bidx * 128 + tidx
    if c < C:
        y[bidy, c] = x[bidy, c] * 2
@cute.jit
def hostp(xp: cute.Pointer, yp: cute.Pointer, R: cutlass.Int64, C: cutlass.Int64, ld: cutlass.Int64, stream: cuda.CUstream):
    kp(xp, yp, R, C, ld).launch(grid=[(C + 127) // 128, R, 1], block=[128, 1, 1], stream=stream)
P = lambda t: make_ptr(cutlass.Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
cp = cute.compile(hostp, P(x0), P(y0), cutlass.Int64(4), cutlass.Int64(1000), cutlass.Int64(1000), st)
for shape in [(4, 1000), (7, 333), (16, 4096)]:
    x = torch.randint(-100, 100, shape, device="cuda", dtype=torch.int32); y = torch.zeros_like(x)
    cp(P(x), P(y), shape[0], shape[1], x.stride(0), st); torch.cuda.synchronize()
    print(f"  make_ptr shape {shape}: ok={torch.equal(y, x*2)}")

# Prove the static-compiled callable silently misbehaves (no validation): larger tensor -> tail untouched
x = torch.randint(1, 100, (16, 4096), device="cuda", dtype=torch.int32); y = torch.zeros_like(x)
compiled_static(from_dlpack(x, assumed_align=16), from_dlpack(y, assumed_align=16), st); torch.cuda.synchronize()
print("  static-compiled (4,1000) called with (16,4096): all correct? ", torch.equal(y, x*2), "| nonzero written:", int((y != 0).sum()), "of", y.numel(), "-> SILENT WRONG RESULT (uses baked shape)")
