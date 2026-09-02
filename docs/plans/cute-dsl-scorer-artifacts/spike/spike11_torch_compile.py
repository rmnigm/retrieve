"""Item 11: wrap the compiled CuTe callable in torch.library.custom_op; run eagerly and under torch.compile(fullgraph=True)."""
import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import cuda.bindings.driver as cuda

@cute.kernel
def k(xp: cute.Pointer, yp: cute.Pointer, n: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx(); bidx, _, _ = cute.arch.block_idx()
    i = cutlass.Int64(bidx * 128 + tidx)
    x = cute.make_tensor(xp, cute.make_layout(n)); y = cute.make_tensor(yp, cute.make_layout(n))
    if i < n:
        y[i] = x[i] * 3.0
@cute.jit
def host(xp: cute.Pointer, yp: cute.Pointer, n: cutlass.Int64, stream: cuda.CUstream):
    k(xp, yp, n).launch(grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream)

_x = torch.zeros(16, device="cuda")
P = lambda t: make_ptr(cutlass.Float32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
compiled = cute.compile(host, P(_x), P(_x), cutlass.Int64(16), cuda.CUstream(torch.cuda.current_stream().cuda_stream))

@torch.library.custom_op("spike::triple", mutates_args=())
def triple(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    compiled(P(x), P(y), x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    return y
@triple.register_fake
def _(x): return torch.empty_like(x)

x = torch.randn(1000, device="cuda")
print("eager custom op ok:", torch.equal(triple(x), x * 3))
f = torch.compile(lambda t: triple(t + 1) - 2, fullgraph=True)
print("torch.compile fullgraph ok:", torch.equal(f(x), (x + 1) * 3 - 2))
x2 = torch.randn(4096, device="cuda")
print("torch.compile new shape ok:", torch.equal(f(x2), (x2 + 1) * 3 - 2))
# side stream: does it launch on the right stream?
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    z = triple(x2)
torch.cuda.synchronize()
print("side-stream ok:", torch.equal(z, x2 * 3))
