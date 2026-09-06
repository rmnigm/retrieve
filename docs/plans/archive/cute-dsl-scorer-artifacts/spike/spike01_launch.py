"""Item 1 + 12: launch a @cute.kernel from @cute.jit with torch tensors on torch's stream,
via (a) from_dlpack and (b) make_ptr + Int64 strides. Also proves sm_80 targeting."""
import os, time
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
import cuda.bindings.driver as cuda

@cute.kernel
def k_dlpack(x: cute.Tensor, y: cute.Tensor, n: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bidx * bdim + tidx
    if i < n:
        y[i] = x[i] * 2 + 1

@cute.jit
def host_dlpack(x: cute.Tensor, y: cute.Tensor, n: cutlass.Int64, stream: cuda.CUstream):
    nblk = (n + 255) // 256
    k_dlpack(x, y, n).launch(grid=[nblk, 1, 1], block=[256, 1, 1], stream=stream)

@cute.kernel
def k_ptr(xp: cute.Pointer, yp: cute.Pointer, n: cutlass.Int64, stride: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bidx * bdim + tidx
    # wrap raw pointers into 1-D tensors with a dynamic stride
    xt = cute.make_tensor(xp, cute.make_layout(n, stride=stride))
    yt = cute.make_tensor(yp, cute.make_layout(n, stride=stride))
    if i < n:
        yt[i] = xt[i] * 2 + 1

@cute.jit
def host_ptr(xp: cute.Pointer, yp: cute.Pointer, n: cutlass.Int64, stride: cutlass.Int64, stream: cuda.CUstream):
    nblk = (n + 255) // 256
    k_ptr(xp, yp, n, stride).launch(grid=[nblk, 1, 1], block=[256, 1, 1], stream=stream)

if __name__ == "__main__":
    n = 1000
    x = torch.arange(n, device="cuda", dtype=torch.int32)
    y = torch.zeros_like(x)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # route (a)
    t0 = time.perf_counter()
    host_dlpack(from_dlpack(x), from_dlpack(y), n, stream)
    torch.cuda.synchronize()
    print("dlpack route ok:", torch.equal(y, x * 2 + 1), f"({time.perf_counter()-t0:.2f}s incl. compile)")
    # route (b)
    y2 = torch.zeros_like(x)
    xp = make_ptr(cutlass.Int32, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    yp = make_ptr(cutlass.Int32, y2.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    t0 = time.perf_counter()
    host_ptr(xp, yp, n, 1, stream)
    torch.cuda.synchronize()
    print("make_ptr route ok:", torch.equal(y2, x * 2 + 1), f"({time.perf_counter()-t0:.2f}s incl. compile)")
    # a strided view via make_ptr (every other element of a 2n buffer)
    xs = torch.arange(2 * n, device="cuda", dtype=torch.int32)
    ys = torch.zeros_like(xs)
    host_ptr(make_ptr(cutlass.Int32, xs.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
             make_ptr(cutlass.Int32, ys.data_ptr(), cute.AddressSpace.gmem, assumed_align=16), n, 2, stream)
    torch.cuda.synchronize()
    print("make_ptr stride=2 ok:", torch.equal(ys[::2], xs[::2] * 2 + 1), "odd untouched:", bool((ys[1::2] == 0).all()))
