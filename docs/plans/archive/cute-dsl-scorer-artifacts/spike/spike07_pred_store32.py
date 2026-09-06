"""Item 7: predicated load on a dynamic scalar condition; 32-bit store into an Int64 buffer (pointer recast)."""
import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
import cuda.bindings.driver as cuda

@cute.kernel
def k(ids: cute.Tensor, rows: cute.Tensor, out: cute.Tensor, mask64: cute.Pointer, n: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = cutlass.Int64(bidx * 64 + tidx)
    lane = cute.arch.lane_idx()
    if i < n:
        idv = ids[i]
        keep = idv >= 0
        # (a) predicated load via if/else on a dynamic Boolean, both branches yield the same var
        v = cutlass.Int32(0)
        if keep:
            v = rows[idv]
        # (b) select-style: clamp index then load unconditionally, then select
        row = cutlass.select_(keep, idv, cutlass.Int64(0))   # ternary
        v2 = rows[row]
        v2 = cutlass.select_(keep, v2, cutlass.Int32(0))
        out[i] = v + v2
        # (c) ballot + 32-bit half store into the Int64 mask word: word = i//64, half = (i//32)&1
        packed = cute.arch.vote_ballot_sync(keep)
        if lane == 0:
            m32 = cute.recast_ptr(mask64, dtype=cutlass.Int32)         # view Int64 buffer as Int32
            mt = cute.make_tensor(m32, cute.make_layout(n // 32 * 2))
            mt[(i // 64) * 2 + ((i // 32) & 1)] = packed

@cute.jit
def host(ids, rows, out, mask64: cute.Pointer, n: cutlass.Int64, stream: cuda.CUstream):
    k(ids, rows, out, mask64, n).launch(grid=[(n + 63) // 64, 1, 1], block=[64, 1, 1], stream=stream)

torch.manual_seed(0)
n, R = 1024, 300
ids = torch.randint(-1, R, (n,), device="cuda", dtype=torch.int64)
rows = torch.randint(-1000, 1000, (R,), device="cuda", dtype=torch.int32)
out = torch.zeros(n, device="cuda", dtype=torch.int32)
mask = torch.zeros(n // 64, device="cuda", dtype=torch.int64)
st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
host(from_dlpack(ids), from_dlpack(rows), from_dlpack(out),
     make_ptr(cutlass.Int64, mask.data_ptr(), cute.AddressSpace.gmem, assumed_align=8), cutlass.Int64(n), st)
torch.cuda.synchronize()
ref = torch.where(ids >= 0, rows[ids.clamp(min=0)], torch.zeros((), device="cuda", dtype=torch.int32)) * 2
bits = (ids >= 0).view(-1, 64).to(torch.int64)
ref_mask = (bits << torch.arange(64, device="cuda")).sum(1)
print("pred load ok:", torch.equal(out, ref), "| 32-bit halves into int64 ok:", torch.equal(mask, ref_mask))
