"""Probe the DSL patterns the port relies on beyond the spike: jit helpers with dynamic control flow called
from a kernel, a loop-carried list across a dynamic-bound range loop, Boolean &/|/!=, Uint8 pointer reads,
and make_ptr with a dummy address as a cute.compile example arg."""
import torch, cutlass, cutlass.cute as cute
from cutlass import Int32, Int64, Boolean, Uint8, Float32
from cutlass.cute.runtime import make_ptr
import cuda.bindings.driver as cuda

def _ld(ptr, off):
    return cute.make_tensor(ptr + off, cute.make_layout(1))[0]

def _st(ptr, off, v):
    cute.make_tensor(ptr + off, cute.make_layout(1))[0] = v

@cute.jit
def ld_pred(ptr, off, pred):
    v = Int64(-1)
    if pred:
        v = _ld(ptr, off)
    return v

@cute.jit
def carry(keep, slot, cl, max_size):
    if keep:
        s = slot
        c = cl
        while s >= max_size:
            s -= max_size
            c += 1
        keep = ((c * 1000 + s) & 1) != 0
    return keep

@cute.kernel
def k(ids: cute.Pointer, flags: cute.Pointer, out: cute.Pointer, n: Int64, max_size: Int64, U: cutlass.Constexpr[int]):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        nxt = [ld_pred(ids, u, u < n) for u in range(U)]
        acc = Int64(0)
        for p0 in range(Int64(0), n, U):
            cur = nxt
            nxt = [ld_pred(ids, p0 + U + u, p0 + U + u < n) for u in range(U)]
            keep = [c >= 0 for c in cur]
            keep = [carry(keep[u], p0 + u, Int64(0), max_size) for u in range(U)]
            for u in cutlass.range_constexpr(U):
                if keep[u]:
                    acc = acc + cur[u]
        f0 = _ld(flags, 0) != 0
        f1 = _ld(flags, 1) != 0
        bb = Boolean(True)
        bb = bb & ((f0 != f1) | (n == -1))
        _st(out, 0, acc)
        _st(out, 1, Int64(bb))

@cute.jit
def host(ids: cute.Pointer, flags: cute.Pointer, out: cute.Pointer, n: Int64, max_size: Int64, U: cutlass.Constexpr[int], stream: cuda.CUstream):
    k(ids, flags, out, n, max_size, U).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

P = lambda t, ty, al: make_ptr(ty, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=al)
st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
n, ms = 37, 5
ids = torch.arange(n, device="cuda", dtype=torch.int64); ids[3] = -1; ids[10] = -1
flags = torch.tensor([True, False], device="cuda")
out = torch.zeros(2, device="cuda", dtype=torch.int64)
for U in (1, 2, 4):
    # dummy-address example args for compile
    ex = lambda ty, al: make_ptr(ty, 256, cute.AddressSpace.gmem, assumed_align=al)
    comp = cute.compile(host, ex(Int64, 8), ex(Uint8, 1), ex(Int64, 8), Int64(1), Int64(1), U, st)
    comp(P(ids, Int64, 8), P(flags, Uint8, 1), P(out, Int64, 8), n, ms, st)
    torch.cuda.synchronize()
    ref = 0
    for p in range(n):
        v = int(ids[p]); 
        if v < 0: continue
        s, c = p, 0
        while s >= ms: s -= ms; c += 1
        if ((c * 1000 + s) & 1) != 0: ref += v
    print(f"U={U}: acc {int(out[0])} ref {ref} ok={int(out[0])==ref} | bool {int(out[1])} (expect 1)")
