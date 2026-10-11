"""Item 6: 128-bit int4 load per lane (with .cs streaming hint) -> registers; check SASS for LDG.E.128."""
import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector as mlir_vector
import cuda.bindings.driver as cuda

def load_int4_cs(ptr):
    """ptr: cute.Pointer to Int32 (16B aligned). Returns 4 Int32 via ld.global.cs.v4.s32."""
    vt = ir.VectorType.get([4], cutlass.Int32.mlir_type)
    v = cute.arch.load(ptr, vt, cop="cs")                    # nvvm.load_ext, vector result
    return tuple(cutlass.Int32(llvm.extractelement(v, cutlass.Int32(i).ir_value())) for i in range(4))

def load_int4_ptx(ptr):
    """same via inline_ptx, for comparison"""
    return cute.arch.inline_ptx("ld.global.cs.v4.s32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
                                write_only_types=[cutlass.Int32] * 4, read_only_args=[ptr])

@cute.kernel
def k(xp: cute.Pointer, o_a: cute.Tensor, o_b: cute.Tensor, n_words: cutlass.Int64, MODE: cutlass.Constexpr[int]):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = cutlass.Int64(bidx * 128 + tidx)              # chunk index
    if i * 4 < n_words:
        p = xp + i * 4                                 # pointer arithmetic in elements
        if cutlass.const_expr(MODE == 0):
            a, b, c, d = load_int4_cs(p)
        else:
            a, b, c, d = load_int4_ptx(p)
        o_a[i] = a + b
        o_b[i] = c ^ d

@cute.jit
def host(xp: cute.Pointer, o_a, o_b, n_words: cutlass.Int64, MODE: cutlass.Constexpr[int], stream: cuda.CUstream):
    k(xp, o_a, o_b, n_words, MODE).launch(grid=[(n_words // 4 + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream)

torch.manual_seed(0)
n_words = 4096
x8 = torch.randint(-128, 128, (n_words * 4,), device="cuda", dtype=torch.int8)
x = x8.view(torch.int32)
st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
xp = make_ptr(cutlass.Int32, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
xv = x.view(-1, 4).to(torch.int64)
ref_a = ((xv[:, 0] + xv[:, 1]) & 0xFFFFFFFF); ref_a = torch.where(ref_a >= 2**31, ref_a - 2**32, ref_a).to(torch.int32)
ref_b = (xv[:, 2] ^ xv[:, 3]).to(torch.int32)
for MODE in (0, 1):
    o_a = torch.zeros(n_words // 4, device="cuda", dtype=torch.int32); o_b = torch.zeros_like(o_a)
    host(xp, from_dlpack(o_a), from_dlpack(o_b), cutlass.Int64(n_words), MODE, st)
    torch.cuda.synchronize()
    print(f"MODE={MODE} ({'arch.load vec cs' if MODE==0 else 'inline_ptx ld.global.cs.v4'}): ok =", torch.equal(o_a, ref_a) and torch.equal(o_b, ref_b))

# PTX/SASS accessors on a compiled object (no env vars needed)
c = cute.compile(host, xp, from_dlpack(torch.zeros(n_words // 4, device="cuda", dtype=torch.int32)), from_dlpack(torch.zeros(n_words // 4, device="cuda", dtype=torch.int32)), cutlass.Int64(n_words), 0, st)
ptx = c.__ptx__()
print("compiled.__ptx__() ->", type(ptx).__name__, len(ptx or ""), "chars; ld.global.cs.v4 present:", "ld.global.cs.v4.s32" in (ptx or ""))
sass = c.__sass__()
print("compiled.__sass__() ->", type(sass).__name__, "; LDG.E.128 lines:", [l.strip() for l in (sass or "").splitlines() if "LDG.E" in l][:3])
