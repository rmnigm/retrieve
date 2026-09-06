"""Item 5: Int64 ops (&, >>, <<, ~, -, //, %, compares), cttz on Int64 (the __ffsll set-bit walk),
dynamic `while`, dynamic-trip-count `for` via cutlass.range / range(dyn_start, dyn_end)."""
import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import math as mlir_math, llvm
import cuda.bindings.driver as cuda

def cttz64(x: cutlass.Int64) -> cutlass.Int32:
    # math.cttz on i64 -> i64; narrow to Int32
    return cutlass.Int32(cutlass.Int64(mlir_math.cttz(x.ir_value())))

@cute.kernel
def k(qb: cute.Tensor, sigs: cute.Tensor, out: cute.Tensor, o_ops: cute.Tensor, W: cutlass.Int32, ncols: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx()
    col = cutlass.Int64(tidx)
    if col < ncols:
        result = ~cutlass.Int64(0)                     # AND identity
        # dynamic-trip-count for: W is a runtime Int32
        for qw in cutlass.range(W):
            bits = qb[qw]                               # Int64
            while bits != 0:                            # dynamic while
                j = cttz64(bits)                        # __ffsll(bits)-1
                bits = bits & (bits - 1)
                m = cutlass.Int64(qw) * 64 + j
                result = result & sigs[m * ncols + col]
        out[col] = result
    if tidx == 0:
        a = cutlass.Int64(-0x123456789ABCDEF)
        b = cutlass.Int64(0x0F0F0F0F0F0F0F0F)
        o_ops[0] = a & b
        o_ops[1] = a >> 7          # arithmetic shift (signed)
        o_ops[2] = a << 5
        o_ops[3] = ~a
        o_ops[4] = a - b
        o_ops[5] = a // 1000       # floor or trunc? record it
        o_ops[6] = a % 1000
        o_ops[7] = cutlass.Int64(a < b) + cutlass.Int64(a == a) * 2 + cutlass.Int64(a >= 0) * 4
        o_ops[8] = cutlass.Int64(cttz64(cutlass.Int64(1) << 40))
        # range(dyn_start, dyn_end, step) accumulate
        s = cutlass.Int64(0)
        lo = cutlass.Int64(W) * 3
        hi = ncols * 2
        for i in range(lo, hi, 2):
            s += i
        o_ops[9] = s
        # 64-bit unsigned-style shift on Int64 with sign bit set
        o_ops[10] = cutlass.Int64(cutlass.Uint64(a) >> 60)
        o_ops[11] = cutlass.Int64(cutlass.Uint64(a) // cutlass.Uint64(1000))

@cute.jit
def host(qb, sigs, out, o_ops, W: cutlass.Int32, ncols: cutlass.Int64, stream: cuda.CUstream):
    k(qb, sigs, out, o_ops, W, ncols).launch(grid=[1, 1, 1], block=[128, 1, 1], stream=stream)

torch.manual_seed(0)
W, ncols = 3, 100
qb = torch.randint(-2**63, 2**63 - 1, (W,), device="cuda", dtype=torch.int64) & torch.randint(-2**63, 2**63 - 1, (W,), device="cuda", dtype=torch.int64)
sigs = torch.randint(-2**63, 2**63 - 1, (W * 64 * ncols,), device="cuda", dtype=torch.int64)
out = torch.zeros(ncols, device="cuda", dtype=torch.int64)
o_ops = torch.zeros(12, device="cuda", dtype=torch.int64)
st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
host(from_dlpack(qb), from_dlpack(sigs), from_dlpack(out), from_dlpack(o_ops), cutlass.Int32(W), cutlass.Int64(ncols), st)
torch.cuda.synchronize()
# reference
ref = torch.full((ncols,), -1, device="cuda", dtype=torch.int64)
S2 = sigs.view(W * 64, ncols)
for qw in range(W):
    b = int(qb[qw].item()) & (2**64 - 1)
    while b:
        j = (b & -b).bit_length() - 1
        b &= b - 1
        ref &= S2[qw * 64 + j]
print("bloom-style cttz walk ok:", torch.equal(out, ref), "| set bits:", sum(bin(int(v) & (2**64-1)).count('1') for v in qb.tolist()))
a, b = -0x123456789ABCDEF, 0x0F0F0F0F0F0F0F0F
def s64(v): v &= 2**64-1; return v - 2**64 if v >= 2**63 else v
got = o_ops.tolist()
exp = [a & b, a >> 7, s64(a << 5), ~a, a - b, None, None, int(a < b) + 2 + 0, 40, sum(range(W*3, ncols*2, 2)), (a & (2**64-1)) >> 60, (a & (2**64-1)) // 1000]
for i, (g, e) in enumerate(zip(got, exp)):
    if i == 5: print(f"  a//1000: got {g}; python floor {a//1000}; C trunc {int(a/1000)}")
    elif i == 6: print(f"  a%1000: got {g}; python {a%1000}; C rem {a - int(a/1000)*1000}")
    else: print(f"  op[{i}] {'ok' if g == e else 'MISMATCH'} got={g} exp={e}")
