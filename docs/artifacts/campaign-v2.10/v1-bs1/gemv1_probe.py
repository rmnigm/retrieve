"""V1-BS1: the B = 1 sequential-FMA GEMV (one lane per item, k = 0 .. D-1 in order, fp32) against cuBLAS's
`torch.mm(q_fp16, E_t_fp16, out_dtype=fp32)` on a dataset's full item matrix ([D, N] contiguous, int64 offsets): the
cuBLAS kernel, `torch.equal` on 16 real queries, and device µs (CUDA events, 50 calls) per tile config.

    cd evaluation && PYTHONPATH=.:../retrieve/src python gemv1_probe.py DATASET DIM OUT.json
"""

import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from bench import config, inputs
from torch.profiler import ProfilerActivity, profile

DEV = torch.device("cuda")
ds, dim, out = sys.argv[1:4]
inp = inputs.load_inputs(
    config.load_dataset(Path(f"config/{ds}.yaml"), int(dim)), DEV, with_filters=False
)
E_t = inp["item_embs"].to(DEV).to(torch.float16).t().contiguous()
del inp["item_embs"]
D, N = E_t.shape


@triton.jit
def _gemv1(q_ptr, e_ptr, out_ptr, N, D: tl.constexpr, BLOCK_N: tl.constexpr):
    n = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    live = n < N
    acc = tl.zeros([BLOCK_N], tl.float32)
    for k in tl.range(0, D):
        qk = tl.load(q_ptr + k).to(tl.float32)
        ek = tl.load(e_ptr + k * N + n, mask=live, other=0.0).to(tl.float32)
        acc = tl.fma(qk, ek, acc)
    tl.store(out_ptr + n, acc, mask=live)


def gemv1(q, block_n, warps):
    out = torch.empty((1, N), dtype=torch.float32, device=DEV)
    _gemv1[(triton.cdiv(N, block_n),)](
        q, E_t, out, N, D=D, BLOCK_N=block_n, num_warps=warps
    )
    return out


def dev_us(fn):
    for _ in range(5):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 50 * 1e3


qs = inp["queries"].to(DEV).to(torch.float16)
CFGS = ((256, 4), (512, 4), (1024, 4), (1024, 8), (2048, 8))
eq = {f"{bn}x{w}": all(torch.equal(gemv1(qs[i : i + 1].contiguous(), bn, w),
                                    torch.mm(qs[i : i + 1], E_t, out_dtype=torch.float32))
                       for i in range(16)) for bn, w in CFGS}  # fmt: skip
q = qs[:1].contiguous()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    torch.mm(q, E_t, out_dtype=torch.float32)
    torch.cuda.synchronize()
kname = [
    e.key for e in p.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA
][0][:80]
row = {"dataset": ds, "D": D, "N": N, "cublas_kernel": kname, "equal": eq,
       "cublas_us": dev_us(lambda: torch.mm(q, E_t, out_dtype=torch.float32)),
       "triton_us": {f"{bn}x{w}": dev_us(lambda bn=bn, w=w: gemv1(q, bn, w)) for bn, w in CFGS}}  # fmt: skip
print(json.dumps(row), flush=True)
Path(out).write_text(json.dumps(row, indent=1))
