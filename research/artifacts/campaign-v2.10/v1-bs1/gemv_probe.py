"""V1-BS1: can a Triton `tl.dot` GEMV reproduce cuBLAS's `torch.mm(q_fp16, E_t_fp16, out_dtype=fp32)` bit for bit?
On a dataset's real item embeddings (fp16, [D, N] contiguous as LiNR V1 holds them) and its queries, B in {1, 2, 4,
8, 16}: `torch.equal` against torch.mm, the cuBLAS kernel's name, and both device µs (CUDA events, 50 calls).

    cd evaluation && PYTHONPATH=.:../retrieve/src python gemv_probe.py DATASET DIM OUT.json
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
def _gemv(q_ptr, e_ptr, out_ptr, B, N, D: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr,
          BLOCK_K: tl.constexpr):  # fmt: skip
    n0 = tl.program_id(0) * BLOCK_N
    b = tl.arange(0, BLOCK_B)
    n = n0 + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_B, BLOCK_N], tl.float32)
    for k0 in tl.static_range(0, D, BLOCK_K):
        d = k0 + tl.arange(0, BLOCK_K)
        q = tl.load(q_ptr + b[:, None] * D + d[None, :], mask=b[:, None] < B, other=0.0)
        e = tl.load(e_ptr + d[:, None] * N + n[None, :], mask=n[None, :] < N, other=0.0)
        acc = tl.dot(q, e, acc=acc, out_dtype=tl.float32)
    tl.store(
        out_ptr + b[:, None] * N + n[None, :],
        acc,
        mask=(b[:, None] < B) & (n[None, :] < N),
    )


def triton_mm(q, block_n, warps):
    B = q.shape[0]
    out = torch.empty((B, N), dtype=torch.float32, device=DEV)
    _gemv[(triton.cdiv(N, block_n),)](
        q,
        E_t,
        out,
        B,
        N,
        D=D,
        BLOCK_B=16,
        BLOCK_N=block_n,
        BLOCK_K=128,
        num_warps=warps,
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


rows = []
qs = inp["queries"].to(DEV).to(torch.float16)
for B in (1, 2, 4, 16):
    eq_all = True
    for t in range(4):
        q = qs[t * B : (t + 1) * B].contiguous()
        ref = torch.mm(q, E_t, out_dtype=torch.float32)
        for block_n, warps in ((128, 4), (256, 4), (256, 8)):
            eq_all &= torch.equal(triton_mm(q, block_n, warps), ref)
    q = qs[:B].contiguous()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        torch.mm(q, E_t, out_dtype=torch.float32)
        torch.cuda.synchronize()
    kname = [
        e.key
        for e in p.key_averages()
        if e.device_type == torch.autograd.DeviceType.CUDA
    ][0][:100]
    t_ref = dev_us(lambda: torch.mm(q, E_t, out_dtype=torch.float32))
    t_tri = {
        f"{bn}x{w}": dev_us(lambda bn=bn, w=w: triton_mm(q, bn, w))
        for bn, w in ((128, 4), (256, 4), (256, 8))
    }
    row = {"dataset": ds, "D": D, "N": N, "B": B, "equal": eq_all, "cublas_kernel": kname, "cublas_us": t_ref,
           "triton_us": t_tri}  # fmt: skip
    rows.append(row)
    print(json.dumps(row), flush=True)
Path(out).write_text(json.dumps(rows, indent=1))
