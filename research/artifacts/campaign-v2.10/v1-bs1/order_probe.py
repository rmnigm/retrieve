"""V1-BS1 (runtime k loop: a static unroll at D 768 compiled for tens of minutes): which fp32 accumulation order reproduces cuBLAS's small-B fp16 GEMV (`gemv2N` at B = 1, `gemmSN` at
B = 2 / 4) bit for bit? Candidate orders in a Triton kernel with explicit per-k ops, each item row by one lane:
SPLIT partial accumulators over k (contiguous chunks, or INTERLEAVE'd k mod SPLIT), combined left to right, with FMA
or a separate multiply and add. `torch.equal` against torch.mm on 8 query batches of real data.

    cd evaluation && PYTHONPATH=.:../retrieve/src python order_probe.py DATASET DIM OUT.json
"""

import itertools
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from bench import config, inputs

DEV = torch.device("cuda")
ds, dim, out = sys.argv[1:4]
CANDIDATES = tuple(
    int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,2,4").split(",")
)
inp = inputs.load_inputs(
    config.load_dataset(Path(f"config/{ds}.yaml"), int(dim)), DEV, with_filters=False
)
E_t = inp["item_embs"].to(DEV).to(torch.float16)[:200_000].t().contiguous()
D, N = E_t.shape


@triton.jit
def _order(q_ptr, e_ptr, out_ptr, N, b, D: tl.constexpr, SPLIT: tl.constexpr, INTERLEAVE: tl.constexpr,
           FMA: tl.constexpr, BLOCK_N: tl.constexpr):  # fmt: skip
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    live = n < N
    total = tl.zeros([BLOCK_N], tl.float32)
    for s in tl.static_range(SPLIT):
        acc = tl.zeros([BLOCK_N], tl.float32)
        for j in tl.range(0, D // SPLIT):
            k = j * SPLIT + s if INTERLEAVE else s * (D // SPLIT) + j
            qk = tl.load(q_ptr + b * D + k).to(tl.float32)
            ek = tl.load(e_ptr + k * N + n, mask=live, other=0.0).to(tl.float32)
            if FMA:
                acc = tl.fma(qk, ek, acc)
            else:
                acc = acc + qk * ek
        total = acc if s == 0 else total + acc
    tl.store(out_ptr + b * N + n, total, mask=live)


qs = inp["queries"].to(DEV).to(torch.float16)
rows = []
for B in (1, 2, 4):
    refs = [(qs[t * B : (t + 1) * B].contiguous(),) for t in range(8)]
    refs = [(q, torch.mm(q, E_t, out_dtype=torch.float32)) for (q,) in refs]
    for split, inter, fma in itertools.product(
        (1, 2, 4, 8), (False, True), (True, False)
    ):
        if split == 1 and inter:
            continue
        ok = True
        for q, ref in refs:
            got = torch.empty_like(ref)
            for b in range(B):
                _order[(triton.cdiv(N, 256),)](q, E_t, got, N, b, D=D, SPLIT=split, INTERLEAVE=inter, FMA=fma,
                                               BLOCK_N=256)  # fmt: skip
            ok &= torch.equal(got, ref)
        row = {
            "dataset": ds,
            "D": D,
            "B": B,
            "split": split,
            "interleave": inter,
            "fma": fma,
            "equal": bool(ok),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
Path(out).write_text(json.dumps(rows, indent=1))
