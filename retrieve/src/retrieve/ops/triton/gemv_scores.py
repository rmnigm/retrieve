"""``[1, N]`` fp32 dense scores of one fp16 query against the ``[D, N]`` fp16 item table,
accumulated in the order of cuBLAS's ``gemv2N`` (one fp32 FMA per k, k = 0 .. D-1), so the
result is ``torch.mm(q, E_t, out_dtype=float32)`` bit for bit where cuBLAS runs that kernel —
``PostfilterKNN`` checks which at ``register_index`` (kernels.md § gemv_scores)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

# Tuned on A100 at D 128 / 256 (goodreads 0.8 M, arXiv 3 M): 1024 items a program, 4 warps.
BLOCK_N = 1024
NUM_WARPS = 4


@triton.jit
def _gemv_scores_kernel(q_ptr, e_ptr, out_ptr, N, D: tl.constexpr, BLOCK_N: tl.constexpr):
    n = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    live = n < N
    acc = tl.zeros([BLOCK_N], tl.float32)
    for k in tl.range(0, D):
        qk = tl.load(q_ptr + k).to(tl.float32)
        ek = tl.load(e_ptr + k * N + n, mask=live, other=0.0).to(tl.float32)
        acc = tl.fma(qk, ek, acc)
    tl.store(out_ptr + n, acc, mask=live)


@triton_op("retrieve::gemv_scores", mutates_args=())
def gemv_scores(query: Tensor, item_embs_t: Tensor) -> Tensor:
    """``query [1, D]`` fp16 against ``item_embs_t [D, N]`` fp16 (contiguous) → ``[1, N]`` fp32,
    one item per lane."""
    if query.shape[0] != 1 or query.dtype != torch.float16 or item_embs_t.dtype != torch.float16:
        raise ValueError("gemv_scores takes one fp16 query row and an fp16 [D, N] table")
    d, n = item_embs_t.shape
    out = torch.empty((1, n), dtype=torch.float32, device=query.device)
    grid = (triton.cdiv(n, BLOCK_N),)
    wrap_triton(_gemv_scores_kernel)[grid](
        query.contiguous(), item_embs_t, out, n, D=d, BLOCK_N=BLOCK_N, num_warps=NUM_WARPS
    )
    return out
