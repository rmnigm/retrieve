"""L5 §7.3 items 1 and 5 on the C4 goodreads-d128 `c0_genre` cell (the L4 inputs, via
`../probe_v2_parity.py`): the shipped kernel now accumulates in fp32.

A. `linr_v2` torch-vs-triton parity over the 9 859 kept rows (L4: 0.998743 / 624 rows).
B. Max abs error of the shipped kernel against an fp64 dot of the same fp16 inputs over every
   candidate of chunk 0, next to a probe-only copy of the pre-L5 fp16-accumulating body.
C. `do_bench` of both bodies on the cell's real shapes, same session, same clocks.
D. The compiled PTX's arithmetic op counts (and the PTX itself, `--ptx-out`).

    cd /workspace/wt/l5/retrieve && flock /workspace/gpu.lock uv run --no-sync python \
        ../docs/artifacts/linr-v2-backend-parity/l5/parity_and_cost.py \
        --out ../docs/artifacts/linr-v2-backend-parity/l5/parity_and_cost.json \
        --ptx-out ../docs/artifacts/linr-v2-backend-parity/l5/fused_masked_knn_topk_fp32.ptx
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from probe_v2_parity import CHUNK, KS, build, jaccard, load  # noqa: E402

from retrieve.ops.triton.fused_masked_knn_topk import (  # noqa: E402
    DEFAULT_CONFIG,
    _fmkt_prep,
    _fused_masked_knn_topk_kernel,
)
from retrieve.ops.tune import _bench  # noqa: E402

PTX_OPS = (
    "add.f16", "add.rn.f16", "fma.rn.f16", "mul.f16", "mul.rn.f16",
    "add.f32", "add.rn.f32", "fma.rn.f32", "mul.f32", "mul.rn.f32", "cvt.f32.f16",
)  # fmt: skip


@triton.jit
def _kernel_fp16_acc(
    query_ptr, item_embs_ptr, pos_indices_ptr, counts_ptr, out_scores_ptr,
    P: tl.constexpr, D: tl.constexpr,
    stride_qb, stride_qd, stride_in, stride_id, stride_pb, stride_pp, stride_sb, stride_sp,
    BLOCK_N: tl.constexpr,
):  # fmt: skip
    """The pre-L5 body, verbatim: no cast, so tl.sum reduces in the operand dtype."""
    tile_id = tl.program_id(0)
    bid = tl.program_id(1)
    n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, D)
    p_valid = n_offsets < P
    count = tl.load(counts_ptr + bid)
    in_count = n_offsets < count
    q = tl.load(query_ptr + bid * stride_qb + d_offsets * stride_qd)
    item_ids = tl.load(pos_indices_ptr + bid * stride_pb + n_offsets * stride_pp, mask=in_count, other=0)
    emb_rows = tl.load(
        item_embs_ptr + item_ids[:, None] * stride_in + d_offsets[None, :] * stride_id,
        mask=in_count[:, None],
        other=0.0,
    )
    dots = tl.sum(emb_rows * q[None, :], axis=1)
    dots = tl.where(in_count, dots, float("-inf"))
    tl.store(out_scores_ptr + bid * stride_sb + n_offsets * stride_sp, dots, mask=p_valid)


BODIES = (("shipped_fp32_acc", _fused_masked_knn_topk_kernel), ("pre_l5_fp16_acc", _kernel_fp16_acc))


def sm_mhz() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout  # fmt: skip
    return int(out.strip().splitlines()[0])


def all_scores(kern, q16, item16, cand, counts):
    launch = _fmkt_prep(q16, item16, cand, counts, DEFAULT_CONFIG, bucket=False)
    kern[launch.grid](**launch.kwargs)
    return launch.all_scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ptx-out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")
    t0 = time.time()
    item_embs, queries, attrs, reverse, qa, keep = load(device)
    rows = keep.nonzero().reshape(-1)
    mt = build("torch", item_embs, attrs, reverse)
    mr = build("triton", item_embs, attrs, reverse)
    item16 = mr.idx.item_embs
    out = {"n_kept_rows": int(rows.numel()), "sm_mhz_start": sm_mhz()}

    # --- A. parity over every kept row ---------------------------------------------------------
    ids_t, sc_t, ids_r, sc_r = [], [], [], []
    for s in range(0, rows.numel(), CHUNK):
        sel = rows[s : s + CHUNK]
        q = queries[sel].to(device)
        qs = qa[sel].to(device)
        it, st = mt(q, qs)
        ir, sr = mr(q, qs)
        ids_t.append(it); sc_t.append(st.float()); ids_r.append(ir); sc_r.append(sr)  # noqa: E702
    ids_t, sc_t, ids_r, sc_r = map(torch.cat, (ids_t, sc_t, ids_r, sc_r))
    both = torch.isfinite(sc_t) & torch.isfinite(sc_r)
    differ = (jaccard(ids_t, ids_r, 100) < 1).nonzero().reshape(-1)
    ties = 0
    for r in differ.tolist():
        a = set(ids_t[r, :100].tolist()) - {-1}
        b = set(ids_r[r, :100].tolist()) - {-1}
        s_t = {int(i): float(v) for i, v in zip(ids_t[r].tolist(), sc_t[r].tolist())}
        pair = [s_t[i] for i in sorted(a - b)] + [s_t.get(i) for i in sorted(b - a)]
        ties += all(p is not None and p == pair[0] for p in pair)
    out["parity_triton_vs_torch"] = {
        **{f"jaccard@{k}": jaccard(ids_t, ids_r, k).mean().item() for k in KS},
        "rows_top100_differ": int(differ.numel()),
        "rows_top100_differ_where_torch_scores_tie_exactly": ties,
        "score_max_abs_diff": (sc_t - sc_r).abs()[both].max().item(),
    }
    print(json.dumps(out, indent=1), flush=True)

    # --- B. error against fp64 over every candidate of chunk 0 --------------------------------
    sel = rows[:CHUNK]
    q16 = queries[sel].to(device).to(torch.float16)
    qs = qa[sel].to(device)
    cand, counts = mr.filter.evaluate_indices(qs)
    cand = cand[:, : int(counts.max())].contiguous()
    valid = torch.arange(cand.shape[1], device=device)[None, :] < counts[:, None]
    truth = torch.einsum("bpd,bd->bp", item16[cand.clamp_min(0)].double(), q16.double())
    out["fp64_error_chunk0"] = {"P": int(cand.shape[1]), "abs_score_max": truth[valid].abs().max().item()}
    for name, kern in BODIES:
        err = (all_scores(kern, q16, item16, cand, counts).double() - truth).abs()[valid]
        out["fp64_error_chunk0"][name] = {"max_abs": err.max().item(), "mean_abs": err.mean().item()}
    print(json.dumps(out["fp64_error_chunk0"], indent=1), flush=True)

    # --- C. cost, both bodies, same session ----------------------------------------------------
    timing = {}
    for bsz in (1, 16):
        sel = rows[:bsz]
        q16 = queries[sel].to(device).to(torch.float16)
        qs = qa[sel].to(device)
        cand, counts = mr.filter.evaluate_indices(qs)
        cand = cand[:, : int(counts.max())].contiguous()
        for name, kern in BODIES:
            launch = _fmkt_prep(q16, item16, cand, counts, DEFAULT_CONFIG, bucket=False)
            fn = lambda: kern[launch.grid](**launch.kwargs)  # noqa: E731
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            m0 = sm_mhz()
            ms = _bench(fn)
            timing[f"kernel_B{bsz}_P{cand.shape[1]}_{name}"] = {"ms": ms, "sm_mhz": [m0, sm_mhz()]}
        q = queries[sel].to(device)
        for name, fn in (("linr_v2_triton_eager", lambda: mr(q, qs)), ("linr_v2_torch_eager", lambda: mt(q, qs))):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            m0 = sm_mhz()
            ms = _bench(fn)
            timing[f"forward_B{bsz}_{name}"] = {"ms": ms, "sm_mhz": [m0, sm_mhz()]}
    out["timing"] = timing
    print(json.dumps(timing, indent=1), flush=True)

    # --- D. PTX ---------------------------------------------------------------------------------
    ptx = {}
    for name, kern in BODIES:
        for cache in kern.device_caches.values():
            for compiled in cache[0].values():
                ptx[name] = compiled.asm["ptx"]
    out["ptx_op_counts"] = {
        name: {pat: len(re.findall(r"\b" + re.escape(pat) + r"\b", text)) for pat in PTX_OPS}
        for name, text in ptx.items()
    }
    print(json.dumps(out["ptx_op_counts"], indent=1), flush=True)
    if args.ptx_out:
        Path(args.ptx_out).write_text(ptx["shipped_fp32_acc"])

    out["elapsed_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"done {out['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
