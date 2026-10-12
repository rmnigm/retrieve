"""V2-FILL: split-loop fmkt bodies at D_PAD <= 256 against campaign-v2.4's grid-strided body, kernel-only
(fill_gate.py's inputs and ABAB graphs). S = v2.4's own SPLIT body (q reloaded per tile, -inf in FILL_N chunks);
Sq = the same split with q held across tiles; Hf = v2.4's constexpr-bounded loop scoring only counted tiles, then the fill; A = v2.4's own launch (the A/A floor); L = this tree's library launch; a "32" suffix (Sq32, Hb32) casts the row's count to int32;
Hb = v2.4's branched tile body with
its loop bounded at the counted tiles, then the same fill. Each takes ":w<num_warps>:b<block_n>:g<programs>:s<num_stages>" overrides of the
d <= 256 config (e.g. "Sq:b16:g1728"). Each variant's [B, P] scores are checked torch.equal to v2.4's before timing.

    PYTHONPATH=<pkgs>:retrieve/src:. python fill_variants.py N D out.json [--variants S,Sq] [--skew 0.12:2]
"""

import argparse
import dataclasses
import json
import statistics

import torch
import triton
import triton.language as tl
from fill_gate import DEV, MODS, POOL, candidates, graph_of, ratio_ci, sm_mhz, window_ms

from retrieve.ops.triton.common import row_base


@triton.jit
def fmkt_split_qheld_kernel(query_ptr, item_embs_ptr, pos_indices_ptr, counts_ptr, out_scores_ptr, P: tl.constexpr,
                            D: tl.constexpr, D_PAD: tl.constexpr, stride_qb, stride_qd, stride_in, stride_id,
                            stride_pb, stride_pp, stride_sb, stride_sp, BLOCK_N: tl.constexpr, WIDE: tl.constexpr,
                            SPLIT: tl.constexpr, FILL_N: tl.constexpr, I32: tl.constexpr = False):  # fmt: skip
    bid = tl.program_id(1)
    pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    count = tl.load(counts_ptr + bid)
    if I32:
        count = count.to(tl.int32)
    out_row = row_base(out_scores_ptr, bid, stride_sb, WIDE)
    d_offsets = tl.arange(0, D_PAD)
    d_in = d_offsets < D
    q = tl.load(
        query_ptr + bid * stride_qb + d_offsets * stride_qd, mask=d_in, other=0.0
    ).to(tl.float32)
    n_score = tl.cdiv(count, BLOCK_N)
    for tile_id in range(pid, n_score, n_prog):
        n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
        in_count = n_offsets < count
        item_ids = tl.load(row_base(pos_indices_ptr, bid, stride_pb, WIDE) + n_offsets * stride_pp,
                           mask=in_count, other=0)  # fmt: skip
        emb_rows = tl.load(item_embs_ptr + item_ids[:, None] * stride_in + d_offsets[None, :] * stride_id,
                           mask=in_count[:, None] & d_in[None, :], other=0.0).to(tl.float32)  # fmt: skip
        dots = tl.sum(emb_rows * q[None, :], axis=1)
        dots = tl.where(in_count, dots, float("-inf"))
        tl.store(out_row + n_offsets * stride_sp, dots, mask=n_offsets < P)
    tail0 = n_score * BLOCK_N
    for c in range(pid, tl.cdiv(P - tail0, FILL_N), n_prog):
        offs = tail0 + c * FILL_N + tl.arange(0, FILL_N)
        tl.store(
            out_row + offs * stride_sp,
            tl.full((FILL_N,), float("-inf"), tl.float32),
            mask=offs < P,
        )


@triton.jit
def fmkt_hybrid_kernel(query_ptr, item_embs_ptr, pos_indices_ptr, counts_ptr, out_scores_ptr, P: tl.constexpr,
                       D: tl.constexpr, D_PAD: tl.constexpr, stride_qb, stride_qd, stride_in, stride_id,
                       stride_pb, stride_pp, stride_sb, stride_sp, BLOCK_N: tl.constexpr, WIDE: tl.constexpr,
                       SPLIT: tl.constexpr, FILL_N: tl.constexpr, I32: tl.constexpr = False):  # fmt: skip
    """Hb: v2.4's branched tile body, its loop bounded at the row's counted tiles; then the FILL_N -inf fill."""
    bid = tl.program_id(1)
    pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    d_offsets = tl.arange(0, D_PAD)
    d_in = d_offsets < D
    count = tl.load(counts_ptr + bid)
    if I32:
        count = count.to(tl.int32)
    out_row = row_base(out_scores_ptr, bid, stride_sb, WIDE)
    q = tl.load(
        query_ptr + bid * stride_qb + d_offsets * stride_qd, mask=d_in, other=0.0
    ).to(tl.float32)
    n_score = tl.cdiv(count, BLOCK_N)
    for tile_id in range(pid, n_score, n_prog):
        n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
        p_valid = n_offsets < P
        if tile_id * BLOCK_N < count:
            in_count = n_offsets < count
            item_ids = tl.load(row_base(pos_indices_ptr, bid, stride_pb, WIDE) + n_offsets * stride_pp,
                               mask=in_count, other=0)  # fmt: skip
            emb_rows = tl.load(item_embs_ptr + item_ids[:, None] * stride_in + d_offsets[None, :] * stride_id,
                               mask=in_count[:, None] & d_in[None, :], other=0.0).to(tl.float32)  # fmt: skip
            dots = tl.sum(emb_rows * q[None, :], axis=1)
            dots = tl.where(in_count, dots, float("-inf"))
            tl.store(out_row + n_offsets * stride_sp, dots, mask=p_valid)
    tail0 = n_score * BLOCK_N
    for c in range(pid, tl.cdiv(P - tail0, FILL_N), n_prog):
        offs = tail0 + c * FILL_N + tl.arange(0, FILL_N)
        tl.store(
            out_row + offs * stride_sp,
            tl.full((FILL_N,), float("-inf"), tl.float32),
            mask=offs < P,
        )


@triton.jit
def fmkt_constloop_kernel(query_ptr, item_embs_ptr, pos_indices_ptr, counts_ptr, out_scores_ptr, P: tl.constexpr,
                          D: tl.constexpr, D_PAD: tl.constexpr, stride_qb, stride_qd, stride_in, stride_id,
                          stride_pb, stride_pp, stride_sb, stride_sp, BLOCK_N: tl.constexpr, WIDE: tl.constexpr,
                          SPLIT: tl.constexpr, FILL_N: tl.constexpr, I32: tl.constexpr = False):  # fmt: skip
    """Hf: v2.4's loop over every tile (constexpr bound) scoring the counted ones, no per-tile -inf; then the fill."""
    bid = tl.program_id(1)
    pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    d_offsets = tl.arange(0, D_PAD)
    d_in = d_offsets < D
    count = tl.load(counts_ptr + bid)
    if I32:
        count = count.to(tl.int32)
    out_row = row_base(out_scores_ptr, bid, stride_sb, WIDE)
    q = tl.load(
        query_ptr + bid * stride_qb + d_offsets * stride_qd, mask=d_in, other=0.0
    ).to(tl.float32)
    for tile_id in range(pid, tl.cdiv(P, BLOCK_N), n_prog):
        n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
        p_valid = n_offsets < P
        if tile_id * BLOCK_N < count:
            in_count = n_offsets < count
            item_ids = tl.load(row_base(pos_indices_ptr, bid, stride_pb, WIDE) + n_offsets * stride_pp,
                               mask=in_count, other=0)  # fmt: skip
            emb_rows = tl.load(item_embs_ptr + item_ids[:, None] * stride_in + d_offsets[None, :] * stride_id,
                               mask=in_count[:, None] & d_in[None, :], other=0.0).to(tl.float32)  # fmt: skip
            dots = tl.sum(emb_rows * q[None, :], axis=1)
            dots = tl.where(in_count, dots, float("-inf"))
            tl.store(out_row + n_offsets * stride_sp, dots, mask=p_valid)
    tail0 = tl.cdiv(count, BLOCK_N) * BLOCK_N
    for c in range(pid, tl.cdiv(P - tail0, FILL_N), n_prog):
        offs = tail0 + c * FILL_N + tl.arange(0, FILL_N)
        tl.store(
            out_row + offs * stride_sp,
            tl.full((FILL_N,), float("-inf"), tl.float32),
            mask=offs < P,
        )


def launch(variant, q, items, cand, counts):
    name, *opts = variant.split(":")
    mod = MODS["after" if name == "L" else "before"]
    i32 = name.endswith("32")
    name = name.removesuffix("32")
    cfg = mod.config_for_width(q.shape[1])
    names = {"w": "num_warps", "b": "block_n", "g": "programs", "s": "num_stages"}
    for o in opts:
        cfg = dataclasses.replace(cfg, **{names[o[0]]: int(o[1:])})
    la = mod._fmkt_prep(q, items, cand, counts, cfg, bucket=False)
    kern = mod._fused_masked_knn_topk_kernel
    if name == "S":
        la.kwargs["SPLIT"] = True
    elif name == "Sq":
        kern = fmkt_split_qheld_kernel
        la.kwargs["I32"] = i32
    elif name == "Hf":
        kern = fmkt_constloop_kernel
    elif name == "Hb":
        kern = fmkt_hybrid_kernel
        la.kwargs["I32"] = i32
    elif name not in ("before", "A", "L"):
        raise ValueError(variant)
    return kern, la


ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("d", type=int)
ap.add_argument("out")
ap.add_argument("--rates", default="0.001,0.01,0.1,1.0")
ap.add_argument("--pairs", type=int, default=8)
ap.add_argument("--variants", default="S,Sq")
ap.add_argument(
    "--skew",
    default="",
    help="p_hi:m — the first m rows of each batch at p_hi, the rest at the rate",
)
args = ap.parse_args()
VARIANTS = tuple(args.variants.split(","))
g = torch.Generator(device=DEV).manual_seed(0)
items = torch.randn(args.n, args.d, device=DEV, generator=g).to(torch.float16)
queries = torch.randn(POOL, 16, args.d, device=DEV, generator=g).to(torch.float16)
rows = []
for p in map(float, filter(None, args.rates.split(","))):
    cand16, counts16 = candidates(args.n, 16, p, seed=1)
    if args.skew:
        p_hi, m = args.skew.split(":")
        hi, hi_counts = candidates(args.n, int(m), float(p_hi), seed=2)
        cand16[: int(m)], counts16[: int(m)] = hi, hi_counts
    for bs in (1, 16):
        cand, counts = cand16[:bs].contiguous(), counts16[:bs].contiguous()
        ls = {v: [launch(v, queries[i, :bs].contiguous(), items, cand, counts) for i in range(POOL)]
              for v in ("before", *VARIANTS)}  # fmt: skip
        eq, regs = {}, {}
        for v in VARIANTS:
            same = True
            for (kb, lb), (kv, lv) in zip(ls["before"], ls[v], strict=True):
                cb = kb[lb.grid](**lb.kwargs)
                cv = kv[lv.grid](**lv.kwargs)
                same &= torch.equal(lb.all_scores, lv.all_scores)
            eq[v] = same
            regs["before"], regs[v] = (cb.n_regs, cb.n_spills), (cv.n_regs, cv.n_spills)
        graphs = {v: graph_of(x) for v, x in ls.items()}
        for _ in range(2):
            for gr in graphs.values():
                window_ms(gr)
        t = {v: [] for v in graphs}
        mhz = []
        for _ in range(args.pairs):
            for v in graphs:
                t[v].append(window_ms(graphs[v]))
            mhz.append(sm_mhz())
        row = {"n": args.n, "d": args.d, "p": p, "bs": bs, "mean_count": float(counts.float().mean()),
               "ms": {v: statistics.median(x) for v, x in t.items()},
               "over_before": {v: ratio_ci(t[v], t["before"]) for v in VARIANTS},
               "equal": eq, "regs_spills": regs, "sm_mhz": [min(mhz), max(mhz)]}  # fmt: skip
        rows.append(row)
        print(f"p={p} bs={bs}: before {row['ms']['before']:.3f} ms | "
              + " | ".join(f"{v} {row['ms'][v]:.3f} ({row['over_before'][v][0]:.3f} [{row['over_before'][v][1]:.3f},"
                           f" {row['over_before'][v][2]:.3f}]) eq={eq[v]}" for v in VARIANTS)
              + f" | sm {row['sm_mhz']} | regs/spills {regs}", flush=True)  # fmt: skip
        del graphs, ls
        torch.cuda.empty_cache()
json.dump(rows, open(args.out, "w"), indent=1)
