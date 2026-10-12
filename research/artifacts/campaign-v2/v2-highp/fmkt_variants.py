"""V2-HIGHP: three candidate fmkt bodies above D_PAD 256 against campaign-v2.2's, kernel-only (fmkt_gate.py's
inputs and ABAB graphs). A = v2.2's grid-strided kernel launched with num_stages 1; B = straight-line, one tile
per program, early exit for tiles wholly past the row's count (defined here); C = campaign-v2's straight-line
kernel; D = the strided loop with q loaded per tile (D1: D at num_stages 1); E / F = v2.2's / D's body at maxnreg 64. Each variant's [B, P] scores are checked torch.equal to v2.2's before timing.

    PYTHONPATH=<pkgs>:retrieve/src:. python fmkt_variants.py N D out.json [--rates ...] [--pairs 8]
"""

import argparse
import dataclasses
import json
import statistics

import torch
import triton
import triton.language as tl
from fmkt_gate import DEV, MODS, POOL, candidates, graph_of, ratio_ci, sm_mhz, window_ms

from retrieve.ops.triton.common import row_base


@triton.jit
def fmkt_exit_kernel(query_ptr, item_embs_ptr, pos_indices_ptr, counts_ptr, out_scores_ptr, P: tl.constexpr,
                     D: tl.constexpr, D_PAD: tl.constexpr, stride_qb, stride_qd, stride_in, stride_id, stride_pb,
                     stride_pp, stride_sb, stride_sp, BLOCK_N: tl.constexpr, WIDE: tl.constexpr):  # fmt: skip
    tile_id = tl.program_id(0)
    bid = tl.program_id(1)
    n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    p_valid = n_offsets < P
    count = tl.load(counts_ptr + bid)
    out_row = row_base(out_scores_ptr, bid, stride_sb, WIDE)
    if tile_id * BLOCK_N >= count:
        tl.store(
            out_row + n_offsets * stride_sp,
            tl.full((BLOCK_N,), float("-inf"), tl.float32),
            mask=p_valid,
        )
    else:
        d_offsets = tl.arange(0, D_PAD)
        d_in = d_offsets < D
        in_count = n_offsets < count
        q = tl.load(
            query_ptr + bid * stride_qb + d_offsets * stride_qd, mask=d_in, other=0.0
        ).to(tl.float32)
        item_ids = tl.load(row_base(pos_indices_ptr, bid, stride_pb, WIDE) + n_offsets * stride_pp,
                           mask=in_count, other=0)  # fmt: skip
        emb_rows = tl.load(item_embs_ptr + item_ids[:, None] * stride_in + d_offsets[None, :] * stride_id,
                           mask=in_count[:, None] & d_in[None, :], other=0.0).to(tl.float32)  # fmt: skip
        dots = tl.sum(emb_rows * q[None, :], axis=1)
        dots = tl.where(in_count, dots, float("-inf"))
        tl.store(out_row + n_offsets * stride_sp, dots, mask=p_valid)


@triton.jit
def fmkt_strided_qin_kernel(query_ptr, item_embs_ptr, pos_indices_ptr, counts_ptr, out_scores_ptr, P: tl.constexpr,
                            D: tl.constexpr, D_PAD: tl.constexpr, stride_qb, stride_qd, stride_in, stride_id,
                            stride_pb, stride_pp, stride_sb, stride_sp, BLOCK_N: tl.constexpr,
                            WIDE: tl.constexpr):  # fmt: skip
    """D: v2.2's grid-strided loop, but q is loaded inside the tile body instead of held across tiles."""
    bid = tl.program_id(1)
    count = tl.load(counts_ptr + bid)
    out_row = row_base(out_scores_ptr, bid, stride_sb, WIDE)
    for tile_id in range(tl.program_id(0), tl.cdiv(P, BLOCK_N), tl.num_programs(0)):
        n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
        p_valid = n_offsets < P
        if tile_id * BLOCK_N < count:
            d_offsets = tl.arange(0, D_PAD)
            d_in = d_offsets < D
            in_count = n_offsets < count
            q = tl.load(
                query_ptr + bid * stride_qb + d_offsets * stride_qd,
                mask=d_in,
                other=0.0,
            )
            q = q.to(tl.float32)
            item_ids = tl.load(row_base(pos_indices_ptr, bid, stride_pb, WIDE) + n_offsets * stride_pp,
                               mask=in_count, other=0)  # fmt: skip
            emb_rows = tl.load(item_embs_ptr + item_ids[:, None] * stride_in + d_offsets[None, :] * stride_id,
                               mask=in_count[:, None] & d_in[None, :], other=0.0).to(tl.float32)  # fmt: skip
            dots = tl.sum(emb_rows * q[None, :], axis=1)
            dots = tl.where(in_count, dots, float("-inf"))
            tl.store(out_row + n_offsets * stride_sp, dots, mask=p_valid)
        else:
            inf = tl.full((BLOCK_N,), float("-inf"), tl.float32)
            tl.store(out_row + n_offsets * stride_sp, inf, mask=p_valid)


@triton.jit
def fmkt_split_kernel(query_ptr, item_embs_ptr, pos_indices_ptr, counts_ptr, out_scores_ptr, P: tl.constexpr,
                      D: tl.constexpr, D_PAD: tl.constexpr, stride_qb, stride_qd, stride_in, stride_id, stride_pb,
                      stride_pp, stride_sb, stride_sp, BLOCK_N: tl.constexpr, WIDE: tl.constexpr,
                      Q_PER_TILE: tl.constexpr):  # fmt: skip
    """G: the strided grid split into a scoring loop over the row's counted tiles (no branch) and a -inf loop over
    the rest, starting at this program's first tile past the count; Q_PER_TILE loads q inside the scoring loop."""
    bid = tl.program_id(1)
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    count = tl.load(counts_ptr + bid)
    out_row = row_base(out_scores_ptr, bid, stride_sb, WIDE)
    d_offsets = tl.arange(0, D_PAD)
    d_in = d_offsets < D
    n_score = tl.cdiv(count, BLOCK_N)
    if not Q_PER_TILE:
        q0 = tl.load(
            query_ptr + bid * stride_qb + d_offsets * stride_qd, mask=d_in, other=0.0
        ).to(tl.float32)
    for tile_id in range(pid, n_score, stride):
        n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
        in_count = n_offsets < count
        if Q_PER_TILE:
            q = tl.load(
                query_ptr + bid * stride_qb + d_offsets * stride_qd,
                mask=d_in,
                other=0.0,
            ).to(tl.float32)
        else:
            q = q0
        item_ids = tl.load(row_base(pos_indices_ptr, bid, stride_pb, WIDE) + n_offsets * stride_pp,
                           mask=in_count, other=0)  # fmt: skip
        emb_rows = tl.load(item_embs_ptr + item_ids[:, None] * stride_in + d_offsets[None, :] * stride_id,
                           mask=in_count[:, None] & d_in[None, :], other=0.0).to(tl.float32)  # fmt: skip
        dots = tl.sum(emb_rows * q[None, :], axis=1)
        dots = tl.where(in_count, dots, float("-inf"))
        tl.store(out_row + n_offsets * stride_sp, dots, mask=n_offsets < P)
    start = pid + tl.cdiv(tl.maximum(n_score - pid, 0), stride) * stride
    for tile_id in range(start, tl.cdiv(P, BLOCK_N), stride):
        n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
        inf = tl.full((BLOCK_N,), float("-inf"), tl.float32)
        tl.store(out_row + n_offsets * stride_sp, inf, mask=n_offsets < P)


@triton.jit
def fmkt_split_fill_kernel(query_ptr, item_embs_ptr, pos_indices_ptr, counts_ptr, out_scores_ptr, P: tl.constexpr,
                           D: tl.constexpr, D_PAD: tl.constexpr, stride_qb, stride_qd, stride_in, stride_id,
                           stride_pb, stride_pp, stride_sb, stride_sp, BLOCK_N: tl.constexpr, WIDE: tl.constexpr,
                           FILL_N: tl.constexpr):  # fmt: skip
    """H: Gq's split loops, but the -inf fill covers [n_score * BLOCK_N, P) in FILL_N-lane chunks, strided over
    the programs, so its cost does not grow as BLOCK_N shrinks."""
    bid = tl.program_id(1)
    pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    count = tl.load(counts_ptr + bid)
    out_row = row_base(out_scores_ptr, bid, stride_sb, WIDE)
    d_offsets = tl.arange(0, D_PAD)
    d_in = d_offsets < D
    n_score = tl.cdiv(count, BLOCK_N)
    for tile_id in range(pid, n_score, n_prog):
        n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
        in_count = n_offsets < count
        q = tl.load(
            query_ptr + bid * stride_qb + d_offsets * stride_qd, mask=d_in, other=0.0
        ).to(tl.float32)
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


def launch(variant, q, items, cand, counts):
    """``(kernel, launch)`` for a variant name: before / A / B / C / D / D1, or E / F / G / Gq with an optional
    register cap suffix (E = v2.2's body, F = D's body, G / Gq = the split loops; "E" = 64 registers, "E80" = 80,
    "G0" = uncapped), optionally followed by ":w<num_warps>:b<block_n>:g<programs>" (e.g. "Gq0:w16:b32")."""
    v22, old = MODS["before"], MODS["old"]
    name, *opts = variant.split(":")
    cfg = v22.DEFAULT_CONFIG
    if name in ("A", "D1"):
        cfg = dataclasses.replace(cfg, num_stages=1)
    names = {"w": "num_warps", "b": "block_n", "g": "programs"}
    for o in opts:
        cfg = dataclasses.replace(cfg, **{names[o[0]]: int(o[1:])})
    kern, la = launch_of_mod(
        old if name in ("B", "C") else v22, q, items, cand, counts, cfg
    )
    if name[0] in "EFG":  # H is never capped
        cap = name.lstrip("EFGq")
        if cap != "0":
            la.kwargs["maxnreg"] = int(cap or 64)
    if name == "B":
        kern = fmkt_exit_kernel
    elif name[0] in "DF":
        kern = fmkt_strided_qin_kernel
    elif name[0] == "G":
        kern = fmkt_split_kernel
        la.kwargs["Q_PER_TILE"] = name.startswith("Gq")
    elif (
        name[0] == "H"
    ):  # H = the split loops with a FILL_N-lane -inf fill, "H" = FILL_N 1024
        kern = fmkt_split_fill_kernel
        la.kwargs["FILL_N"] = int(name[1:] or 1024)
    return kern, la


def launch_of_mod(mod, q, items, cand, counts, cfg):
    return mod._fused_masked_knn_topk_kernel, mod._fmkt_prep(
        q, items, cand, counts, cfg, bucket=False
    )


ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("d", type=int)
ap.add_argument("out")
ap.add_argument("--rates", default="0.0002,0.001,0.01,0.1,1.0")
ap.add_argument("--pairs", type=int, default=8)
ap.add_argument("--variants", default="A,B,C,D,D1")
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
for p in map(float, args.rates.split(",")):
    cand16, counts16 = candidates(args.n, 16, p, seed=1)
    if args.skew:
        p_hi, m = args.skew.split(":")
        hi, hi_counts = candidates(args.n, int(m), float(p_hi), seed=2)
        cand16[: int(m)], counts16[: int(m)] = hi, hi_counts
    for bs in (1, 16):
        cand, counts = cand16[:bs].contiguous(), counts16[:bs].contiguous()
        ls = {v: [launch(v, queries[i, :bs].contiguous(), items, cand, counts) for i in range(POOL)]
              for v in ("before", *VARIANTS)}  # fmt: skip
        eq = {}
        for v in VARIANTS:
            same = True
            for (kb, lb), (kv, lv) in zip(ls["before"], ls[v], strict=True):
                kb[lb.grid](**lb.kwargs)
                kv[lv.grid](**lv.kwargs)
                same &= torch.equal(lb.all_scores, lv.all_scores)
            eq[v] = same
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
               "equal": eq, "sm_mhz": [min(mhz), max(mhz)]}  # fmt: skip
        rows.append(row)
        print(f"p={p} bs={bs}: before {row['ms']['before']:.3f} ms | "
              + " | ".join(f"{v} {row['ms'][v]:.3f} ({row['over_before'][v][0]:.3f} [{row['over_before'][v][1]:.3f},"
                           f" {row['over_before'][v][2]:.3f}]) eq={eq[v]}" for v in VARIANTS)
              + f" | sm {row['sm_mhz']}", flush=True)  # fmt: skip
        del graphs, ls
        torch.cuda.empty_cache()
json.dump(rows, open(args.out, "w"), indent=1)
