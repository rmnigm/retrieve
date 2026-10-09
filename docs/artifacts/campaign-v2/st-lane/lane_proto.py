"""ST-LANE prototype: the v2.4 bloom scorer with a compacted-lane path for rows whose pass-rate bound is under `P_MAX`
(kernels.md § SilverTorch kernels, "Gated tile skip" supplies the bound). In such a program: store -inf over the tile's
valid slots, scan `keep` into a per-program scratch slot of passing lane indices, then loop over chunks of `C` passing lanes:
gather their code rows, one int32 dot, store their scores. Rows above `P_MAX` run the v2.4 body. Bit-exact by construction
(the same int32 dot and fp32 epilogue per lane); every config's `[B, width]` scores are checked `torch.equal` to v2.4's.

Kernel-only against the library kernel (v2.4), ABAB graphs, on dloop_gate.py's index (one clause at item pass rate p).

    PYTHONPATH=retrieve/src:retrieve:docs/artifacts/campaign-v2/st-dloop:<v21 pkg> python lane_proto.py out.json \
        [--d 128] [--configs C:P_MAX:WARPS,...]
"""

import argparse
import importlib
import json
import statistics

import torch
import triton
import triton.language as tl
from dloop_gate import (
    DEV,
    POOL,
    build_index,
    filtered,
    graph_of,
    ratio_ci,
    sm_mhz,
    window_us,
)

from retrieve.ops.triton.common import probe_dots, probe_tile, row_base

cps = importlib.import_module("retrieve.ops.triton.codesigned_probe_score")
host = importlib.import_module("retrieve.ops.triton._host")


@triton.jit
def lane_kernel(q_codes_ptr, q_scales_ptr, probe_ids_ptr, offsets_ptr, item_codes_ptr, qpos_ptr, bloom_t_ptr,
                bit_freq_ptr, out_scores_ptr, global_scale, n_probe, width, tiles_y, n_qbits, D: tl.constexpr,
                D_PAD: tl.constexpr, NPP: tl.constexpr, stride_qcb, stride_cn, stride_qpos, stride_tm, stride_ob,
                HAS_QB: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_D: tl.constexpr, SKIP: tl.constexpr,
                GATED: tl.constexpr, NQB: tl.constexpr, WIDE: tl.constexpr, scratch_ptr, p_max,
                C: tl.constexpr):  # fmt: skip
    bid = tl.program_id(0)
    t = tl.program_id(2) * tiles_y + tl.program_id(1)
    pos, slot, valid, tail = probe_tile(
        probe_ids_ptr, offsets_ptr, bid, t, n_probe, NPP, BLOCK_P
    )
    out_row = row_base(out_scores_ptr, bid, stride_ob, WIDE)
    qi = tl.arange(0, NQB)
    qbits = tl.load(qpos_ptr + bid * stride_qpos + qi, mask=qi < n_qbits, other=-1)
    bound = tl.min(tl.load(bit_freq_ptr + qbits, mask=qbits >= 0, other=1.0), axis=0)
    if tail:
        tl.store(
            out_row + slot,
            tl.full([BLOCK_P], float("-inf"), tl.float32),
            mask=slot < width,
        )
    else:
        keep = valid
        word = pos >> 6
        bit = pos & 63
        for i in range(n_qbits):
            m = tl.load(qpos_ptr + bid * stride_qpos + i)
            w = tl.load(
                bloom_t_ptr + m * stride_tm + word, mask=keep & (m >= 0), other=-1
            )
            keep = keep & (((w >> bit) & 1) != 0)
        if bound < p_max:
            lane = tl.arange(0, BLOCK_P)
            tl.store(
                out_row + slot,
                tl.full([BLOCK_P], float("-inf"), tl.float32),
                mask=valid,
            )
            k32 = keep.to(tl.int32)
            dest = tl.cumsum(k32, 0) - 1
            n_pass = tl.sum(k32, axis=0)
            prog = (
                bid * tl.num_programs(2) + tl.program_id(2)
            ) * tiles_y + tl.program_id(1)
            sp = scratch_ptr + prog.to(tl.int64) * BLOCK_P
            tl.store(sp + dest, lane, mask=keep)
            tl.debug_barrier()
            base_pos = tl.sum(tl.where(lane == 0, pos, 0), axis=0)
            base_slot = tl.sum(tl.where(lane == 0, slot, 0), axis=0)
            d_off = tl.arange(0, D_PAD)
            d_in = d_off < D
            q_codes = tl.load(
                q_codes_ptr + bid * stride_qcb + d_off, mask=d_in, other=0
            )
            q_scale = tl.load(q_scales_ptr + bid)
            for c0 in range(0, n_pass, C):
                ci = c0 + tl.arange(0, C)
                cm = ci < n_pass
                cl = tl.load(sp + ci, mask=cm, other=0)
                codes = tl.load(item_codes_ptr + (base_pos + cl)[:, None] * stride_cn + d_off[None, :],
                                mask=cm[:, None] & d_in[None, :], other=0)  # fmt: skip
                dots_i32 = tl.sum(
                    tl.dot(q_codes[None, :], tl.trans(codes), out_dtype=tl.int32),
                    axis=0,
                )
                dots = (
                    dots_i32.to(tl.float32)
                    * tl.cast(q_scale, tl.float32)
                    * tl.cast(global_scale, tl.float32)
                )
                tl.store(out_row + base_slot + cl, dots, mask=cm)
        else:
            dots_i32, q_scale = probe_dots(q_codes_ptr + bid * stride_qcb, q_scales_ptr, bid, item_codes_ptr, pos,
                                           stride_cn, keep, D, D_PAD, BLOCK_D, BLOCK_P)  # fmt: skip
            dots = (
                dots_i32.to(tl.float32)
                * tl.cast(q_scale, tl.float32)
                * tl.cast(global_scale, tl.float32)
            )
            dots = tl.where(keep, dots, float("-inf"))
            tl.store(out_row + slot, dots, mask=valid)


ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=2_000_000)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--rates", default="0.001,0.003,0.01,0.03,0.136,0.3,1.0")
ap.add_argument("--configs", default="32:0.25:4,64:0.25:4,32:0.25:8")
ap.add_argument("--pairs", type=int, default=8)
args = ap.parse_args()
base = build_index(args.n, args.d)
g = torch.Generator(device=DEV).manual_seed(1)
pool16 = [torch.randn(16, args.d, device=DEV, generator=g) for _ in range(POOL)]
qa = torch.ones(16, 1, dtype=torch.long, device=DEV)
width = base._probe_width
rows = []
for p in map(float, args.rates.split(",")):
    ga = torch.Generator(device=DEV).manual_seed(2)
    attrs = (torch.rand(args.n, 1, 1, device=DEV, generator=ga) < p).long()
    m = filtered(base, "bloom", attrs)
    for bs in (1, 16):
        pool_q = [q[:bs].contiguous() for q in pool16]
        probes = [m._phase1_probe_ids(q) for q in pool_q]
        cfg = host.tile_for_width(cps.CONFIGS, args.d, bs, width)
        launches = [cps._cps_prep(q, pr, m.cluster_offsets, m.item_codes, m.sort_perm, m._global_scale_f, width,
                                  query_bit_positions=m._query_bit_positions(qa[:bs]),
                                  bloom_transposed=m.bloom_transposed, cfg=cfg, bit_freq=m.bloom_bit_freq)
                    for q, pr in zip(pool_q, probes, strict=True)]  # fmt: skip
        lib = [(cps._codesigned_probe_score_kernel, la) for la in launches]
        for kb, lb in lib:
            kb[lb.grid](**lb.kwargs)
        ref = [la.all_scores.clone() for la in launches]
        gb = graph_of(lib)
        for spec in args.configs.split(","):
            c, p_max, nw = spec.split(":")
            ls = []
            for la in launches:
                b_, ty, tx = la.grid
                scratch = torch.empty(
                    b_ * ty * tx * la.kwargs["BLOCK_P"], dtype=torch.int32, device=DEV
                )
                out = torch.empty_like(la.all_scores)
                kw = dict(la.kwargs, out_scores_ptr=out, scratch_ptr=scratch, p_max=float(p_max), C=int(c),
                          num_warps=int(nw))  # fmt: skip
                ls.append((lane_kernel, type(la)(la.grid, kw, out)))
            for k_, l_ in ls:
                k_[l_.grid](**l_.kwargs)
            same = all(
                torch.equal(l_.all_scores, r)
                for (_, l_), r in zip(ls, ref, strict=True)
            )
            ga_ = graph_of(ls)
            for _ in range(2):
                window_us(gb), window_us(ga_)
            tb, ta, mhz = [], [], []
            for _ in range(args.pairs):
                tb.append(window_us(gb))
                ta.append(window_us(ga_))
                mhz.append(sm_mhz())
            r, lo, hi = ratio_ci(tb, ta)
            row = {"p": p, "bs": bs, "cfg": spec, "v24_us": statistics.median(tb), "lane_us": statistics.median(ta),
                   "ratio": [r, lo, hi], "equal": same, "sm_mhz": [min(mhz), max(mhz)]}  # fmt: skip
            rows.append(row)
            print(f"p={p} bs={bs} {spec}: v2.4 {row['v24_us']:.1f} lane {row['lane_us']:.1f} us ratio {r:.3f} "
                  f"[{lo:.3f}, {hi:.3f}] equal {same} sm {row['sm_mhz']}", flush=True)  # fmt: skip
            del ga_
        del gb
    del m
    torch.cuda.empty_cache()
json.dump(
    {"n": args.n, "d": args.d, "width": width, "rows": rows},
    open(args.out, "w"),
    indent=1,
)
