"""ST-LANE, measured first: where the d128 bloom scorer's time goes as p falls, by ablation. A copy of the v2.4 bloom
scorer with constexpr switches is launched with the library's own prep (D_PAD <= 256 tile, 256 x 4, one dot):

  full     the v2.4 body (BLOOM 1, DOT 2, STORE 1)
  nodot    codes still loaded and consumed (a cheap int reduction), no tl.dot
  noload   no code load, no dot (scores = 0 where kept)
  nobloom  keep = valid (no bloom test), full dot
  nostore  the score store guarded by a runtime-false condition (so the work is not dead-code eliminated)
  empty    probe table + -inf store only
  bloompar the full body with the bloom word loads masked by `valid` (independent loads), not by the running `keep`

Each is timed kernel-only (CUDA graph of 64 launches over 8 batches, median of `--windows` windows) on dloop_gate.py's
index (N 2 M, n_lists 1024, n_probe 24, one clause at item pass rate p); Meta's scorer (`process_cluster*`) and its
phase-2 mask kernels on the same cells by torch.profiler, for reference. Not bit-exact variants: attribution only.

    PYTHONPATH=retrieve/src:retrieve:docs/artifacts/campaign-v2/st-dloop:<v21 pkg> python kernel_split.py out.json
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
    official_us,
    sm_mhz,
    window_us,
)

from retrieve import OfficialConfig
from retrieve.ops.triton.common import probe_dots, probe_tile, row_base

cps = importlib.import_module("retrieve.ops.triton.codesigned_probe_score")
host = importlib.import_module("retrieve.ops.triton._host")


@triton.jit
def split_kernel(q_codes_ptr, q_scales_ptr, probe_ids_ptr, offsets_ptr, item_codes_ptr, qpos_ptr, bloom_t_ptr,
                 bit_freq_ptr, out_scores_ptr, global_scale, n_probe, width, tiles_y, n_qbits, D: tl.constexpr,
                 D_PAD: tl.constexpr, NPP: tl.constexpr, stride_qcb, stride_cn, stride_qpos, stride_tm, stride_ob,
                 HAS_QB: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_D: tl.constexpr, SKIP: tl.constexpr,
                 GATED: tl.constexpr, NQB: tl.constexpr, WIDE: tl.constexpr, BLOOM: tl.constexpr,
                 DOT: tl.constexpr, STORE: tl.constexpr):  # fmt: skip
    bid = tl.program_id(0)
    t = tl.program_id(2) * tiles_y + tl.program_id(1)
    pos, slot, valid, tail = probe_tile(
        probe_ids_ptr, offsets_ptr, bid, t, n_probe, NPP, BLOCK_P
    )
    out_row = row_base(out_scores_ptr, bid, stride_ob, WIDE)
    if tail:
        tl.store(
            out_row + slot,
            tl.full([BLOCK_P], float("-inf"), tl.float32),
            mask=slot < width,
        )
    else:
        keep = valid
        if BLOOM:
            word = pos >> 6
            bit = pos & 63
            for i in range(n_qbits):
                m = tl.load(qpos_ptr + bid * stride_qpos + i)
                w = tl.load(
                    bloom_t_ptr + m * stride_tm + word,
                    mask=(keep if BLOOM == 1 else valid)
                    & (m >= 0),  # BLOOM 2: independent loads
                    other=-1,
                )
                keep = keep & (((w >> bit) & 1) != 0)
        if DOT == 2:
            dots_i32, q_scale = probe_dots(q_codes_ptr + bid * stride_qcb, q_scales_ptr, bid, item_codes_ptr, pos,
                                           stride_cn, keep, D, D_PAD, BLOCK_D, BLOCK_P)  # fmt: skip
        elif DOT == 1:
            d_off = tl.arange(0, D_PAD)
            codes = tl.load(item_codes_ptr + pos[:, None] * stride_cn + d_off[None, :],
                            mask=keep[:, None] & (d_off < D)[None, :], other=0)  # fmt: skip
            dots_i32 = tl.sum(codes.to(tl.int32), axis=1)
            q_scale = tl.load(q_scales_ptr + bid)
        else:
            dots_i32 = tl.zeros([BLOCK_P], dtype=tl.int32)
            q_scale = tl.load(q_scales_ptr + bid)
        dots = (
            dots_i32.to(tl.float32)
            * tl.cast(q_scale, tl.float32)
            * tl.cast(global_scale, tl.float32)
        )
        dots = tl.where(keep, dots, float("-inf"))
        if STORE:
            tl.store(out_row + slot, dots, mask=valid)
        else:
            tl.store(out_row + slot, dots, mask=valid & (n_probe < 0))


VARIANTS = {"bloompar": (2, 2, 1), "full": (1, 2, 1), "nodot": (1, 1, 1), "noload": (1, 0, 1), "nobloom": (0, 2, 1), "nostore": (1, 2, 0),
            "empty": (0, 0, 1)}  # fmt: skip

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=2_000_000)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--rates", default="0.001,0.01,0.136,1.0")
ap.add_argument("--windows", type=int, default=8)
args = ap.parse_args()
base = build_index(args.n, args.d)
g = torch.Generator(device=DEV).manual_seed(1)
pool16 = [torch.randn(16, args.d, device=DEV, generator=g) for _ in range(POOL)]
qa = torch.ones(16, 1, dtype=torch.long, device=DEV)
ocfg = OfficialConfig(score_path="fp16", cache_plans=False)
width = base._probe_width
rows = []
for p in map(float, args.rates.split(",")):
    ga = torch.Generator(device=DEV).manual_seed(2)
    attrs = (torch.rand(args.n, 1, 1, device=DEV, generator=ga) < p).long()
    m = filtered(base, "bloom", attrs)
    m_off = filtered(base, "bloom", attrs, backend="official", official=ocfg)
    for bs in (1, 16):
        pool_q = [q[:bs].contiguous() for q in pool16]
        probes = [m._phase1_probe_ids(q) for q in pool_q]
        cfg = host.tile_for_width(cps.CONFIGS, args.d, bs, width)
        launches = [cps._cps_prep(q, pr, m.cluster_offsets, m.item_codes, m.sort_perm, m._global_scale_f, width,
                                  query_bit_positions=m._query_bit_positions(qa[:bs]),
                                  bloom_transposed=m.bloom_transposed, cfg=cfg, bit_freq=m.bloom_bit_freq)
                    for q, pr in zip(pool_q, probes, strict=True)]  # fmt: skip
        row = {"p": p, "bs": bs, "gated": bool(launches[0].kwargs["GATED"]), "us": {}}
        lib = graph_of([(cps._codesigned_probe_score_kernel, la) for la in launches])
        variants = {"library": lib}
        for name, (bl, dot, st) in VARIANTS.items():
            ls = []
            for la in launches:
                kw = dict(la.kwargs, BLOOM=bl, DOT=dot, STORE=st)
                ls.append((split_kernel, type(la)(la.grid, kw, la.all_scores)))
            variants[name] = graph_of(ls)
        for gr in variants.values():
            window_us(gr)
        t = {k: [] for k in variants}
        mhz = []
        for _ in range(args.windows):
            for k, gr in variants.items():
                t[k].append(window_us(gr))
            mhz.append(sm_mhz())
        row["us"] = {k: statistics.median(v) for k, v in t.items()}
        o = official_us(m_off, "bloom", pool_q, probes, qa, ocfg)
        row["official_scorer_us"], row["official_mask_us"] = o["scorer"], o["mask"]
        row["sm_mhz"] = [min(mhz), max(mhz)]
        rows.append(row)
        u = row["us"]
        print(f"p={p} bs={bs} gated={row['gated']}: " + " ".join(f"{k} {v:.1f}" for k, v in u.items())
              + f" | official scorer {o['scorer']:.1f} mask {o['mask']:.1f} | sm {row['sm_mhz']}", flush=True)  # fmt: skip
        del variants
    del m, m_off
    torch.cuda.empty_cache()
json.dump(
    {"n": args.n, "d": args.d, "width": width, "rows": rows},
    open(args.out, "w"),
    indent=1,
)
