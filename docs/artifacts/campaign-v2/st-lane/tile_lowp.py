"""ST-LANE: does a different d128 bloom tile move the per-program latency floor at low p? The library (v2.4) bloom scorer
at its shipped tile (256 x 4) against config overrides, kernel-only ABAB, with the shipped pass-rate gate and table. Every
override's scores are torch.equal to the shipped tile's (the tiling only moves slots).

    PYTHONPATH=retrieve/src:retrieve:docs/artifacts/campaign-v2/st-dloop:<v21 pkg> python tile_lowp.py out.json
"""

import argparse
import dataclasses
import importlib
import json
import statistics

import torch
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

cps = importlib.import_module("retrieve.ops.triton.codesigned_probe_score")
host = importlib.import_module("retrieve.ops.triton._host")

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--rates", default="0.001,0.01,0.136,1.0")
ap.add_argument("--tiles", default="128:4,256:8,512:4,512:8,1024:8")
ap.add_argument("--pairs", type=int, default=8)
args = ap.parse_args()
base = build_index(2_000_000, 128)
g = torch.Generator(device=DEV).manual_seed(1)
pool16 = [torch.randn(16, 128, device=DEV, generator=g) for _ in range(POOL)]
qa = torch.ones(16, 1, dtype=torch.long, device=DEV)
width = base._probe_width
rows = []
for p in map(float, args.rates.split(",")):
    ga = torch.Generator(device=DEV).manual_seed(2)
    m = filtered(
        base,
        "bloom",
        (torch.rand(2_000_000, 1, 1, device=DEV, generator=ga) < p).long(),
    )
    for bs in (1, 16):
        pool_q = [q[:bs].contiguous() for q in pool16]
        probes = [m._phase1_probe_ids(q) for q in pool_q]

        def launches(cfg):
            return [(cps._codesigned_probe_score_kernel, cps._cps_prep(
                q, pr, m.cluster_offsets, m.item_codes, m.sort_perm, m._global_scale_f, width,
                query_bit_positions=m._query_bit_positions(qa[:bs]), bloom_transposed=m.bloom_transposed, cfg=cfg,
                bit_freq=m.bloom_bit_freq)) for q, pr in zip(pool_q, probes, strict=True)]  # fmt: skip

        shipped = host.tile_for_width(cps.CONFIGS, 128, bs, width)
        lb = launches(shipped)
        for k, la in lb:
            k[la.grid](**la.kwargs)
        ref = [torch.topk(la.all_scores, 100, dim=1).values for _, la in lb]
        gb = graph_of(lb)
        for spec in args.tiles.split(","):
            bp, nw = map(int, spec.split(":"))
            la_ = launches(dataclasses.replace(shipped, block_p=bp, num_warps=nw))
            for k, la in la_:
                k[la.grid](**la.kwargs)
            same = all(torch.equal(torch.topk(la.all_scores, 100, dim=1).values, r)
                       for (_, la), r in zip(la_, ref, strict=True))  # fmt: skip
            gt = graph_of(la_)
            for _ in range(2):
                window_us(gb), window_us(gt)
            tb, ta, mhz = [], [], []
            for _ in range(args.pairs):
                tb.append(window_us(gb))
                ta.append(window_us(gt))
                mhz.append(sm_mhz())
            r, lo, hi = ratio_ci(tb, ta)
            row = {"p": p, "bs": bs, "tile": spec, "gated": bool(la_[0][1].kwargs["GATED"]),
                   "shipped_us": statistics.median(tb), "tile_us": statistics.median(ta), "ratio": [r, lo, hi],
                   "topk_scores_equal": same, "sm_mhz": [min(mhz), max(mhz)]}  # fmt: skip
            rows.append(row)
            print(f"p={p} bs={bs} {spec} gated={row['gated']}: shipped {row['shipped_us']:.1f} tile {row['tile_us']:.1f} us "
                  f"ratio {r:.3f} [{lo:.3f}, {hi:.3f}] eq={same}", flush=True)  # fmt: skip
json.dump(rows, open(args.out, "w"), indent=1)
