"""ST-DLOOP: the d768 tile against batch size, kernel-only (dloop_gate.py's graph method and index), to test
the bs 1 regression's mechanism (too few programs with a 256-item tile on a narrow probe width): the scorer
at each config and bs, none / bloom / exact, against v2.1's kernel at the same cell, ABAB `--pairs` pairs.

    PYTHONPATH=<v21 pkg>:retrieve/src python bs_sweep.py out.json [--n 2000000] [--p 0.01] \
        [--bs 1,2,4,8,16] [--grid bp:nw:ns:bd,...]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from pathlib import Path

import torch
from dloop_gate import (
    ARMS,
    DEV,
    POOL,
    build_index,
    filtered,
    graph_of,
    prep,
    ratio_ci,
    sm_mhz,
    window_us,
)

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=2_000_000)
ap.add_argument("--d", type=int, default=768)
ap.add_argument("--p", type=float, default=0.01)
ap.add_argument("--bs", default="1,2,4,8,16")
ap.add_argument("--modes", default="none,bloom,exact")
ap.add_argument(
    "--grid", default="64:4:2:128,128:4:2:128,256:4:2:128,64:4:2:256,128:4:2:256"
)
ap.add_argument("--pairs", type=int, default=8)
args = ap.parse_args()

base = build_index(args.n, args.d)
g = torch.Generator(device=DEV).manual_seed(1)
pool16 = [torch.randn(16, args.d, device=DEV, generator=g) for _ in range(POOL)]
qa = torch.ones(16, 1, dtype=torch.long, device=DEV)
ga = torch.Generator(device=DEV).manual_seed(2)
attrs = (torch.rand(args.n, 1, 1, device=DEV, generator=ga) < args.p).long()
width = base._probe_width
res = {"n": args.n, "d": args.d, "p": args.p, "width": width, "rows": []}
print(f"width {width}", flush=True)
for mode in args.modes.split(","):
    m = filtered(base, mode, attrs)
    mod = ARMS["after"][mode]
    for bs in map(int, args.bs.split(",")):
        pool_q = [q[:bs].contiguous() for q in pool16]
        probes = [m._phase1_probe_ids(q) for q in pool_q]
        before = [prep(ARMS["before"][mode], mode, m, q, pr, qa[:bs], width)
                  for q, pr in zip(pool_q, probes, strict=True)]  # fmt: skip
        gb = graph_of(before)
        for e in args.grid.split(","):
            bp, nw, ns, bd = map(int, e.split(":"))
            shipped = mod.CONFIGS[1024][0]
            cfg = dataclasses.replace(
                shipped, block_p=bp, num_warps=nw, num_stages=ns, block_d=bd
            )
            ls = [prep(mod, mode, m, q, pr, qa[:bs], width, cfg)
                  for q, pr in zip(pool_q, probes, strict=True)]  # fmt: skip
            ga_ = graph_of(ls)
            for _ in range(2):
                window_us(gb), window_us(ga_)
            tb, ta, mhz = [], [], []
            for _ in range(args.pairs):
                tb.append(window_us(gb))
                ta.append(window_us(ga_))
                mhz.append(sm_mhz())
            r, lo, hi = ratio_ci(tb, ta)
            programs = ls[0][1].grid[0] * ls[0][1].grid[1] * ls[0][1].grid[2]
            row = {"mode": mode, "bs": bs, "cfg": e, "programs": programs,
                   "before_us": statistics.median(tb), "after_us": statistics.median(ta),
                   "ratio": r, "ci": [lo, hi], "sm_mhz": [min(mhz), max(mhz)]}  # fmt: skip
            res["rows"].append(row)
            print(f"{mode:5s} bs={bs:2d} {e:14s} programs {programs:6d} before {row['before_us']:6.1f} "
                  f"after {row['after_us']:6.1f} us  ratio {r:.3f} [{lo:.3f}, {hi:.3f}] sm {row['sm_mhz']}",
                  flush=True)  # fmt: skip
            del ga_, ls
        del gb
Path(args.out).write_text(json.dumps(res, indent=1))
