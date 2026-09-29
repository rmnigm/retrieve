"""Fix 1 tile sweep: kernel-only `_impl` time of the three probe-scorer variants (none, bloom,
exact) over (BLOCK_P, num_warps) at D in {128, 192, 384, 768}, B in {1, 16}. Layout as the L4
gate (docs/artifacts/l4-pow2-pad/bench_d128.py): 1024 skewed clusters, max 5000, n_probe 24,
k 100. Timing: kernel-opt's time_fn. Each config's result is checked torch.equal against the
(256, 4) run at the same shape before it is timed.

    PYTHONPATH=retrieve:docs/artifacts/kernel-opt python tile_sweep.py out.json [--d 128,768]
"""

import argparse
import json

import torch
from bench_kernels import time_fn

from retrieve.indexing.quantize import quantize_int8_global
from retrieve.ops.triton.codesigned_probe_score import (
    CodesignedProbeScoreConfig,
    _codesigned_probe_score_impl,
)
from retrieve.ops.triton.codesigned_probe_score_exact import (
    CodesignedProbeScoreExactConfig,
    _codesigned_probe_score_exact_impl,
)
from tests.parity.conftest import make_bloom, make_exact, make_probe_family

GRID = [(bp, nw) for bp in (16, 32, 64, 128, 256) for nw in (4, 8)]

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--d", default="128,192,384,768")
ap.add_argument("--b", default="1,16")
ap.add_argument("--grid", default=None, help="bp:nw,bp:nw")
args = ap.parse_args()
grid = [tuple(map(int, e.split(":"))) for e in args.grid.split(",")] if args.grid else GRID

dev = torch.device("cuda")
out = {"cases": {}}
for b in map(int, args.b.split(",")):
    lay = make_probe_family(b, 1024, 5000, 24)
    n = lay.n
    qpos, bt, _, _ = make_bloom(n, b, m_bits=1024)
    attrs, rev, qa = make_exact(n, b)
    for d in map(int, args.d.split(",")):
        g = torch.Generator(device=dev).manual_seed(0)
        codes, gs = quantize_int8_global(torch.randn(n, d, device=dev, generator=g))
        q = torch.randn(b, d, device=dev, generator=g)
        csr = (q, lay.probe_ids, lay.cluster_offsets, codes, lay.sort_perm, gs, 100, lay.width)
        variants = {
            "none": lambda cfg: _codesigned_probe_score_impl(
                *csr, config=CodesignedProbeScoreConfig(*cfg)
            ),
            "bloom": lambda cfg: _codesigned_probe_score_impl(
                *csr,
                query_bit_positions=qpos,
                bloom_transposed=bt,
                config=CodesignedProbeScoreConfig(*cfg),
            ),
            "exact": lambda cfg: _codesigned_probe_score_exact_impl(
                *csr,
                item_clause_attrs=attrs,
                clause_is_reverse=rev,
                query_clause_attrs=qa,
                config=CodesignedProbeScoreExactConfig(*cfg),
            ),
        }
        for v, run in variants.items():
            ref_ids, ref_s = run((256, 4))
            for cfg in grid:
                ids, s = run(cfg)
                same = torch.equal(s, ref_s)
                r = time_fn(torch, lambda: run(cfg))
                key = f"{v}_d{d}_b{b}_bp{cfg[0]}_w{cfg[1]}"
                out["cases"][key] = {**r, "scores_equal_256w4": same}
                print(
                    f"{key:28s} {r['us_median']:10.1f} us  spread {r['spread']:.3f} "
                    f"sm {r['sm_mhz'][0]}-{r['sm_mhz'][1]} eq={same}",
                    flush=True,
                )
        del codes
with open(args.out, "w") as fh:
    json.dump(out, fh, indent=1)
