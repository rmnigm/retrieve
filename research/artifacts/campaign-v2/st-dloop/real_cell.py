"""ST-DLOOP on the real D3 cell: SilverTorch at the bench's build (n_lists 1024, n_probe 24, bloom
1024 bits / k_hash 5, seed 0), a 16-query pool from ``inputs.query_pool`` (seed 0), k 100.

  profile  torch.profiler kernel split, eager, per arm (triton, official int32 + fp16): every CUDA
           kernel's total device time over CALLS calls, per call; the whole stream window too.
  sweep    the Triton scorer ``_impl`` alone over tile configs on the batches' real probe ids and
           query bit positions; every config's ids and scores torch.equal to the shipped one.

    cd evaluation && python real_cell.py profile pubmed 768 c0_mesh out.json [--modes none,bloom]
    cd evaluation && python real_cell.py sweep pubmed 768 c0_mesh out.json --grid bp:nw:ns:bd,...
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from bench_kernels import time_fn

from bench import algos, config, inputs

cps = importlib.import_module("retrieve.ops.triton.codesigned_probe_score")
cpse = importlib.import_module("retrieve.ops.triton.codesigned_probe_score_exact")
from retrieve.ops.triton._host import width_tiles  # noqa: E402

DEV = torch.device("cuda")
CALLS = 50
N_POOL = 16

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["profile", "sweep"])
ap.add_argument("dataset")
ap.add_argument("dim", type=int)
ap.add_argument("sweep")
ap.add_argument("out")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--modes", default="bloom")
ap.add_argument("--backends", default="triton,official")
ap.add_argument("--grid", default="")
args = ap.parse_args()

ds = config.load_dataset(Path(f"config/{args.dataset}.yaml"), args.dim)
inp = inputs.load_inputs(ds, DEV, with_filters=True)
out: dict = {"cell": vars(args), "device": torch.cuda.get_device_name(DEV), "arms": {}}


def build(mode: str, backend: str, score_path: str | None = None) -> torch.nn.Module:
    kind = {"none": "none", "bloom": "bloom", "exact": "clause"}[mode]
    params = {} if score_path is None else {"score_path": score_path}
    return algos.build(
        "silvertorch", inp["item_embs"], k=args.k, backend=backend, filter_kind=kind,
        item_attrs=inp["item_attrs"], clause_is_reverse=inp["clause_is_reverse"],
        params=params, seed=0,
    )  # fmt: skip


def pool(mode: str):
    kind = "clause" if mode == "exact" else mode
    clauses = None if mode == "none" else ds.clauses[kind][args.sweep]
    qa, skip = inputs.sweep_qa(inp["qa"], clauses)
    q, a = inputs.query_pool(
        inp, qa, skip, bs=args.bs, seed=0, n_pool=N_POOL, device=DEV
    )
    return [(q[i], None if a is None else a[i]) for i in range(N_POOL)]


def kernel_split(fn) -> dict:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(CALLS):
            fn()
        torch.cuda.synchronize()
    per: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            per[ev.name][0] += ev.device_time
            per[ev.name][1] += 1
    rows = sorted(
        ({"kernel": k[:140], "us_per_call": v[0] / CALLS, "launches_per_call": v[1] / CALLS}
         for k, v in per.items()),
        key=lambda r: -r["us_per_call"],
    )  # fmt: skip
    return {"kernels": rows, "sum_us_per_call": sum(r["us_per_call"] for r in rows)}


for mode in args.modes.split(","):
    batches = pool(mode)
    arms = []
    for backend in args.backends.split(","):
        if backend == "official":
            arms += [
                (f"official-{sp}", build(mode, "official", sp))
                for sp in ("int32", "fp16")
            ]
        else:
            arms.append((backend, build(mode, backend)))
    for name, mod in arms:
        it = iter(range(10**9))

        def call(mod=mod, it=it):
            q, a = batches[next(it) % N_POOL]
            return mod(q, a)

        with torch.no_grad():
            if args.what == "profile":
                r = {"wall": time_fn(torch, call), **kernel_split(call)}
                out["arms"][f"{mode}/{name}"] = r
                print(f"== {mode}/{name}: wall {r['wall']['us_median']:.1f} us, kernels "
                      f"{r['sum_us_per_call']:.1f} us/call, sm {r['wall']['sm_mhz']}", flush=True)  # fmt: skip
                for row in r["kernels"][:12]:
                    print(f"   {row['us_per_call']:9.1f} us  x{row['launches_per_call']:.0f}  "
                          f"{row['kernel'][:100]}", flush=True)  # fmt: skip
                continue
            if name.startswith("official"):
                continue
            # sweep: the scorer alone on the module's real phase-1 output
            calls = []
            for q, a in batches:
                lay = (q, mod._phase1_probe_ids(q), mod.cluster_offsets, mod.item_codes,
                       mod.sort_perm, mod._global_scale_f, mod.k, mod._probe_width)  # fmt: skip
                if mode == "bloom":
                    kw = dict(query_bit_positions=mod._query_bit_positions(a),
                              bloom_transposed=mod.bloom_transposed)  # fmt: skip
                    calls.append((cps._codesigned_probe_score_impl, lay, kw))
                elif mode == "exact":
                    kw = dict(item_clause_attrs=mod.item_clause_attrs,
                              clause_is_reverse=mod.clause_is_reverse,
                              query_clause_attrs=a.long())  # fmt: skip
                    calls.append((cpse._codesigned_probe_score_exact_impl, lay, kw))
                else:
                    calls.append((cps._codesigned_probe_score_impl, lay, {}))
            kmod = cpse if mode == "exact" else cps
            shipped = width_tiles(kmod.CONFIGS, args.dim)[
                0
            ]  # the largest tile, the sweep's incumbent
            cfgs = [shipped] + [
                type(shipped)(*map(int, e.split(":")))
                for e in args.grid.split(",")
                if e
            ]
            refs = [f(*lay, **kw, config=shipped) for f, lay, kw in calls]
            for cfg in cfgs:
                same = all(
                    torch.equal(i, ri) and torch.equal(s, rs)
                    for (f, lay, kw), (ri, rs) in zip(calls, refs, strict=True)
                    for i, s in [f(*lay, **kw, config=cfg)]
                )
                cyc = iter(range(10**9))
                r = time_fn(torch, lambda: (lambda c: c[0](*c[1], **c[2], config=cfg))(
                    calls[next(cyc) % N_POOL]))  # fmt: skip
                key = f"{mode}/bp{cfg.block_p}_w{cfg.num_warps}_s{cfg.num_stages}_bd{getattr(cfg, 'block_d', '-')}"
                out["arms"][key] = {
                    **r,
                    "equal_shipped": same,
                    "shipped": cfg == shipped,
                }
                print(f"{key:36s} {r['us_median']:9.1f} us  sm {r['sm_mhz']} eq={same}"
                      f"{'  (shipped)' if cfg == shipped else ''}", flush=True)  # fmt: skip
        del mod
        torch.cuda.empty_cache()
json.dump(out, open(args.out, "w"), indent=1)
