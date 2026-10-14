"""CLAUSE-OPT microbench: SilverTorch (official-fork) exact clause filter vs bloom, same index.

profile: per (n_probe, sweep, bs, k), device us per call of the filter-payload kernel, the
scorer and the whole search, exact vs bloom (recheck off, as the timed bloom legs), from
torch.profiler over --reps pool batches. dump: ids + scores of the first --batches pool
batches, exact and bloom with recheck (C13), to --out for the bit-exact gate (compare.py).

  python clause_bench.py --dataset pubmed --dim 768 --n-lists 4096 --n-probe 24 512 \
      --sweeps all5 c0_mesh --bs 1 16 64 --k 100 --mode profile --json out.json
"""

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from bench import algos, inputs, run
from bench.config import load_dataset
from torch.profiler import ProfilerActivity, profile

import bytes_model


FILTER = (
    "exact_payloads",
    "bloom_payloads",
    "bloom_recheck",
    "process_documents",
    "column_info",
)


def kernel_group(name: str) -> str:
    if any(x in name for x in FILTER):
        return "filter"
    return "scorer" if "process_cluster" in name else "other"


def build(ds, inp, n_lists, k, seed, recheck):
    params = {"n_lists": n_lists, "n_probe": 1}
    t0 = time.perf_counter()
    exact = algos.build(
        "silvertorch", inp["item_embs"], k=k, backend="official-fork", filter_kind="clause",
        filter_mod=None, item_attrs=inp["item_attrs"],
        clause_is_reverse=inp["clause_is_reverse"], params=dict(params), seed=seed,
    )  # fmt: skip
    torch.cuda.synchronize()
    print(f"build exact {time.perf_counter() - t0:.1f}s", flush=True)
    blooms = {}
    for rc in recheck:
        # the same clustering: the exact module's index, the bloom's filter buffers on top
        b = copy.copy(exact)
        b._buffers = dict(exact._buffers)
        for name in ("item_clause_attrs", "clause_is_reverse"):
            b._buffers.pop(name, None)
        b.filter_mode = "bloom"
        b.m_bits, b.k_hash = (
            algos.BLOOM_DEFAULTS["m_bits"],
            algos.BLOOM_DEFAULTS["k_hash"],
        )
        b.official = algos.official_config(
            "silvertorch", "bloom", "official-fork", {"recheck": rc}
        )
        b._register_filter_buffers(
            inp["n_items"], inp["item_attrs"], None, exact.sort_perm
        )
        blooms[rc] = b
    torch.cuda.synchronize()
    return exact, blooms


def prof_one(fn, reps):
    fn(0)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(reps):
            fn(i)
        torch.cuda.synchronize()
    groups = defaultdict(float)
    names = defaultdict(float)
    for e in prof.key_averages():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        groups[kernel_group(e.key)] += e.self_device_time_total / reps
        names[e.key] += e.self_device_time_total / reps
    groups["total"] = sum(v for g, v in groups.items() if g != "total")
    top = sorted(names.items(), key=lambda kv: -kv[1])[:4]
    return dict(groups), top


def read_peak() -> float:
    """GB/s of a streaming read (sum of 2 GiB fp32): the practical DRAM ceiling."""
    x = torch.ones(1 << 29, dtype=torch.float32, device="cuda")
    x.sum()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(10):
        x.sum()
    end.record()
    torch.cuda.synchronize()
    return 10 * x.numel() * 4 / (start.elapsed_time(end) * 1e6)


def model(exact, sorted_attrs, inp, pool, qa_pool, batches) -> dict:
    """bytes_model per forward, averaged over the first ``batches`` pool batches."""
    tot = {"docs": 0.0, "exact_lane_B": 0.0, "exact_sector_B": 0.0, "exact_before_B": 0.0,
           "bloom_B": 0.0}  # fmt: skip
    rev = inp["clause_is_reverse"]
    rev = rev.tolist() if rev is not None else [False] * sorted_attrs.shape[1]
    for i in range(batches):
        qa = qa_pool[i].long()
        rows, doc, group, _ = bytes_model.probed_docs(exact, pool[i])
        lane, sector, before = bytes_model.exact_bytes(
            sorted_attrs, rev, qa, rows, doc, group
        )
        tot["docs"] += doc.numel()
        tot["exact_lane_B"] += lane
        tot["exact_sector_B"] += sector
        tot["exact_before_B"] += before
        tot["bloom_B"] += bytes_model.bloom_bytes(
            qa, rows, algos.BLOOM_DEFAULTS["k_hash"]
        )
    return {k: v / batches for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--n-lists", type=int, required=True)
    ap.add_argument("--n-probe", type=int, nargs="+", required=True)
    ap.add_argument("--sweeps", nargs="+", required=True)
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--k", type=int, nargs="+", default=[100])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config-dir", type=Path, default=Path("/scratch/fr/config"))
    ap.add_argument("--mode", choices=["profile", "dump"], required=True)
    ap.add_argument("--reps", type=int, default=20, help="profile: batches profiled")
    ap.add_argument("--json", type=Path, help="profile: the table rows")
    ap.add_argument(
        "--model-batches", type=int, default=4, help="profile: batches modelled"
    )
    ap.add_argument("--batches", type=int, default=8, help="dump: batches dumped")
    ap.add_argument("--out", type=Path, help="dump: the .pt file")
    a = ap.parse_args()

    dev = torch.device("cuda")
    ds = load_dataset(a.config_dir / f"{a.dataset}.yaml", a.dim)
    inp = inputs.load_inputs(ds, dev)
    k_max = max(a.k)
    recheck = [False] if a.mode == "profile" else [True]
    exact, blooms = build(ds, inp, a.n_lists, k_max, a.seed, recheck)
    mods = {
        "exact": exact,
        **{f"bloom{'_rc' if rc else ''}": m for rc, m in blooms.items()},
    }
    rows, dumps = [], {}
    sorted_attrs = (
        inp["item_attrs"].long()[exact.sort_perm].sort(dim=2, descending=True).values
    )
    with torch.inference_mode():
        if a.mode == "profile":
            rows.append({"key": "peak", "read_GBps": read_peak()})
        for sweep in a.sweeps:
            qa_s, skip = inputs.sweep_qa(inp["qa"], ds.clauses["clause"][sweep])
            for bs in a.bs:
                pool, qa_pool = inputs.query_pool(
                    inp, qa_s, skip, bs=bs, seed=a.seed, n_pool=64, device=dev
                )
                models = {}
                if a.mode == "profile":
                    for n_probe in a.n_probe:
                        exact.set_query_params(n_probe=n_probe)
                        models[n_probe] = model(exact, sorted_attrs, inp, pool, qa_pool,
                                                a.model_batches)  # fmt: skip
                for name, m in mods.items():
                    preps = [
                        run.filter_arg(m, qa_pool[i]) for i in range(pool.shape[0])
                    ]
                    for n_probe in a.n_probe:
                        m.set_query_params(n_probe=n_probe)
                        for k in a.k:
                            m.k = k

                            def fn(i, m=m, preps=preps):
                                return m(
                                    pool[i % pool.shape[0]], preps[i % pool.shape[0]]
                                )

                            key = f"{sweep}/bs{bs}/np{n_probe}/k{k}/{name}"
                            if a.mode == "profile":
                                g, top = prof_one(fn, a.reps)
                                rows.append(
                                    {"key": key, **g, **models[n_probe], "top": top}
                                )
                                print(
                                    key,
                                    {x: round(v, 1) for x, v in g.items()},
                                    flush=True,
                                )
                            else:
                                out = [
                                    fn(i) for i in range(min(a.batches, pool.shape[0]))
                                ]
                                dumps[key] = [(i.cpu(), s.cpu()) for i, s in out]
    if a.mode == "profile" and a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rows, indent=1))
    if a.mode == "dump":
        a.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dumps, a.out)
        print(f"dumped {len(dumps)} cells to {a.out}")


if __name__ == "__main__":
    main()
