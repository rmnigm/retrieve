"""ST-WIDE gate: our SilverTorch triton forward at staging 1d9abd15 (`make_pkg.sh` package `rv_before`) against
this tree, one process, each arm
building the same index (same seed): per n_probe {24, 256,
1024} x bs {1, 16}, the outputs `torch.equal` (ids and scores) on 16 pool batches, then ABAB windows of CALLS
eager calls (or, with `graph`, replays of one captured CUDA graph a pool batch); ratio = after / before of the
medians.

    cd evaluation && PYTHONPATH=.:../retrieve/src:PKGS python gate.py DATASET DIM KIND SWEEP N_LISTS MODE OUT.json [graph]
      e.g. pubmed 768 bloom c0_mesh 4096 bloom out.json
"""

import json
import statistics
import sys
import time
from pathlib import Path

import rv_before.modules.silvertorch as before_mod
import torch
import yaml
from bench import config, inputs

from retrieve.modules.silvertorch import SilverTorch

DEV = torch.device("cuda")
CALLS, ROUNDS, N_POOL = 30, 8, 16
GRAPH = "graph" in sys.argv
ds, dim, kind, sweep, n_lists, mode, out = [a for a in sys.argv[1:] if a != "graph"]
cfg_path = Path(f"config/{ds}.yaml")
inp = inputs.load_inputs(
    config.load_dataset(cfg_path, int(dim)), DEV, with_filters=True
)
qa_s, skip = inputs.sweep_qa(
    inp["qa"], tuple(yaml.safe_load(cfg_path.read_text())["filters"][kind][sweep])
)
kw = {
    "k": 100,
    "n_lists": int(n_lists),
    "n_probe": 24,
    "filter_mode": mode,
    "n_iter": 10,
    "seed": 0,
}
if mode == "bloom":
    kw |= {"m_bits": 1024, "k_hash": 5}
arms = {"after": SilverTorch(**kw), "before": before_mod.SilverTorch(**kw)}
for m in arms.values():
    m.register_index(
        inp["item_embs"].to(DEV), item_clause_attrs=inp["item_attrs"].to(DEV)
    )
del inp["item_embs"]
torch.cuda.empty_cache()


def capture(m, pool, prep):
    """One CUDA graph a pool batch (the forward is capture-safe: no host sync)."""
    graphs = []
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(N_POOL):
            m(pool[i], prep[i])
    torch.cuda.current_stream().wait_stream(s)
    for i in range(N_POOL):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            m(pool[i], prep[i])
        graphs.append(g)
    return graphs


def window(m, pool, prep):
    if GRAPH:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(CALLS):
            m[i % N_POOL].replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / CALLS * 1e3
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(CALLS):
        m(pool[i % N_POOL], prep[i % N_POOL])
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / CALLS * 1e3


rows = []
with torch.inference_mode():
    for n_probe in (24, 256, 1024):
        for m in arms.values():
            m.set_query_params(n_probe=n_probe)
        for bs in (1, 16):
            pool, qa_pool = inputs.query_pool(
                inp, qa_s, skip, bs=bs, seed=0, n_pool=N_POOL, device=DEV
            )
            prep = {
                a: [m.prepare_queries(qa_pool[i]) for i in range(N_POOL)]
                for a, m in arms.items()
            }
            outs = {
                a: [m(pool[i], prep[a][i]) for i in range(N_POOL)]
                for a, m in arms.items()
            }
            equal = all(
                torch.equal(x[0], y[0]) and torch.equal(x[1], y[1])
                for x, y in zip(outs["before"], outs["after"], strict=True)
            )
            for a, m in arms.items():
                for i in range(10):
                    m(pool[i % N_POOL], prep[a][i % N_POOL])
            run = {
                a: capture(m, pool, prep[a]) if GRAPH else m for a, m in arms.items()
            }
            t = {"before": [], "after": []}
            for _ in range(ROUNDS):
                for a in arms:
                    t[a].append(window(run[a], pool, prep[a]))
            med = {a: statistics.median(v) for a, v in t.items()}
            row = {"dataset": ds, "dim": int(dim), "mode": mode, "graph": GRAPH, "n_probe": n_probe, "bs": bs,
                   "width": arms["after"]._probe_width, "equal": equal, "ms": med, "windows_ms": t,
                   "ratio": med["after"] / med["before"]}  # fmt: skip
            rows.append(row)
            print(f"{ds} d{dim} {mode}{' graph' if GRAPH else ''} np{n_probe} bs{bs} eq={equal} before {med['before']:.3f} after "
                  f"{med['after']:.3f} ratio {row['ratio']:.3f}", flush=True)  # fmt: skip
Path(out).write_text(json.dumps(rows, indent=1))
