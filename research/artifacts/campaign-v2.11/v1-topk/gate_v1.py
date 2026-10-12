"""V1-TOPK gate (V1-BS1's, with k=K): LiNR V1 (triton) at a before package (`make_pkg.sh`, imported as `rv_before`) against this tree, one
process, each arm building the same index. Per sweep x bs: ids + scores on 16 pool batches (strict, scores-equal, ids
equal up to ties), then ABAB windows of CALLS calls, eager or (`graph`) replays of one captured CUDA graph per pool
batch; ratio = after / before of the medians. `swap` builds and times the before arm first.

    cd evaluation && PYTHONPATH=.:../retrieve/src:PKGS python gate_v1.py DATASET DIM KIND SWEEPS MODE BSS OUT.json \
        [graph] [swap] [k=K]
      e.g. goodreads 128 clause c0_genre,all4 clause 1,16,64 out.json graph swap
"""

import json
import statistics
import sys
import time
from pathlib import Path

import rv_before.modules.filters as before_filters
import rv_before.modules.linr as before_linr
import torch
import yaml
from bench import config, inputs

import retrieve.modules.filters as after_filters
import retrieve.modules.linr as after_linr

DEV = torch.device("cuda")
CALLS, ROUNDS, N_POOL = 30, 8, 16
flags = {a for a in sys.argv[1:] if a in ("graph", "swap") or a.startswith("k=")}
K = next((int(a[2:]) for a in flags if a.startswith("k=")), 100)
GRAPH, SWAP = "graph" in flags, "swap" in flags
ds, dim, kind, sweeps, mode, bss, out = [a for a in sys.argv[1:] if a not in flags]
cfg_path = Path(f"config/{ds}.yaml")
# On the host, the items fp16 there: each arm moves a 2-byte table over when it builds, so the GPU
# never holds the fp32 one (10 M d768 is 30 GB fp32).
inp = inputs.load_inputs(
    config.load_dataset(cfg_path, int(dim)), torch.device("cpu"), with_filters=True
)
items16 = inp.pop("item_embs").half()
attrs = inp["item_attrs"].to(
    DEV
)  # shared by both arms (the filters keep a contiguous input)
sweep_cfg = yaml.safe_load(cfg_path.read_text())["filters"][kind]


def build(linr, filters):
    f = (
        filters.BloomFilter(m_bits=1024, k_hash=5)
        if mode == "bloom"
        else filters.ExactAttributeFilter()
    )
    m = linr.LiNRV1(k=K, filter=f)
    m.register_index(
        items16.to(DEV),
        attrs,
        inp["clause_is_reverse"].to(DEV) if mode == "clause" else None,
    )
    torch.cuda.empty_cache()
    return m


mods = {"after": (after_linr, after_filters), "before": (before_linr, before_filters)}
arms = {
    a: build(*mods[a]) for a in (("before", "after") if SWAP else ("after", "before"))
}
items16 = None  # free the host copy
torch.cuda.empty_cache()


def same_up_to_ties(x, y):
    """With the scores equal: ids equal row by row above the row's k-th score; inside the tied
    group, distinct ids, or ``-1`` where it is ``-inf``."""
    (ia, sa), (ib, _) = x, y
    for r in range(ia.shape[0]):
        keep = sa[r] > sa[r, -1]
        if not torch.equal(ia[r][keep], ib[r][keep]):
            return False
        for ids in (ia[r][~keep], ib[r][~keep]):
            if sa[r, -1] == float("-inf"):
                if not (ids == -1).all():
                    return False
            elif ids.unique().numel() != ids.numel():
                return False
    return True


def capture(m, pool, prep):
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


def window(run, pool, prep):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(CALLS):
        if GRAPH:
            run[i % N_POOL].replay()
        else:
            run(pool[i % N_POOL], prep[i % N_POOL])
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / CALLS * 1e3


rows = []
with torch.inference_mode():
    for sweep in sweeps.split(","):
        qa_s, skip = inputs.sweep_qa(inp["qa"], tuple(sweep_cfg[sweep]))
        for bs in map(int, bss.split(",")):
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
            pairs = list(zip(outs["before"], outs["after"], strict=True))
            equal = all(
                torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in pairs
            )
            scores_equal = all(torch.equal(x[1], y[1]) for x, y in pairs)
            ids_up_to_ties = scores_equal and all(
                same_up_to_ties(x, y) for x, y in pairs
            )
            del outs
            for a, m in arms.items():
                for i in range(10):
                    m(pool[i % N_POOL], prep[a][i % N_POOL])
            run = {
                a: capture(m, pool, prep[a]) if GRAPH else m for a, m in arms.items()
            }
            t = {a: [] for a in arms}
            for _ in range(ROUNDS):
                for a in arms:
                    t[a].append(window(run[a], pool, prep[a]))
            del run
            med = {a: statistics.median(v) for a, v in t.items()}
            row = {"dataset": ds, "dim": int(dim), "mode": mode, "sweep": sweep, "graph": GRAPH, "swap": SWAP,
                   "bs": bs, "k": K, "equal": equal, "scores_equal": scores_equal, "ids_up_to_ties": ids_up_to_ties,
                   "ms": med, "windows_ms": t, "ratio": med["after"] / med["before"]}  # fmt: skip
            rows.append(row)
            print(f"{ds} d{dim} {mode} {sweep}{' graph' if GRAPH else ''} bs{bs} eq={equal} scores_eq={scores_equal} "
                  f"ties_ok={ids_up_to_ties} before {med['before']:.3f} after {med['after']:.3f} "
                  f"ratio {row['ratio']:.3f}", flush=True)  # fmt: skip
Path(out).write_text(json.dumps(rows, indent=1))
