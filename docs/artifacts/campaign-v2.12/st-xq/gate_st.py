"""V1-TOPK SilverTorch check (ST-TOPK's gate): our SilverTorch triton forward at a before package (`make_pkg.sh`, imported as `rv_before`)
against this tree in one process, each arm building the same index (same seed). Per sweep x n_probe x bs: ids +
scores `torch.equal` on 16 pool batches, then ABAB windows of CALLS calls, eager or (`graph`) replays of one captured
CUDA graph a pool batch; ratio = after / before of the medians. `swap` builds and times the before arm first (the
first-built 10 M index ran up to 14 % faster with identical code, ST-WIDE); `aa` puts the before code in the after
slot (the identical-code floor); `slice=D` keeps the first D embedding dims (a kernel keep rule at a width with no
staged data, e.g. 192 from arXiv's 256).

    cd evaluation && PYTHONPATH=.:../retrieve/src:PKGS python gate.py DATASET DIM KIND SWEEPS N_LISTS MODE NPROBES BSS OUT.json \
        [graph] [swap] [aa] [slice=D] [k=K] [sparse=B]
      e.g. arxiv-synth 256 bloom p001,p01,p1 2048 bloom 24,128,256 1,16,64 out.json graph
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
flags = {
    a
    for a in sys.argv[1:]
    if a in ("graph", "swap", "aa") or a.startswith(("slice=", "k=", "sparse="))
}
GRAPH, SWAP, AA = "graph" in flags, "swap" in flags, "aa" in flags
SLICE = next((int(a[6:]) for a in flags if a.startswith("slice=")), None)
K = next((int(a[2:]) for a in flags if a.startswith("k=")), 100)
# sparse=B: the after arm's SPARSE_PASS_BOUND (2.0: every batch takes the two-pass with the table).
SPARSE = next((float(a[7:]) for a in flags if a.startswith("sparse=")), None)
if SPARSE is not None:
    import retrieve.modules.silvertorch as after_silvertorch

    after_silvertorch.SPARSE_PASS_BOUND = SPARSE
ds, dim, kind, sweeps, n_lists, mode, nprobes, bss, out = [
    a for a in sys.argv[1:] if a not in flags
]
cfg_path = Path(f"config/{ds}.yaml")
inp = inputs.load_inputs(
    config.load_dataset(cfg_path, int(dim)), DEV, with_filters=True
)
if SLICE:
    inp["item_embs"] = inp["item_embs"][:, :SLICE].contiguous()
    inp["queries"] = inp["queries"][:, :SLICE].contiguous()
sweep_cfg = yaml.safe_load(cfg_path.read_text())["filters"][kind]
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
after_cls = before_mod.SilverTorch if AA else SilverTorch
order = ("before", "after") if SWAP else ("after", "before")
arms = {a: (after_cls if a == "after" else before_mod.SilverTorch)(**kw) for a in order}
for m in arms.values():
    m.register_index(
        inp["item_embs"].to(DEV),
        *(() if mode == "none" else (inp["item_attrs"].to(DEV),)),
    )
del inp["item_embs"]
torch.cuda.empty_cache()


def same_up_to_ties(x, y):
    """With the scores equal: ids equal row by row above the row's k-th score; inside the tied
    group at the k-th score, distinct ids (any of the tied items), or ``-1`` if it is ``-inf``."""
    (ia, sa), (ib, _) = x, y
    for r in range(ia.shape[0]):
        keep = sa[r] > sa[r, -1]
        if not torch.equal(ia[r][keep], ib[r][keep]):
            return False
        tied = ~keep
        for ids in (ia[r][tied], ib[r][tied]):
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
        qa_s, skip = (
            (None, None)
            if mode == "none"
            else inputs.sweep_qa(inp["qa"], tuple(sweep_cfg[sweep]))
        )
        for n_probe in map(int, nprobes.split(",")):
            for m in arms.values():
                m.set_query_params(n_probe=n_probe)
            for bs in map(int, bss.split(",")):
                pool, qa_pool = inputs.query_pool(
                    inp, qa_s, skip, bs=bs, seed=0, n_pool=N_POOL, device=DEV
                )
                prep = {
                    a: [
                        None if mode == "none" else m.prepare_queries(qa_pool[i])
                        for i in range(N_POOL)
                    ]
                    for a, m in arms.items()
                }
                sparse = (
                    sum(getattr(p, "sparse", False) for p in prep["after"]) / N_POOL
                )
                outs = {
                    a: [m(pool[i], prep[a][i]) for i in range(N_POOL)]
                    for a, m in arms.items()
                }
                pairs = list(zip(outs["before"], outs["after"], strict=True))
                equal = all(
                    torch.equal(x[0], y[0]) and torch.equal(x[1], y[1])
                    for x, y in pairs
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
                    a: capture(m, pool, prep[a]) if GRAPH else m
                    for a, m in arms.items()
                }
                t = {a: [] for a in arms}
                for _ in range(ROUNDS):
                    for a in arms:
                        t[a].append(window(run[a], pool, prep[a]))
                del run
                med = {a: statistics.median(v) for a, v in t.items()}
                row = {"dataset": ds, "dim": SLICE or int(dim), "mode": mode, "sweep": sweep, "graph": GRAPH,
                       "swap": SWAP, "aa": AA, "k": K, "n_probe": n_probe, "bs": bs, "width": arms["after"]._probe_width,
                       "equal": equal, "scores_equal": scores_equal, "ids_up_to_ties": ids_up_to_ties, "sparse": sparse, "ms": med, "windows_ms": t, "ratio": med["after"] / med["before"]}  # fmt: skip
                rows.append(row)
                print(f"{ds} d{SLICE or dim} {mode} {sweep}{' graph' if GRAPH else ''}{' swap' if SWAP else ''}"
                      f"{' aa' if AA else ''} k{K} np{n_probe} bs{bs} sparse={sparse:.2f} eq={equal} scores_eq={scores_equal} ties_ok={ids_up_to_ties} before {med['before']:.3f} after "
                      f"{med['after']:.3f} ratio {row['ratio']:.3f}", flush=True)  # fmt: skip
Path(out).write_text(json.dumps(rows, indent=1))
