"""ST-DLOOP's graph-capture gate in its launch-count form: SilverTorch (Triton) at D 768 under
``torch.compile(mode="reduce-overhead")``, warmed up, then 10 replays under ``torch.profiler``. Counts per
replay: ``cudaGraphLaunch`` calls, and kernel launches outside a graph (``cudaLaunchKernel`` /
``cuLaunchKernel(Ex)``). none / bloom / exact × bs {1, 16}; N 200,000 Gaussian items, n_lists 256, n_probe 24.

    PYTHONPATH=retrieve/src:retrieve python graph_launches.py out.json
"""

import json
import sys
from collections import Counter

import torch

from retrieve.modules.silvertorch import SilverTorch
from tests.conftest import make_attrs, make_query_attrs

D, N, REPLAYS = 768, 200_000, 10
dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)
items = torch.randn(N, D, device=dev, generator=g)
attrs = make_attrs(N, c=2, a_max=2)
out = {}
for mode in ("none", "bloom", "exact"):
    kw = {"m_bits": 1024, "k_hash": 5} if mode == "bloom" else {}
    m = SilverTorch(k=100, n_lists=256, n_probe=24, filter_mode=mode, seed=0, **kw)
    if mode == "none":
        m.register_index(items)
    else:
        m.register_index(items, item_clause_attrs=attrs)
    for bs in (1, 16):
        q = torch.randn(bs, D, device=dev, generator=g)
        qa = None if mode == "none" else make_query_attrs(bs, c=2)
        torch._dynamo.reset()
        c = torch.compile(m, mode="reduce-overhead")
        for _ in range(5):
            c(q, qa)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            for _ in range(REPLAYS):
                c(q, qa)
            torch.cuda.synchronize()
        names = Counter(e.name for e in prof.events())
        launches = sum(
            v
            for k, v in names.items()
            if k in ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx")
        )
        row = {"graph_launches_per_replay": names["cudaGraphLaunch"] / REPLAYS,
               "kernel_launches_outside_graph_per_replay": launches / REPLAYS}  # fmt: skip
        out[f"{mode}/bs{bs}"] = row
        print(mode, bs, row, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
