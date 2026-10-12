"""V1-BS1, measured first: LiNR V1 (triton) forward per cell, device µs by kernel family (torch.profiler, CALLS calls,
µs per call) plus the eager wall ms (C3's compiled arm is d-run's / v-pod1-run's).
Families: gemm (cuBLAS / cutlass / triton mm), mask (our fused clause / bloom score mask), topk (radix / sort / gather),
other.

    cd evaluation && PYTHONPATH=.:../retrieve/src python prof.py DATASET DIM KIND SWEEP MODE BSS OUT.json
"""

import json
import re
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml
from bench import config, inputs
from torch.profiler import ProfilerActivity, profile

from retrieve.modules.filters import BloomFilter, ExactAttributeFilter
from retrieve.modules.linr import LiNRV1

DEV = torch.device("cuda")
CALLS = 20
ds, dim, kind, sweep, mode, bss, out = sys.argv[1:8]
cfg_path = Path(f"config/{ds}.yaml")
inp = inputs.load_inputs(
    config.load_dataset(cfg_path, int(dim)), DEV, with_filters=True
)
filt = BloomFilter(m_bits=1024, k_hash=5) if mode == "bloom" else ExactAttributeFilter()
m = LiNRV1(k=100, filter=filt)
m.register_index(
    inp["item_embs"].to(DEV),
    inp["item_attrs"].to(DEV),
    inp["clause_is_reverse"].to(DEV),
)
qa_s, skip = inputs.sweep_qa(
    inp["qa"], tuple(yaml.safe_load(cfg_path.read_text())["filters"][kind][sweep])
)
FAMILIES = [
    ("topk", re.compile(r"topk|TopK|radix|Radix|Digit|WithinK|sort|Sort", re.I)),
    ("gemm", re.compile(r"gemm|gemv|cutlass|ampere|sm80|mm_|matmul|xmma|cublas", re.I)),
    ("mask", re.compile(r"clause|bloom|mask", re.I)),
]


def family(name):
    return next((f for f, rx in FAMILIES if rx.search(name)), "other")


def run_arm(fn, pool, prep):
    for i in range(10):
        fn(pool[i % 4], prep[i % 4])
    w = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(30):
            fn(pool[i % 4], prep[i % 4])
        torch.cuda.synchronize()
        w.append((time.perf_counter() - t0) / 30 * 1e3)
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for i in range(CALLS):
            fn(pool[i % 4], prep[i % 4])
        torch.cuda.synchronize()
    fam, kern = {}, []
    for e in p.key_averages():
        if e.device_type != torch.autograd.DeviceType.CUDA or e.key.startswith("##"):
            continue
        f = family(e.key)
        fam[f] = fam.get(f, 0.0) + e.self_device_time_total / CALLS
        kern.append(
            (e.key[:90], round(e.self_device_time_total / CALLS, 1), e.count / CALLS, f)
        )
    kern.sort(key=lambda x: -x[1])
    return statistics.median(w), fam, kern[:10]


rows = []
with torch.inference_mode():
    for bs in map(int, bss.split(",")):
        pool, qa_pool = inputs.query_pool(
            inp, qa_s, skip, bs=bs, seed=0, n_pool=4, device=DEV
        )
        prep = [m.prepare_queries(qa_pool[i]) for i in range(4)]
        for arm, fn in (("triton", m),):
            wall, fam, kern = run_arm(fn, pool, prep)
            row = {"dataset": ds, "dim": int(dim), "mode": mode, "sweep": sweep, "bs": bs, "arm": arm,
                   "wall_ms": wall, "family_us": fam, "kernels": kern}  # fmt: skip
            rows.append(row)
            print(f"{ds} d{dim} {mode} {sweep} bs{bs} {arm:8s} wall {wall:.3f} ms | "
                  + " ".join(f"{k} {v:.0f}" for k, v in sorted(fam.items(), key=lambda x: -x[1])), flush=True)  # fmt: skip
Path(out).write_text(json.dumps(rows, indent=1))
