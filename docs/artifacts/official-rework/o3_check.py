"""OFFICIAL-REWORK M3: Meta's extension as shipped (pip build of 21aa35e: nvcc host code at gcc -O0) against the same
sources built with host + nvcc -O3 (/scratch/meta-o3/src, kernels unmodified), one build per process: both register the
`st` namespace, so the driver alternates processes (shipped, O3, shipped, O3, ...) and compares medians, with the
same-build rounds as the method floor. Per process: the official forward of every path (none / exact / bloom partial /
bloom full), fp16 score path, filter prepared outside, bs {1, 16}, k 100, on goodreads or arXiv; outputs saved for the
bit-exact check, and 8 windows of CALLS calls timed per path.

    cd evaluation && PYTHONPATH=[/scratch/meta-o3/src:].:../retrieve/src python o3_check.py {goodreads|arxiv} LABEL OUT_DIR
"""

import json
import statistics
import sys
import time
from pathlib import Path

import silvertorch
import torch
import yaml
from bench import config, inputs

from retrieve.modules.silvertorch import OfficialConfig, SilverTorch

DEV = torch.device("cuda")
SWEEP = {"goodreads": "c0_genre", "arxiv": "c0_maincat"}
CALLS, WINDOWS, N_POOL = 100, 8, 16
ds, label, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
cfg_path = Path(f"config/{ds}.yaml")
inp = inputs.load_inputs(config.load_dataset(cfg_path, 128), DEV, with_filters=True)
clauses = tuple(yaml.safe_load(cfg_path.read_text())["filters"]["clause"][SWEEP[ds]])
qa_s, skip = inputs.sweep_qa(inp["qa"], clauses)
item_embs, attrs = inp["item_embs"].to(DEV), inp["item_attrs"].to(DEV)
rows, saved = [], {}
with torch.inference_mode():
    for mode, path in (
        ("none", None),
        ("exact", None),
        ("bloom", "partial"),
        ("bloom", "full"),
    ):
        kw = {"k": 100, "n_lists": 1024, "n_probe": 32, "n_iter": 10, "seed": 0, "backend": "official",
              "official": OfficialConfig(score_path="fp16", **({"bloom_path": path} if path else {}))}  # fmt: skip
        if mode != "none":
            kw["filter_mode"] = mode
        if mode == "bloom":
            kw["k_hash"] = 5
        m = SilverTorch(**kw)
        m.register_index(item_embs, *(() if mode == "none" else (attrs,)))
        for bs in (16, 1):
            pool, qa_pool = inputs.query_pool(
                inp, qa_s, skip, bs=bs, seed=0, n_pool=N_POOL, device=DEV
            )
            prep = [
                None if mode == "none" else m.prepare_queries(qa_pool[i])
                for i in range(N_POOL)
            ]
            key = f"{mode}-{path}-bs{bs}"
            saved[key] = [tuple(t.cpu() for t in m(pool[i], prep[i])) for i in range(4)]
            for i in range(20):
                m(pool[i % N_POOL], prep[i % N_POOL])
            t = []
            for _ in range(WINDOWS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for i in range(CALLS):
                    m(pool[i % N_POOL], prep[i % N_POOL])
                torch.cuda.synchronize()
                t.append((time.perf_counter() - t0) / CALLS * 1e3)
            rows.append({"dataset": ds, "build": label, "mode": mode, "bloom_path": path, "bs": bs,
                         "median_ms": statistics.median(t), "windows_ms": t})  # fmt: skip
            print(f"{label} {ds} {key}: {statistics.median(t):.3f} ms", flush=True)
torch.save(saved, out / f"outputs_{ds}_{label}.pt")
(out / f"time_{ds}_{label}.json").write_text(
    json.dumps({"silvertorch": silvertorch.__file__, "rows": rows}, indent=1)
)
