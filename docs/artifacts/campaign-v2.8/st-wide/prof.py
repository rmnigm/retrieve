"""ST-WIDE, measured first: our SilverTorch triton bloom forward on PubMed (10 M, d768, `c0_mesh`, n_lists 4096,
m_bits 1024, k_hash 5; C7's cell) per n_probe {24, 256, 1024} x bs {1, 16}, k 100: the forward's wall ms (8 synced
windows, median), the top kernels' device us per call (torch.profiler, 20 calls), the probe width and the scorer tile
`tile_for_width` picks.

    cd evaluation && PYTHONPATH=.:../retrieve/src python prof.py out.json
"""

import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml
from bench import config, inputs
from torch.profiler import ProfilerActivity, profile

from retrieve.modules.silvertorch import SilverTorch

cps = importlib.import_module("retrieve.ops.triton.codesigned_probe_score")
host = importlib.import_module("retrieve.ops.triton._host")
DEV = torch.device("cuda")
cfg_path = Path("config/pubmed.yaml")
inp = inputs.load_inputs(config.load_dataset(cfg_path, 768), DEV, with_filters=True)
qa_s, skip = inputs.sweep_qa(
    inp["qa"],
    tuple(yaml.safe_load(cfg_path.read_text())["filters"]["bloom"]["c0_mesh"]),
)
m = SilverTorch(
    k=100,
    n_lists=4096,
    n_probe=24,
    filter_mode="bloom",
    m_bits=1024,
    k_hash=5,
    n_iter=10,
    seed=0,
)
t0 = time.perf_counter()
m.register_index(inp["item_embs"].to(DEV), item_clause_attrs=inp["item_attrs"].to(DEV))
print(f"build {time.perf_counter() - t0:.0f} s", flush=True)
rows = []
with torch.inference_mode():
    for n_probe in (24, 256, 1024):
        m.set_query_params(n_probe=n_probe)
        for bs in (1, 16):
            pool, qa_pool = inputs.query_pool(
                inp, qa_s, skip, bs=bs, seed=0, n_pool=16, device=DEV
            )
            prep = [m.prepare_queries(qa_pool[i]) for i in range(16)]
            for i in range(10):
                m(pool[i % 16], prep[i % 16])
            w = []
            for _ in range(8):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for i in range(30):
                    m(pool[i % 16], prep[i % 16])
                torch.cuda.synchronize()
                w.append((time.perf_counter() - t0) / 30 * 1e3)
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for i in range(20):
                    m(pool[i % 16], prep[i % 16])
                torch.cuda.synchronize()
            ev = sorted((e for e in p.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA),
                        key=lambda e: -e.self_device_time_total)  # fmt: skip
            tile = host.tile_for_width(cps.CONFIGS, 768, bs, m._probe_width)
            row = {"n_probe": n_probe, "bs": bs, "width": m._probe_width, "wall_ms": statistics.median(w),
                   "tile": [tile.block_p, tile.num_warps, tile.block_d, tile.skip],
                   "kernels_us": [(e.key[:80], e.self_device_time_total / 20, e.count / 20) for e in ev[:6]]}  # fmt: skip
            rows.append(row)
            print(json.dumps(row), flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
