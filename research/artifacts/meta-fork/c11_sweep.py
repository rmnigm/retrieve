"""META-FORK C11: the fork's scorer variant per (D, bs, path). One process per variant (STFORK_QUERY_SHARED is
read once); per cell the device time of one forward (20 profiled forwards, mean) in total and in the scorer kernels
(`process_cluster*`). Prints one JSON line per cell.

    cd evaluation && STFORK_QUERY_SHARED=0|1 PYTHONPATH=... python c11_sweep.py DATASET DIM N_LISTS NPROBES BSS \
        MODE SWEEP OUT.json
"""

import collections
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from bench import config, inputs
from torch.profiler import ProfilerActivity, profile

from retrieve import OfficialConfig, SilverTorch

DEV = torch.device("cuda")
ds_name, dim, n_lists, nprobes, bss, mode, sweep, out = sys.argv[1:9]
cfg_path = Path(f"config/{ds_name}.yaml")
inp = inputs.load_inputs(config.load_dataset(cfg_path, int(dim)), DEV, with_filters=True)
filt = yaml.safe_load(cfg_path.read_text())["filters"]
m = SilverTorch(k=100, n_lists=int(n_lists), n_probe=24, n_iter=10, seed=0, filter_mode=mode,
                backend="official-fork", official=OfficialConfig(score_path="int32"))  # fmt: skip
m.register_index(inp["item_embs"], *(() if mode == "none" else (inp["item_attrs"].to(DEV),)))
qa_s, skip = (None, None) if mode == "none" else inputs.sweep_qa(inp["qa"], tuple(filt["clause"][sweep]))
rows = []
for n_probe in map(int, nprobes.split(",")):
    m.set_query_params(n_probe=n_probe)
    for bs in map(int, bss.split(",")):
        pool, qa = inputs.query_pool(inp, qa_s, skip, bs=bs, seed=0, n_pool=4, device=DEV)
        p = None if mode == "none" else m.prepare_queries(qa[0])
        with torch.inference_mode():
            for _ in range(5):
                m(pool[0], p)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(20):
                    m(pool[0], p)
                torch.cuda.synchronize()
        t = collections.Counter()
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                t["total"] += e.device_time / 20
                if "process_cluster" in e.name:
                    t["scorer"] += e.device_time / 20
        row = {"dataset": ds_name, "dim": int(dim), "mode": mode, "n_probe": n_probe, "bs": bs,
               "query_shared": os.environ.get("STFORK_QUERY_SHARED"), "total_us": round(t["total"], 1),
               "scorer_us": round(t["scorer"], 1)}  # fmt: skip
        rows.append(row)
        print(json.dumps(row), flush=True)
Path(out).write_text(json.dumps(rows, indent=1))
