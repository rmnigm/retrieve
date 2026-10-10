"""ST-XQ, measured first: how much do a batch's rows share probed clusters? Per n_probe x bs on a dataset's SilverTorch
index (n_lists as given): over 16 pool batches, the mean of B * n_probe (row-cluster pairs), the distinct clusters, the
pairs per distinct cluster, and the item rows read row-major (sum of probed cluster sizes) against cluster-major (sum
over distinct clusters), i.e. the code bytes a cluster-major schedule would save.

    cd evaluation && PYTHONPATH=.:../retrieve/src python sharing.py DATASET DIM N_LISTS NPROBES BSS OUT.json
"""

import json
import statistics
import sys
from pathlib import Path

import torch
from bench import config, inputs

from retrieve.modules.silvertorch import SilverTorch

DEV = torch.device("cuda")
ds, dim, n_lists, nprobes, bss, out = sys.argv[1:7]
inp = inputs.load_inputs(
    config.load_dataset(Path(f"config/{ds}.yaml"), int(dim)),
    torch.device("cpu"),
    with_filters=False,
)
m = SilverTorch(k=100, n_lists=int(n_lists), n_probe=24, n_iter=10, seed=0)
m.register_index(inp.pop("item_embs").to(DEV))
sizes = m.cluster_sizes.long()
rows = []
with torch.inference_mode():
    for n_probe in map(int, nprobes.split(",")):
        m.set_query_params(n_probe=n_probe)
        for bs in map(int, bss.split(",")):
            pool, _ = inputs.query_pool(
                inp, None, None, bs=bs, seed=0, n_pool=16, device=DEV
            )
            stats = []
            for i in range(16):
                p = m._phase1_probe_ids(pool[i])
                distinct = torch.unique(p)
                stats.append(
                    (
                        p.numel(),
                        distinct.numel(),
                        sizes[p].sum().item(),
                        sizes[distinct].sum().item(),
                    )
                )
            pairs, dist, rm, cm = (
                statistics.mean(x[j] for x in stats) for j in range(4)
            )
            row = {"dataset": ds, "dim": int(dim), "n_lists": int(n_lists), "n_probe": n_probe, "bs": bs,
                   "pairs": pairs, "distinct": dist, "share": pairs / dist, "items_row_major": rm,
                   "items_cluster_major": cm, "read_ratio": cm / rm}  # fmt: skip
            rows.append(row)
            print(
                json.dumps(
                    {
                        k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in row.items()
                    }
                ),
                flush=True,
            )
Path(out).write_text(json.dumps(rows, indent=1))
