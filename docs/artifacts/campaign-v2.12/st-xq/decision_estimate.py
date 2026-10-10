"""ST-XQ decision estimate: per row, the estimated share of tiles holding a passing item over the whole index (no
probe ids: prepare_queries runs before phase 1), from per-cluster bit counts; multi-clause rows take the product over
clauses of each clause's rarest-bit share. Printed per sweep: median / max over rows, beside the measured two-pass
ratio for the cells it should predict. Derived from the tighter-skip measurement: a per-(cluster, bloom bit) item count gives each (row, probed cluster) an exact
upper bound on its passing items, min over the row's queried bits; 0 means no item of the cluster can pass. Per sweep x
n_probe at bs 16: the share of (row, cluster) pairs and of probed items with bound 0 (skippable without a bloom or code
load), and of items in clusters whose true pass count is 0 (the ceiling), plus the true bloom pass rate.

    cd evaluation && PYTHONPATH=.:../retrieve/src python cluster_bound.py DATASET DIM SWEEPS N_LISTS NPROBES OUT.json
"""

import json
import sys
from pathlib import Path

import torch
import yaml
from bench import config, inputs

from retrieve.modules.silvertorch import SilverTorch

DEV = torch.device("cuda")
ds, dim, sweeps, n_lists, out = sys.argv[1:6]
cfg_path = Path(f"config/{ds}.yaml")
inp = inputs.load_inputs(
    config.load_dataset(cfg_path, int(dim)), torch.device("cpu"), with_filters=True
)
sweep_cfg = yaml.safe_load(cfg_path.read_text())["filters"]["bloom"]
m = SilverTorch(
    k=100,
    n_lists=int(n_lists),
    n_probe=24,
    filter_mode="bloom",
    m_bits=1024,
    k_hash=5,
    n_iter=10,
    seed=0,
)
m.register_index(
    inp.pop("item_embs").to(DEV), item_clause_attrs=inp["item_attrs"].to(DEV)
)
n = m.item_codes.shape[0]
offsets = m.cluster_offsets.long()
cluster_of = torch.repeat_interleave(
    torch.arange(len(offsets) - 1, device=DEV), offsets.diff()
)
m_bits = m.bloom_transposed.shape[0]
words = (n + 63) // 64
# counts[c, bit]: items of cluster c (cluster-sorted order) with that bloom bit set.
counts = torch.zeros(len(offsets) - 1, m_bits, dtype=torch.int32, device=DEV)
for bit in range(m_bits):
    w = m.bloom_transposed[bit, :words]
    bits = ((w.unsqueeze(1) >> torch.arange(64, device=DEV)) & 1).reshape(-1)[:n].int()
    counts[:, bit] = torch.zeros(
        len(offsets) - 1, dtype=torch.int32, device=DEV
    ).index_add_(0, cluster_of, bits)
sizes = offsets.diff()
tiles = (sizes + 255) // 256
K_HASH = 5
rows = []
with torch.inference_mode():
    for sweep in sweeps.split(","):
        qa_s, skip = inputs.sweep_qa(inp["qa"], tuple(sweep_cfg[sweep]))
        pool, qa_pool = inputs.query_pool(
            inp, qa_s, skip, bs=16, seed=0, n_pool=4, device=DEV
        )
        fr = []
        for i in range(4):
            bits = m.prepare_queries(
                qa_pool[i]
            ).query_bits  # [B, C * K_HASH], clause-major
            b, nq = bits.shape
            cb = bits.view(b, nq // K_HASH, K_HASH)
            active = (cb >= 0).all(2)  # [B, C]
            # rarest bit of each clause, per cluster: [B, C, n_lists]
            c_min = counts.t()[cb.clamp_min(0)].amin(2).double()
            share = torch.where(
                active[:, :, None], c_min / sizes.clamp_min(1).double(), 1.0
            ).prod(1)  # [B, n_lists]
            est = share * sizes.double()  # estimated passing items per cluster
            frac = torch.minimum(tiles.double(), est).sum(1) / tiles.sum().double()
            fr.append(frac)
        fr = torch.cat(fr)
        row = {
            "dataset": ds,
            "sweep": sweep,
            "frac_median": fr.median().item(),
            "frac_max": fr.max().item(),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in row.items()
                }
            ),
            flush=True,
        )
Path(out).write_text(json.dumps(rows, indent=1))
