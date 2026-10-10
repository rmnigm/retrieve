"""ST-XQ tighter skip, measured first: a per-(cluster, bloom bit) item count gives each (row, probed cluster) an exact
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
ds, dim, sweeps, n_lists, nprobes, out = sys.argv[1:7]
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
rows = []
with torch.inference_mode():
    for sweep in sweeps.split(","):
        qa_s, skip = inputs.sweep_qa(inp["qa"], tuple(sweep_cfg[sweep]))
        pool, qa_pool = inputs.query_pool(
            inp, qa_s, skip, bs=16, seed=0, n_pool=4, device=DEV
        )
        for n_probe in map(int, nprobes.split(",")):
            m.set_query_params(n_probe=n_probe)
            agg = torch.zeros(6, dtype=torch.float64)
            for i in range(4):
                bits = m.prepare_queries(qa_pool[i]).query_bits  # [B, nq] (-1 inactive)
                probes = m._phase1_probe_ids(pool[i])  # [B, n_probe]
                c = counts[probes]  # [B, n_probe, m_bits]
                q = bits.clamp_min(0).unsqueeze(1).expand(-1, n_probe, -1)
                bound = torch.where(
                    bits.unsqueeze(1) >= 0, c.gather(2, q), torch.iinfo(torch.int32).max
                ).amin(2)
                # true per-cluster pass counts: AND of the queried bit rows over the cluster's items
                passing = torch.full(
                    (bits.shape[0], words), -1, dtype=torch.int64, device=DEV
                )
                for j in range(bits.shape[1]):
                    mj = bits[:, j]
                    passing &= torch.where(
                        (mj >= 0)[:, None],
                        m.bloom_transposed[mj.clamp_min(0), :words],
                        -1,
                    )
                item_pass = (
                    (passing.unsqueeze(2) >> torch.arange(64, device=DEV)) & 1
                ).reshape(bits.shape[0], -1)[:, :n]
                per_cluster = torch.zeros(
                    bits.shape[0], len(offsets) - 1, dtype=torch.int64, device=DEV
                )
                per_cluster.index_add_(1, cluster_of, item_pass.long())
                true_c = per_cluster.gather(1, probes)
                sz = sizes[probes].double()
                agg += torch.tensor([bound.numel(), (bound == 0).sum().item(), sz.sum().item(),
                                     sz[bound == 0].sum().item(), sz[true_c == 0].sum().item(),
                                     item_pass.double().mean().item()], dtype=torch.float64)  # fmt: skip
            row = {"dataset": ds, "sweep": sweep, "n_lists": int(n_lists), "n_probe": n_probe,
                   "pairs_bound0": agg[1].item() / agg[0].item(), "items_bound0": agg[3].item() / agg[2].item(),
                   "items_true0": agg[4].item() / agg[2].item(), "pass_rate": agg[5].item() / 4}  # fmt: skip
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
