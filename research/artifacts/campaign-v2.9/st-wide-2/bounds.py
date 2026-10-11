"""ST-WIDE-2: per sweep, the rows' pass-rate bounds (the rarest queried bit's item share, `bloom_bit_freq`)
against their true bloom pass rate over all N items (every queried bit set): quantiles over 512 rows.

    cd evaluation && PYTHONPATH=.:../retrieve/src python bounds.py DATASET DIM SWEEPS N_LISTS OUT.json
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
    config.load_dataset(cfg_path, int(dim)), DEV, with_filters=True
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
m.register_index(inp["item_embs"].to(DEV), item_clause_attrs=inp["item_attrs"].to(DEV))
n = m.item_codes.shape[0]
q = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=DEV)
rows = []
with torch.inference_mode():
    for sweep in sweeps.split(","):
        qa_s, skip = inputs.sweep_qa(inp["qa"], tuple(sweep_cfg[sweep]))
        pool, qa_pool = inputs.query_pool(
            inp, qa_s, skip, bs=512, seed=0, n_pool=1, device=DEV
        )
        bits = m.prepare_queries(qa_pool[0]).query_bits
        bound = torch.where(bits >= 0, m.bloom_bit_freq[bits.clamp_min(0)], 1.0).amin(1)
        words = (n + 63) // 64
        passing = torch.ones(bits.shape[0], words, dtype=torch.int64, device=DEV)
        passing = passing * -1
        for j in range(bits.shape[1]):
            mj = bits[:, j]
            w = m.bloom_transposed[mj.clamp_min(0), :words]
            passing &= torch.where((mj >= 0)[:, None], w, torch.full_like(w, -1))
        bitcount = sum(((passing >> s) & 1).sum(1) for s in range(64)).float()
        rate = bitcount / n
        row = {
            "sweep": sweep,
            "bound_q": torch.quantile(bound, q).tolist(),
            "rate_q": torch.quantile(rate, q).tolist(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
Path(out).write_text(json.dumps(rows, indent=1))
