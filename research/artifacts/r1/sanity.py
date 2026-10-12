"""R1 sanity on the E1c goodreads cell: the record's input identity and quality, the query and
item L2 norms out of the encode cache the cell wrote, and the scoring table's padding row.

    python sanity.py config/goodreads.yaml DIM RECORDS_JSONL
"""

import json
import sys
from pathlib import Path

import torch

from bench.config import load_dataset
from training.encode import ENCODE_CACHE, load_model_for_eval

ds = load_dataset(Path(sys.argv[1]), int(sys.argv[2]))
(rec,) = [json.loads(ln) for ln in Path(sys.argv[3]).read_text().splitlines()]
q = rec["quality"]
print(
    f"inputs {rec['inputs']}  status {rec['status']}  n_kept {rec['n_kept']}  "
    f"n_queries_oracle {rec['n_queries_oracle']}  n_queries_heldout {rec['n_queries_heldout']}"
)
for k in rec["ks"]:
    print(
        f"  @{k}: recall_oracle {q['oracle'][f'recall@{k}']:.6f}  "
        f"heldout recall {q['heldout'][f'recall@{k}']:.6f}  ndcg {q['heldout'][f'ndcg@{k}']:.6f}"
    )
blob = torch.load(
    ds.checkpoint.parent / ENCODE_CACHE, map_location="cpu", weights_only=True
)
for name in ("queries", "item_embs"):
    n = blob[name].float().norm(dim=1)
    print(
        f"{name} {tuple(blob[name].shape)} {blob[name].dtype}: norm min {n.min():.6f} max "
        f"{n.max():.6f}; outside 1 +- 1e-3: {int(((n - 1).abs() > 1e-3).sum())}"
    )
num_items = len(json.loads((ds.data_dir / "item_id_map.json").read_text()))
model = load_model_for_eval(ds.checkpoint, num_items, torch.device("cpu"))
print(
    f"scoring table row 0 (padding, dropped) norm {model.scoring_table()[0].norm():.6f}"
)
