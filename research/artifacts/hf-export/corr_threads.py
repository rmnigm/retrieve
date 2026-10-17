"""build_correlated at a given torch thread count into a scratch dir; compare per-query pass counts with the sidecars."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/wt/export-inputs/evaluation")
import numpy as np
import torch

from eval_datasets import synth_filter

threads = int(sys.argv[1])
if threads:
    torch.set_num_threads(threads)
A = Path("/data/anon-export/eval-arxiv-papers")
T = Path(f"/data/work/corr-t{threads}")
T.mkdir(exist_ok=True)
for f in ("item_attrs_narrow.pt", "heldout.parquet", "eval_split.parquet", "content_d128"):
    if not (T / f).exists():
        os.symlink(A / f, T / f)
synth_filter.build_correlated(T, T / "content_d128", draw=0)
items = torch.load(T / "item_attrs_corr.pt", weights_only=True)[:, :, 0]
queries = torch.load(T / "query_attrs_corr.pt", weights_only=True)
R = Path("/data/anon-export/eval-results/runs/linr-corr-s0")
res = {}
for p in R.rglob("arxiv-corr-synth-d128.jsonl"):
    for line in p.open():
        r = json.loads(line)
        if r.get("per_query") and r["sweep"] not in res:
            j = {"c001": 0, "c003": 1, "c01": 2}[r["sweep"]]
            z = np.load(R / r["per_query"])
            w = torch.bincount(items[:, j])[queries[:, j]].numpy()[z["rows"]]
            res[r["sweep"]] = int((w != z["pass_count"]).sum())
print("threads", threads or torch.get_num_threads(), "mismatched queries per sweep", res, flush=True)
