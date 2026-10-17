"""Regenerate the synthetic filter files (uniform draws 0-2, arXiv correlated draw 0) into the archive, CPU only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/wt/export-inputs/evaluation")
import torch

from eval_datasets import synth_filter

A = Path("/data/anon-export")
torch.set_num_threads(64)
which = sys.argv[1]
if which == "corr":
    meta = synth_filter.build_correlated(A / "eval-arxiv-papers", A / "eval-arxiv-papers/content_d128", draw=0)
    print(json.dumps({k: v for k, v in meta.items() if k != "per_rate"})[:2000])
else:
    for draw in (0, 1, 2):
        print(which, draw, json.dumps(synth_filter.build(A / which, draw=draw))[:600], flush=True)
