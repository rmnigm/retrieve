"""ARXIV-N95: the achieved pass rate of each kept arXiv `filter` clause sweep (d128), to pick
the n95 suite's median sweep. The harness's own path (load_inputs, sweep_qa, the exact
clause filter, oracle.pass_counts / pass_rate) on CPU with the torch filter backend, so no
oracle blob is written. Run from evaluation/ with CUDA hidden.
"""

import json

from pathlib import Path

import torch

from bench import inputs, oracle
from bench.config import load_dataset

SWEEPS = ("c3_nversions", "c0_maincat", "all4")
ds = load_dataset(Path("config/arxiv.yaml"), 128)
inp = inputs.load_inputs(ds, torch.device("cpu"))
filt = inputs.build_filters("clause", inp, ["torch"], bloom={"m_bits": 1024, "k_hash": 5})["torch"]
out = {}
for sweep in SWEEPS:
    qa_s, skip = inputs.sweep_qa(inp["qa"], ds.clauses["clause"][sweep])
    counts = oracle.pass_counts(filt, qa_s, skip, inp["n_items"], device=torch.device("cpu"))
    out[sweep] = {"pass_rate": oracle.pass_rate(counts, inp["n_items"]),
                  "n_kept": int((counts >= 0).sum())}  # fmt: skip
    print(sweep, out[sweep], flush=True)
med = sorted(SWEEPS, key=lambda s: out[s]["pass_rate"])[1]
print(json.dumps({"n_items": inp["n_items"], "sweeps": out, "median": med}))
