"""YFCC synth p001: are V1 and V2 (triton) bit-equal on scores and equal on ids up to ties (rule 3)? First 2,000 oracle rows, k 1000.
Run from evaluation/: python v1_v2_ties.py"""

import torch

from bench import inputs, run
from bench.config import load_matrix
from pathlib import Path

dev = torch.device("cuda")
jobs = load_matrix(
    Path("config/yfcc10m-synth.yaml"),
    Path("config/suites.yaml"),
    "synth",
    algos=["linr_v1_filter_mask", "linr_v2"],
    backends=["triton"],
    ks=[1000],
    sweeps=["p001"],
)
inp = inputs.load_inputs(jobs[0].data, dev)
assets = run.sweep_assets(jobs[0], inp, 1000, dev)
rows = assets["oracle_rows"].nonzero().reshape(-1)[:2000]
out = {}
for j in jobs:
    m = run.build_module(j, inp, assets, 1000, {})
    ids, sc = [], []
    with torch.inference_mode():
        for s in range(0, rows.numel(), 16):
            r = rows[s : s + 16]
            i, c = m(inp["queries"][r].to(dev), assets["qa_s"][r].to(dev))
            ids.append(i.cpu())
            sc.append(c.float().cpu())
    out[j.algo] = (torch.cat(ids), torch.cat(sc))
    del m
(i1, s1), (i2, s2) = out["linr_v1_filter_mask"], out["linr_v2"]
print(
    "scores torch.equal:",
    torch.equal(s1, s2),
    "max |diff|:",
    (s1 - s2).abs().max().item(),
)
print(
    "ids equal rows:",
    (i1 == i2).all(1).float().mean().item(),
    "id positions equal:",
    (i1 == i2).float().mean().item(),
)
# up to ties: per row, the set of ids at each distinct score must match, except the last (boundary) score group
bad = 0
for a, b, sa in zip(i1, i2, s1):
    last = sa[-1]
    inner = sa > last
    if set(a[inner].tolist()) != set(b[inner].tolist()):
        bad += 1
print("rows whose ids differ above the boundary score:", bad, "of", rows.numel())
print("rows with ties at the boundary:", int(((s1 == s1[:, -1:]).sum(1) > 1).sum()))
