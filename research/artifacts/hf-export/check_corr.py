"""arxiv-corr-synth: the per-query pass counts in the records' sidecars against the regenerated
cluster ids (pass count of query q at rate j = #items whose cluster equals q's centroid)."""
import json
from pathlib import Path

import numpy as np
import torch

A = Path("/data/anon-export/eval-arxiv-papers")
R = Path("/data/anon-export/eval-results/runs")
items = torch.load(A / "item_attrs_corr.pt", weights_only=True)[:, :, 0]
queries = torch.load(A / "query_attrs_corr.pt", weights_only=True)
sweeps = {"c001": 0, "c003": 1, "c01": 2}
want = {}
for name, j in sweeps.items():
    sizes = torch.bincount(items[:, j])
    want[name] = sizes[queries[:, j]].numpy()
n = bad = 0
for p in R.rglob("arxiv-corr-synth-d128.jsonl"):
    for line in p.open():
        r = json.loads(line)
        if r.get("status") != "ok" or not r.get("per_query"):
            continue
        z = np.load(p.parent.parent / r["per_query"])
        got, rows = z["pass_count"], z["rows"]
        w = want[r["sweep"]][rows]
        n += 1
        if not np.array_equal(got, w):
            bad += 1
            print("MISMATCH", p.parent.parent.name, r["sweep"], int((got != w).sum()), "of", len(rows))
print(f"{n} arxiv-corr-synth records compared query by query, {bad} mismatches")

# size of the difference on the unconverged rates
for p in R.rglob("arxiv-corr-synth-d128.jsonl"):
    seen = set()
    for line in p.open():
        r = json.loads(line)
        if r.get("status") != "ok" or not r.get("per_query") or r["sweep"] in seen:
            continue
        seen.add(r["sweep"])
        z = np.load(p.parent.parent / r["per_query"])
        got = z["pass_count"].astype(float)
        w = want[r["sweep"]][z["rows"]].astype(float)
        rel = np.abs(w - got) / got
        print(p.parent.parent.name, r["sweep"], "mean pass rate record / regenerated",
              f"{got.mean() / items.shape[0]:.6f} / {w.mean() / items.shape[0]:.6f}",
              f"per-query |diff|/count median {np.median(rel):.2e} max {rel.max():.2e}")
    break
