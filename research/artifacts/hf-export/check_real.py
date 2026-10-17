"""Real-filter inputs in the archive against the records: per query, the exact pass count computed
on CPU from item_attrs_narrow.pt + clause_is_reverse_narrow.pt + eval_split.parquet (the library's
clause_subset_match semantics) equals the pass_count the run stored in its per-query sidecar.
Single-clause sweeps: every query; multi-clause sweeps: a fixed sample of SAMPLE queries.

Usage: check_real.py DATASET   (goodreads | arxiv | pubmed)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/wt/export-inputs/evaluation")
import numpy as np
import torch
import yaml

from eval_datasets import layout

torch.set_num_threads(32)
NAME = sys.argv[1]
A = Path("/data/anon-export")
DIR = {"goodreads": "eval-goodreads-work-id", "arxiv": "eval-arxiv-papers", "pubmed": "eval-pubmed"}[NAME]
CFG = yaml.safe_load(Path(f"/scratch/wt/export-inputs/evaluation/config/{NAME}.yaml").read_text())
SAMPLE = 300
d = A / DIR
n_items = {"goodreads": 797084, "arxiv": 2988996, "pubmed": 10_000_000}[NAME]
attrs, rev = layout.load_item_attrs(d / "item_attrs_narrow.pt", d / "clause_is_reverse_narrow.pt", n_items, torch.device("cpu"))
import polars as pl  # noqa: E402

qa = torch.tensor(pl.read_parquet(d / "eval_split.parquet")["query_attrs_narrow"].to_list(), dtype=torch.long)
sweeps = {**CFG["filters"]["clause"], **CFG["filters"].get("bloom", {})}


def counts_single(c: int, qv: torch.Tensor) -> torch.Tensor:
    a = attrs[:, c, :]
    vmax = int(max(a.max().item(), qv.max().item())) + 2
    # count each value once per item
    s, _ = a.sort(dim=1)
    first = torch.ones_like(s, dtype=torch.bool)
    first[:, 1:] = s[:, 1:] != s[:, :-1]
    vals = s[first & (s >= 0)]
    bc = torch.bincount(vals, minlength=vmax)
    out = bc[qv.clamp(min=0)]
    return n_items - out if bool(rev[c]) else out


def count_multi(q: torch.Tensor, clauses) -> int:
    ok = torch.ones(n_items, dtype=torch.bool)
    for c in clauses:
        if q[c] == -1:
            continue
        m = (attrs[:, c, :] == q[c]).any(dim=1)
        ok &= ~m if bool(rev[c]) else m
    return int(ok.sum())


R = A / "eval-results/runs"
done, n_rec, bad = set(), 0, 0
for p in sorted(R.rglob(f"{NAME}-d*.jsonl")):
    for line in p.open():
        r = json.loads(line)
        if r.get("dataset") != NAME or r.get("status") != "ok" or not r.get("per_query"):
            continue
        if r["filter_kind"] == "none" or (r["sweep"], r["filter_kind"]) in done:
            continue
        z = np.load(p.parent.parent / r["per_query"])
        rows, got = torch.from_numpy(z["rows"]).long(), z["pass_count"]
        clauses = sweeps[r["sweep"]]
        if len(clauses) == 1:
            want = counts_single(clauses[0], qa[rows, clauses[0]]).numpy()
            idx = np.arange(len(rows))
        else:
            idx = np.linspace(0, len(rows) - 1, min(SAMPLE, len(rows))).astype(int)
            want = np.array([count_multi(qa[rows[i]], clauses) for i in idx])
        diff = int((want != got[idx]).sum())
        done.add((r["sweep"], r["filter_kind"]))
        n_rec += 1
        bad += diff > 0
        print(f"{NAME} {r['sweep']:22s} {r['filter_kind']:6s} clauses {clauses}: {len(idx)} queries, {diff} differ", flush=True)
print(f"{NAME}: {n_rec} (sweep, filter_kind) compared, {bad} with differences")
