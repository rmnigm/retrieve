"""Pubmed restage gate: is the restaged slice the one D1's pubmed records (code_version 72e5a90)
ran on? Recomputes, from the restaged item attrs, eval split and held-out targets alone (CPU, no
embeddings), every per-sweep fact a record carries — n_items, n_queries, n_kept,
n_queries_oracle, n_queries_heldout, n_targets_in_filter and the exact pass_rate — and compares
them with every record of the d1/pubmed leg. The exact pass counts come from counting each
item's value combinations over the sweep's active clauses (inclusion-exclusion over the reverse
clause), checked against the library's ``clause_subset_match`` on a full 10 M scan for a sample
of queries per sweep.

    python slice_identity.py EVALUATION_DIR RECORDS_JSONL [N_BRUTE]
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

evaluation = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(evaluation))
from bench.config import load_dataset  # noqa: E402
from bench.inputs import sweep_qa  # noqa: E402
from eval_datasets import layout  # noqa: E402
from retrieve.functional import clause_subset_match  # noqa: E402

ds = load_dataset(evaluation / "config" / "pubmed.yaml", 768)
data_dir = evaluation / ds.data_dir if not ds.data_dir.is_absolute() else ds.data_dir
n_brute = int(sys.argv[3]) if len(sys.argv) > 3 else 16

heldout = pl.read_parquet(data_dir / "heldout.parquet")
n_split = heldout.height
item_attrs, rev = layout.load_item_attrs(
    data_dir / ds.attrs.name,
    data_dir / ds.reverse.name,
    10_000_000,
    torch.device("cpu"),
)
N, C, A = item_attrs.shape
qa = layout.load_query_attrs(data_dir / "eval_split.parquet", n_split)
targets = torch.tensor(heldout["item_id"].to_list(), dtype=torch.long) - 1
(qa, targets) = layout.apply_users_limit(ds.users_limit, qa, targets)
U = qa.shape[0]
ia = item_attrs.numpy()
rev_np = rev.numpy()
print(
    f"items {N:,} [{N}, {C}, {A}]  queries {U:,} (of {n_split:,})  reverse {rev_np.tolist()}"
)

# per clause: the slots any item uses, and no item repeating a value inside a clause
used = {c: [a for a in range(A) if (ia[:, c, a] != -1).any()] for c in range(C)}
for c in range(C):
    s = np.sort(ia[:, c, :], axis=1)
    assert not ((s[:, 1:] == s[:, :-1]) & (s[:, 1:] != -1)).any(), (
        f"clause {c} repeats a value"
    )
base = [int(ia[:, c, :].max()) + 2 for c in range(C)]
_counts: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}


def key(vals: dict[int, np.ndarray], K: tuple[int, ...]) -> np.ndarray:
    k = np.zeros(len(next(iter(vals.values()))), dtype=np.int64)
    for c in K:
        k = k * base[c] + (vals[c] + 1)
    return k


def counts(K: tuple[int, ...]):
    """Sorted keys and item counts of every value combination over clauses K."""
    if K not in _counts:
        parts = []
        for slots in itertools.product(*(used[c] for c in K)):
            vals = {c: ia[:, c, a] for c, a in zip(K, slots, strict=True)}
            ok = np.logical_and.reduce([vals[c] != -1 for c in K])
            parts.append(key({c: v[ok] for c, v in vals.items()}, K))
        _counts[K] = np.unique(np.concatenate(parts), return_counts=True)
    return _counts[K]


def lookup(K: tuple[int, ...], q: np.ndarray) -> np.ndarray:
    if not K:
        return np.full(q.shape[0], N, dtype=np.int64)
    keys, cnt = counts(K)
    k = key({c: q[:, c] for c in K}, K)
    i = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
    return np.where(keys[i] == k, cnt[i], 0)


def pass_counts(qs: np.ndarray, keep: np.ndarray) -> np.ndarray:
    out = np.full(U, -1, dtype=np.int64)
    eff = qs != -1
    for pattern in {tuple(r) for r in eff[keep]}:
        rows = keep & (eff == np.array(pattern)).all(axis=1)
        act = [c for c in range(C) if pattern[c]]
        fwd = tuple(c for c in act if not rev_np[c])
        rv = [c for c in act if rev_np[c]]
        tot = np.zeros(int(rows.sum()), dtype=np.int64)
        for r in range(len(rv) + 1):
            for T in itertools.combinations(rv, r):
                tot += (-1) ** r * lookup(tuple(sorted(fwd + T)), qs[rows])
        out[rows] = tot
    return out


recs = [json.loads(ln) for ln in open(sys.argv[2])]
sweeps = {(fk, sw): cl for fk, d in ds.clauses.items() for sw, cl in d.items()}
fails = 0
for (fk, sweep), clauses in sorted(sweeps.items()):
    qs_t, skip = sweep_qa(qa, clauses)
    qs, keep = qs_t.numpy(), ~skip.numpy()
    pc = pass_counts(qs, keep)
    rng = np.random.default_rng(0)
    for u in rng.choice(
        np.flatnonzero(keep), size=min(n_brute, int(keep.sum())), replace=False
    ):
        m = clause_subset_match(item_attrs.unsqueeze(0), qs_t[u : u + 1], rev)[0]
        assert int(m.sum()) == pc[u], (sweep, u, int(m.sum()), pc[u])
    tif = clause_subset_match(item_attrs[targets].unsqueeze(1), qs_t, rev)[:, 0].numpy()
    held = keep & tif
    mine = {
        "n_items": N,
        "n_queries": U,
        "n_kept": int(keep.sum()),
        "n_queries_oracle": int((keep & (pc > 0)).sum()),
        "n_queries_heldout": int(held.sum()),
        "n_targets_in_filter": int(held.sum()),
        "pass_rate": float((pc[keep] / N).mean()),
    }
    theirs = [
        r
        for r in recs
        if (r["filter_kind"], r["sweep"]) == (fk, sweep) and r["status"] == "ok"
    ]
    bad = [
        (r["algo"], r["backend"], f, r[f], v)
        for r in theirs
        for f, v in mine.items()
        if (abs(r[f] - v) > 1e-12 * max(1, abs(v)) if f == "pass_rate" else r[f] != v)
    ]
    fails += len(bad)
    print(
        f"{fk:6s} {sweep:20s} {len(theirs):2d} records  {mine}  brute {n_brute} ok  "
        + (
            "IDENTICAL"
            if theirs and not bad
            else f"DIFF {bad[:4]}"
            if bad
            else "no records"
        )
    )
sys.exit(1 if fails else 0)
