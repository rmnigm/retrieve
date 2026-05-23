"""Compute per-sweep candidate-set selectivity (fraction of items passing the predicate).

For each (dataset, sweep), takes a sample of query attribute vectors from
eval_split.parquet, applies the sweep's active-clause mask to item_attrs_narrow,
and records the mean per-query candidate fraction.

Output: selectivity.json next to this script, mapping
    {dataset: {sweep_name: {"selectivity": float, "n_clauses": int, "tier": str}}}
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import torch

ROOT = Path(__file__).resolve().parent.parent.parent  # .../evaluation
OUT = Path(__file__).resolve().parent / "selectivity.json"

DATASETS = {
    "arxiv": {
        "data_dir": ROOT / "data/arxiv-retrieval",
        "sweeps": {
            "c0_maincat":    [0],
            "c0c2":          [0, 2],
            "c2_year":       [2],
            "c3_nversions":  [3],
            "all4":          [0, 1, 2, 3],
        },
    },
    "goodreads": {
        "data_dir": ROOT / "data/goodreads-work-id",
        "sweeps": {
            "c0_genre":         [0],
            "c0c1":             [0, 1],
            "c1_lang_reverse":  [1],
            "c2_format":        [2],
            "c3_year":          [3],
            "all4":             [0, 1, 2, 3],
        },
    },
}

# Sample of queries to average selectivity over — full population would be 10k+
# per dataset, sample is plenty for a stable mean and keeps runtime down.
SAMPLE_Q = 512


def selectivity_for(item_attrs: torch.Tensor, qa: torch.Tensor, clause_is_reverse: torch.Tensor, active: list[int]) -> float:
    """Mean fraction of items passing all active clauses for each query.

    Matches ExactAttributeFilter semantics: clause c with reverse[c]==True
    flips match -> non-match. Inactive clauses always pass (qa value -1).
    """
    # item_attrs is (n_items, n_clauses, max_vals) — each item can have up to
    # max_vals values per clause. An item matches a clause if ANY of its values
    # equals the query value (then optionally negated by clause_is_reverse).
    n_items = item_attrs.shape[0]
    fracs: list[float] = []
    for q in qa:
        mask = torch.ones(n_items, dtype=torch.bool)
        for c in active:
            qv = int(q[c].item())
            if qv == -1:
                continue
            m = (item_attrs[:, c, :] == qv).any(dim=1)
            if bool(clause_is_reverse[c].item()):
                m = ~m
            mask &= m
            if not mask.any():
                break
        fracs.append(float(mask.sum().item()) / n_items)
    return sum(fracs) / max(len(fracs), 1)


def classify(sel: float) -> str:
    # Buckets by surviving-fraction; "high selectivity" = filter is strict.
    # Thresholds chosen so each dataset has all three tiers populated.
    if sel < 0.05:
        return "high"
    if sel < 0.25:
        return "mid"
    return "low"


def main() -> None:
    result: dict[str, dict] = {}
    for name, cfg in DATASETS.items():
        data_dir: Path = cfg["data_dir"]
        item_attrs = torch.load(str(data_dir / "item_attrs_narrow.pt"), map_location="cpu")
        clause_rev = torch.load(str(data_dir / "clause_is_reverse_narrow.pt"), map_location="cpu")
        eval_split = pl.read_parquet(data_dir / "eval_split.parquet")
        qa_all = torch.tensor(eval_split["query_attrs_narrow"].to_list(), dtype=torch.long)
        n_q = qa_all.shape[0]
        idx = torch.linspace(0, n_q - 1, steps=min(SAMPLE_Q, n_q)).long()
        qa = qa_all[idx]

        print(f"[{name}] items={item_attrs.shape}  queries={qa.shape}  reverse={clause_rev.tolist()}")
        per_sweep: dict[str, dict] = {}
        for sweep_name, active in cfg["sweeps"].items():
            sel = selectivity_for(item_attrs, qa, clause_rev, active)
            per_sweep[sweep_name] = {
                "selectivity": round(sel, 6),
                "n_clauses": len(active),
                "tier": classify(sel),
            }
            print(f"  {sweep_name:18s}  sel={sel*100:7.3f}%  tier={per_sweep[sweep_name]['tier']}")
        result[name] = per_sweep

    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
