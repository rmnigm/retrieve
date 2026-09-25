#!/usr/bin/env python3
"""A1 diagnostic — is a fetched dataset in the legacy 1-indexed layout?

**Read-only.** It touches nothing; it answers the question that cost A1
its first GPU attempt (2026-09-06).

**The finding.** Commit `3b1b5b3` (2026-05-25, on `main`) switched every
retrieval-time tensor from `[N+1, ...]` — 1-indexed with a padding row at
index 0 — to `[N, ...]` 0-indexed dense, in the loaders, the ETL and the
local data ("Results JSONs regenerated under the new layout"), and
[../../system/datasets.md](../../system/datasets.md) documents that
layout. **The copies published to the Hub, which `eval-fetch` pulls, are
the pre-3b1b5b3 `[N+1, ...]` artifacts** — their own README calls the
tensors `[N+1, ...]` and `content_d128/text_emb.meta.json` still says
`row 0 of text_emb left as zeros for item_id=0 padding`. So this is a
data-provenance mismatch, not a refactor regression: `main` and
`development` have byte-identical `load_filter_assets`.

How each dataset showed it:

* goodreads crashed loudly — `oracle.py:103 RuntimeError: The size of
  tensor a (797085) must match the size of tensor b (797084)`. Its
  embeddings come from the checkpoint, whose `nn.Embedding` always has a
  pad row that `load_sasrec_embeddings` drops, so only `item_attrs_narrow`
  was one row long.
* arxiv did **not** crash: its attrs and its `text_emb` are both
  1-indexed, so they agreed with each other, and only the `-1` target
  shift was wrong. Silently:

      cos(query_emb[u], text_emb[target_id])     = 0.989   <- correct pairing
      cos(query_emb[u], text_emb[target_id - 1]) = 0.625   <- what ran

  0.625 is not obviously wrong for a corpus of neighbouring papers. A
  golden baseline built on it would have been wrong and plausible, which
  is why probe 2 below exists.

The fix lives in the harness loaders (`fix(A1): accept the legacy
1-indexed dataset layout`), which drop the padding row and then assert
the lengths agree. The data is left exactly as published; the real
follow-up is to republish it in the documented layout.

    python3 a1_check_item_alignment.py --root /workspace/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# (dataset dir, relative file, kind) — kind decides what a padding row is.
TARGETS: list[tuple[str, str, str]] = [
    ("goodreads-work-id", "item_attrs_narrow.pt", "attrs"),
    ("arxiv-papers", "item_attrs_narrow.pt", "attrs"),
    ("arxiv-papers", "content/text_emb.pt", "emb"),
    ("arxiv-papers", "content_d128/text_emb.pt", "emb"),
    ("arxiv-papers", "content_d64/text_emb.pt", "emb"),
]


def n_items_of(ds_dir: Path) -> int:
    """Authoritative real-item count: the dense 1-indexed id map."""
    with open(ds_dir / "item_id_map.json") as f:
        m = json.load(f)
    ids = [int(v) for v in m.values()]
    if min(ids) != 1 or max(ids) != len(ids):
        raise SystemExit(
            f"{ds_dir}: item_id_map is not dense 1..N "
            f"(min={min(ids)}, max={max(ids)}, n={len(ids)})"
        )
    return len(ids)


def row0_is_padding(t: torch.Tensor, kind: str) -> bool:
    return bool((t[0] == 0).all()) if kind == "emb" else bool((t[0] == -1).all())


def report(root: Path) -> int:
    """Layout of every tensor, plus the two alignment probes."""
    import polars as pl

    legacy = 0
    print("layout (rows vs item_id_map):")
    for ds, rel, kind in TARGETS:
        p = root / ds / rel
        if not p.exists():
            print(f"  MISSING {p}")
            continue
        n = n_items_of(root / ds)
        t = torch.load(str(p), map_location="cpu", weights_only=True)
        rows = t.shape[0]
        if rows == n:
            verdict = "0-indexed [N] (documented layout)"
        elif rows == n + 1 and row0_is_padding(t, kind):
            verdict = "LEGACY 1-indexed [N+1], row 0 is padding"
            legacy += 1
        else:
            verdict = "UNKNOWN — neither N nor N+1-with-pad"
            legacy += 1
        print(f"  {p}\n      rows={rows} n_items={n}  {verdict}")

    print("\nprobe 1 — held-out target satisfies its own predicate")
    for ds in ("goodreads-work-id", "arxiv-papers"):
        d = root / ds
        a = torch.load(str(d / "item_attrs_narrow.pt"), map_location="cpu", weights_only=True)
        es = pl.read_parquet(d / "eval_split.parquet").head(2000)
        tid = torch.tensor(es["target_id"].to_list())
        qa = torch.tensor(es["query_attrs_narrow"].to_list())
        for off, label in ((0, "attrs[target_id]     (1-indexed)"),
                           (-1, "attrs[target_id - 1] (0-indexed)")):
            hits = tot = 0
            for u in range(len(tid)):
                j = int(tid[u]) + off
                if not (0 <= j < a.shape[0]):
                    continue
                row = a[j]
                hits += all(
                    int(qa[u, c]) == -1 or int(qa[u, c]) in row[c].tolist()
                    for c in range(row.shape[0])
                )
                tot += 1
            print(f"  {ds:20} {label}  {hits}/{tot} = {hits / max(tot, 1):.4f}")

    print("\nprobe 2 — arxiv held-out query vs its target embedding")
    d = root / "arxiv-papers"
    t = torch.load(str(d / "content_d128" / "text_emb.pt"), map_location="cpu", weights_only=True).float()
    q = torch.load(str(d / "content_d128" / "query_emb.pt"), map_location="cpu", weights_only=True).float()
    ho = pl.read_parquet(d / "heldout.parquet")
    tid = torch.tensor(ho["item_id"].to_list())[:3000]
    qs = torch.nn.functional.normalize(q[:3000], dim=-1)
    for off, label in ((0, "text_emb[item_id]    "), (-1, "text_emb[item_id - 1]")):
        v = torch.nn.functional.normalize(t[tid + off], dim=-1)
        print(f"  mean cos(query, {label}) = {float((qs * v).sum(-1).mean()):.4f}")

    print(f"\n{legacy} tensor(s) in the legacy layout. The loaders handle it; the data is untouched.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("/workspace/data"))
    return report(ap.parse_args().root)


if __name__ == "__main__":
    raise SystemExit(main())
