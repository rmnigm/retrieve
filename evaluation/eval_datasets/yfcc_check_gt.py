#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy>=1.26", "polars>=1.0"]
# ///
"""Roadmap step E1's gate: reproduce YFCC-10M's *shipped* filtered ground truth.

YFCC-10M is the one dataset in the study whose filtered ground truth we did
not compute — the NeurIPS'23 Big-ANN organisers ship ``GT.public.ibin``,
staged by :mod:`eval_datasets.yfcc` as ``gt_shipped.pt``. E1's gate is that
*our* exact filtered oracle reproduces it on all 100k queries.

What "exact filtered oracle" means here, precisely:

* **Predicate** — an item passes iff its tag bag contains **every** tag of
  the query (conjunctive AND over the query's 1–2 tags). Read from the
  uncapped ``item_tags_csr.pt`` by default (``--tags full``), which is the
  true predicate; ``--tags narrow`` instead evaluates the capped
  ``item_attrs_narrow.pt`` that the harness's clause sweeps use, to measure
  what §3.4 gotcha (a)'s cap costs.
* **Metric** — squared L2 over the raw uint8 vectors (``--metric l2``, the
  default and the only one the shipped GT is defined under). Every distance
  is an integer ≤ 192·255² = 12,484,800 < 2²⁴, and the fp16 embeddings hold
  0..255 exactly, so fp32 arithmetic reproduces them bit-for-bit; TF32 is
  pinned off so a GPU run is the same number. ``--metric ip`` scores
  L2-normalised inner product instead — i.e. what the *harness* measures on
  this dataset — and reports how far that metric's own top-k drifts from the
  shipped GT. It is a diagnostic, never the gate.

Comparison is id-exact, with a tie fallback: a row counts as reproduced if
its ids match the shipped row as a set, or (when equal distances make the
ordering ambiguous) its sorted distance vector equals the shipped one.

Usage::

    export RETRIEVE_DATA_ROOT=/workspace/data

    # E1 gate — the whole 100k query set on the box:
    uv run --directory evaluation python -m eval_datasets.yfcc_check_gt \
        --data-dir /workspace/data/yfcc10m --device cuda \
        --report /workspace/data/yfcc10m/gt_check.json

    # fast CPU sanity check (~20 s):
    uv run --directory evaluation python -m eval_datasets.yfcc_check_gt \
        --data-dir /workspace/data/yfcc10m --device cpu --limit 200

Exit code 0 iff every checked query is reproduced (or, with ``--metric ip``,
always 0 — that mode only reports).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CONTENT_SUBDIR = "content_d192"


# ----- loading ----------------------------------------------------------------


def load_tag_csc(
    data_dir: Path, query_tags: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transpose the item→tags CSR into a tag→items CSC, restricted to the
    tags the queries actually use.

    Returns ``(ptr [n_tags + 1] int64, idx [nnz] int32)`` where
    ``idx[ptr[t]:ptr[t+1]]`` are the ascending 0-indexed rows carrying tag
    ``t``. Tags no query asks for get an empty range, which keeps the
    transpose 29 % smaller without changing any answer.
    """
    blob = torch.load(str(data_dir / "item_tags_csr.pt"), map_location="cpu", weights_only=False)
    indptr = blob["indptr"].numpy()
    indices = blob["indices"].numpy()
    n_items = int(blob["n_items"])
    n_tags = int(blob["n_tags"])

    used = np.zeros(n_tags, dtype=bool)
    used[np.unique(query_tags)] = True
    keep = used[indices]
    item_of = np.repeat(np.arange(n_items, dtype=np.int64), np.diff(indptr))[keep]
    tag_of = indices[keep]
    # Stable sort by tag keeps each tag's item list ascending (the CSR rows
    # were already emitted in item order), which callers rely on.
    order = np.argsort(tag_of, kind="stable")
    idx = item_of[order].astype(np.int32)
    ptr = np.zeros(n_tags + 1, dtype=np.int64)
    np.cumsum(np.bincount(tag_of, minlength=n_tags), out=ptr[1:])
    return ptr, idx


def load_query_predicates(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-query upstream tag ids, ``-1``-padded to width 2, from the raw
    query CSR staged into ``tag_vocab.json`` + ``eval_split.parquet``.

    Returns ``(tags [n_queries, 2] int64 upstream ids, dense [n_queries, 2]
    int64 dense ids)``. Dense ids index the narrow clause tensor; upstream
    ids index the tag CSR/CSC.
    """
    import polars as pl

    with open(data_dir / "tag_vocab.json") as f:
        vocab = np.asarray(json.load(f)["upstream_ids"], dtype=np.int64)
    dense = np.asarray(
        pl.read_parquet(data_dir / "eval_split.parquet")["query_attrs_narrow"].to_list(),
        dtype=np.int64,
    )
    tags = np.where(dense >= 0, vocab[np.clip(dense, 0, None)], -1)
    return tags, dense


# ----- candidate sets ---------------------------------------------------------


def candidates_full(
    ptr: np.ndarray, idx: np.ndarray, tags_row: np.ndarray
) -> np.ndarray:
    """Items passing the true predicate for one query: intersect the CSC
    postings of every active tag.

    Both posting lists are ascending and duplicate-free, so the intersection
    is a ``searchsorted`` probe of the shorter list into the longer one -
    ``O(|a| log |b|)``. ``np.intersect1d`` would concatenate and re-sort both,
    which at YFCC's posting lengths (up to 1.9 M) is the dominant cost of a
    100k-query gate run.
    """
    out: np.ndarray | None = None
    for t in tags_row:
        if t < 0:
            continue
        post = idx[ptr[t] : ptr[t + 1]]
        if out is None:
            out = post
            continue
        a, b = (out, post) if out.size <= post.size else (post, out)
        if b.size == 0 or a.size == 0:
            out = a[:0]
            continue
        j = np.searchsorted(b, a)
        np.clip(j, 0, b.size - 1, out=j)
        out = a[b[j] == a]
    if out is None:
        raise ValueError("query has no active tags")
    return out.astype(np.int64)


def candidates_narrow(narrow: torch.Tensor, dense_row: np.ndarray) -> np.ndarray:
    """Items passing the *capped* clause predicate, evaluated exactly the way
    ``ExactAttributeFilter`` would: per clause, "query value is somewhere in
    this clause's slots"; clauses ANDed; ``-1`` clauses always pass."""
    mask: torch.Tensor | None = None
    for c, v in enumerate(dense_row.tolist()):
        if v < 0:
            continue
        hit = (narrow[:, c, :] == v).any(dim=-1)
        mask = hit if mask is None else (mask & hit)
    if mask is None:
        raise ValueError("query has no active clauses")
    return mask.nonzero(as_tuple=True)[0].cpu().numpy()


# ----- scoring ----------------------------------------------------------------


def topk_for_candidates(
    item_embs: torch.Tensor,
    q: torch.Tensor,
    cand: torch.Tensor,
    k: int,
    metric: str,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact top-k over ``cand`` rows of ``item_embs``.

    ``metric="l2"`` minimises squared L2 in fp32 — exact for uint8-valued
    inputs; ``metric="ip"`` maximises the L2-normalised inner product, the
    harness's metric. Returns ``(ids [k'], scores [k'])`` where the score is
    the squared distance (l2) or the negated similarity (ip), so smaller is
    always better and the two paths share the merge.
    """
    best_s: torch.Tensor | None = None
    best_i: torch.Tensor | None = None
    for start in range(0, cand.numel(), chunk):
        c = cand[start : start + chunk]
        x = item_embs.index_select(0, c).float()
        if metric == "l2":
            s = (x - q).pow(2).sum(dim=-1)
        else:
            s = -(torch.nn.functional.normalize(x, dim=-1) @ q.squeeze(0))
        kk = min(k, s.numel())
        v, i = torch.topk(s, kk, largest=False, sorted=True)
        ids, sc = c[i], v
        if best_s is None:
            best_s, best_i = sc, ids
        else:
            alls = torch.cat([best_s, sc])
            alli = torch.cat([best_i, ids])
            kk = min(k, alls.numel())
            v, i = torch.topk(alls, kk, largest=False, sorted=True)
            best_s, best_i = v, alli[i]
    assert best_s is not None and best_i is not None
    return best_i, best_s


# ----- main check -------------------------------------------------------------


def run_check(args) -> int:
    data_dir = Path(args.data_dir).expanduser()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    gt = torch.load(str(data_dir / "gt_shipped.pt"), map_location="cpu", weights_only=False)
    gt_ids = gt["ids"].numpy()
    gt_d = gt["dists"].numpy()
    k = int(args.k or gt["k"])
    n_all = gt_ids.shape[0]

    tags, dense = load_query_predicates(data_dir)
    if tags.shape[0] != n_all:
        raise RuntimeError(f"eval_split rows {tags.shape[0]} != gt rows {n_all}")

    t0 = time.monotonic()
    print(f"loading item embeddings onto {device} …", flush=True)
    item_embs = torch.load(
        str(data_dir / CONTENT_SUBDIR / "text_emb.pt"), map_location="cpu", weights_only=False
    )
    query_embs = torch.load(
        str(data_dir / CONTENT_SUBDIR / "query_emb.pt"), map_location="cpu", weights_only=False
    )
    if args.tags == "narrow":
        narrow = torch.load(
            str(data_dir / "item_attrs_narrow.pt"), map_location="cpu", weights_only=False
        )
        ptr = idx = None
    else:
        narrow = None
        ptr, idx = load_tag_csc(data_dir, tags[tags >= 0])

    rows = np.arange(n_all)
    if args.limit:
        rng = np.random.default_rng(args.seed)
        rows = np.sort(rng.choice(n_all, min(args.limit, n_all), replace=False))

    # Only move the catalog to the device once we know we need it there.
    item_embs = item_embs.to(device)
    if narrow is not None:
        narrow = narrow.to(device)
    print(f"  ready in {time.monotonic() - t0:.1f}s; checking {rows.size:,} queries", flush=True)

    n_id_exact = 0
    n_tie_equiv = 0
    n_bad = 0
    recall_sum = 0.0
    max_dist_err = 0.0
    pass_sizes: list[int] = []
    bad_examples: list[dict] = []
    t0 = time.monotonic()
    for n_done, r in enumerate(rows, 1):
        if narrow is not None:
            cand_np = candidates_narrow(narrow, dense[r])
        else:
            cand_np = candidates_full(ptr, idx, tags[r])
        pass_sizes.append(int(cand_np.size))
        cand = torch.from_numpy(cand_np).to(device)
        q = query_embs[r : r + 1].to(device).float()
        if args.metric == "ip":
            q = torch.nn.functional.normalize(q, dim=-1)
        ids, sc = topk_for_candidates(item_embs, q, cand, k, args.metric, args.chunk)
        ours = np.sort(ids.cpu().numpy())
        want = np.sort(gt_ids[r][:k])
        recall_sum += float(np.isin(want, ours).mean())
        if ours.size == want.size and np.array_equal(ours, want):
            n_id_exact += 1
        elif args.metric == "l2" and ours.size == want.size and np.allclose(
            np.sort(sc.cpu().numpy()), np.sort(gt_d[r][:k]), rtol=0, atol=0
        ):
            n_tie_equiv += 1
        else:
            n_bad += 1
            if len(bad_examples) < 5:
                bad_examples.append(
                    {
                        "query": int(r),
                        "ours": ours.tolist(),
                        "shipped": want.tolist(),
                        "our_dists": sorted(float(x) for x in sc.cpu().numpy()),
                        "shipped_dists": sorted(float(x) for x in gt_d[r][:k]),
                    }
                )
        if args.metric == "l2":
            common = np.intersect1d(ours, want)
            if common.size:
                d_ours = {int(i): float(s) for i, s in zip(
                    ids.cpu().numpy(), sc.cpu().numpy(), strict=True)}
                d_gt = {int(i): float(s) for i, s in zip(gt_ids[r][:k], gt_d[r][:k], strict=True)}
                for i in common:
                    max_dist_err = max(max_dist_err, abs(d_ours[int(i)] - d_gt[int(i)]))
        if args.progress and n_done % args.progress == 0:
            el = time.monotonic() - t0
            print(
                f"  {n_done:,}/{rows.size:,}  ok={n_id_exact + n_tie_equiv:,} "
                f"bad={n_bad:,}  {el:.0f}s ({n_done / el:.1f} q/s)",
                flush=True,
            )

    elapsed = time.monotonic() - t0
    n = rows.size
    report = {
        "data_dir": str(data_dir),
        "device": args.device,
        "metric": args.metric,
        "tags": args.tags,
        "k": k,
        "n_queries_checked": int(n),
        "n_id_exact": n_id_exact,
        "n_tie_equivalent": n_tie_equiv,
        "n_mismatched": n_bad,
        "reproduced_frac": round((n_id_exact + n_tie_equiv) / max(n, 1), 6),
        "mean_recall_at_k_vs_shipped": round(recall_sum / max(n, 1), 6),
        "max_abs_distance_error": max_dist_err,
        "pass_set_size": {
            "mean": round(float(np.mean(pass_sizes)), 1),
            "median": int(np.median(pass_sizes)),
            "min": int(np.min(pass_sizes)),
            "max": int(np.max(pass_sizes)),
        },
        "wall_clock_sec": round(elapsed, 1),
        "bad_examples": bad_examples,
    }
    print(json.dumps({k2: v for k2, v in report.items() if k2 != "bad_examples"}, indent=2),
          flush=True)
    if bad_examples:
        print("first mismatches:", json.dumps(bad_examples, indent=2), flush=True)
    if args.report:
        Path(args.report).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with open(Path(args.report).expanduser(), "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.report}", flush=True)

    if args.metric != "l2" or args.tags != "full":
        print(
            "NOTE: this is a diagnostic run, not the E1 gate "
            "(the gate is --metric l2 --tags full over all 100k queries).",
            flush=True,
        )
        return 0
    if n_bad:
        print(f"GATE FAILED: {n_bad} of {n} queries not reproduced", flush=True)
        return 1
    print(f"GATE PASSED: {n}/{n} queries reproduce the shipped filtered GT", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="yfcc-check-gt",
        description="Reproduce YFCC-10M's shipped filtered ground truth (roadmap E1 gate)",
    )
    p.add_argument("--data-dir", required=True, help="the prepared data/yfcc10m directory")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--metric", default="l2", choices=["l2", "ip"])
    p.add_argument(
        "--tags",
        default="full",
        choices=["full", "narrow"],
        help="'full' = uncapped item_tags_csr.pt (the gate); "
        "'narrow' = the capped item_attrs_narrow.pt the harness sweeps use",
    )
    p.add_argument("--k", type=int, default=0, help="0 = the shipped depth (10)")
    p.add_argument("--limit", type=int, default=0, help="check a random subset of N queries")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=2_000_000, help="candidates scored per step")
    p.add_argument("--progress", type=int, default=1000, help="log every N queries (0 = off)")
    p.add_argument("--report", default="", help="write the JSON report here")
    args = p.parse_args()
    return run_check(args)


if __name__ == "__main__":
    sys.exit(main())
