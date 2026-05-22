"""Sanity validator for a synthetic catalog produced by ``synth_arxiv``.

Loads the synth dataset directory, runs the brute-force oracle and a
silvertorch IVF on a small query sample, and reports:

- ``recall@10`` and ``recall@100`` of silvertorch vs the oracle (the oracle
  on the *synth* catalog defines ground truth — this is index-vs-oracle, not
  synth-vs-real)
- ``synth_fraction_topk``: mean fraction of top-100 hits whose item id falls
  in the synthetic range (should rise with N_synth/N_total)
- ``cos_to_nearest_real``: cosine distribution between a small synth sample
  and their closest real neighbour (sanity check on SLERP α range — peak in
  [0.7, 0.95] is healthy; a peak near 1.0 means SLERP is producing
  near-duplicates and α range should be widened)

Run from ``evaluation/``::

    uv run python scripts/check_synth_catalog.py \\
        --data-dir data/arxiv-synth-15m \\
        --n-queries 1000 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

from retrieval.algos.silvertorch import SilvertorchAlgo
from retrieval.loaders import _load_sharded_text_emb


def _load_item_embs(data_dir: Path, content_subdir: str, device: torch.device) -> torch.Tensor:
    content = data_dir / content_subdir
    shard_index = content / "shard_index.json"
    if shard_index.exists():
        embs = _load_sharded_text_emb(shard_index, device)
    else:
        embs = torch.load(str(content / "text_emb.pt"), map_location=device)
    embs[0] = 0.0
    embs = embs.float().contiguous()
    embs = F.normalize(embs, dim=-1)
    return embs


def _load_queries(data_dir: Path, content_subdir: str) -> torch.Tensor:
    queries = torch.load(str(data_dir / content_subdir / "query_emb.pt"), map_location="cpu")
    return F.normalize(queries.float(), dim=-1)


@torch.inference_mode()
def _brute_topk(
    item_embs: torch.Tensor,
    queries: torch.Tensor,
    k: int,
    *,
    q_batch: int,
    device: torch.device,
) -> torch.Tensor:
    """Brute-force ``q @ E.T`` top-k. Padding row id=0 is masked. Returns int64 [Q, k]."""
    n_q = queries.shape[0]
    item_t = item_embs.t().contiguous()
    out = torch.empty((n_q, k), dtype=torch.int64, device=device)
    for s in range(0, n_q, q_batch):
        e = min(s + q_batch, n_q)
        q = queries[s:e].to(device, non_blocking=True)
        scores = q @ item_t
        scores[:, 0] = float("-inf")
        out[s:e] = torch.topk(scores, k, dim=1).indices
    return out


def _recall_at_k(pred: torch.Tensor, oracle: torch.Tensor, k: int) -> float:
    """Mean recall@k of ``pred[:, :k]`` against ``oracle[:, :k]`` row-wise."""
    n_q = pred.shape[0]
    p = pred[:, :k]
    o = oracle[:, :k]
    matches = torch.zeros(n_q, dtype=torch.int64, device=p.device)
    for i in range(n_q):
        matches[i] = torch.isin(p[i], o[i]).sum()
    return float((matches.float() / k).mean().item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--content-subdir", type=str, default="content")
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--q-batch", type=int, default=16,
                        help="brute-force oracle query batch (keep small at large N)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cos-sample", type=int, default=200,
                        help="number of synth items to probe for cos_to_nearest_real")
    parser.add_argument("--n-lists", type=int, default=0,
                        help="silvertorch n_lists; 0 → round(sqrt(N))")
    parser.add_argument("--n-probe", type=int, default=48)
    parser.add_argument("--out", type=str, default="",
                        help="optional path for synth_check.json (default: <data-dir>/synth_check.json)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    content = data_dir / args.content_subdir
    shard_index_path = content / "shard_index.json"
    if not shard_index_path.exists():
        print(f"ERROR no shard_index.json at {shard_index_path} — not a synth catalog", flush=True)
        return 1
    with open(shard_index_path) as f:
        shard_index = json.load(f)
    n_total_plus_one = int(shard_index["n_items_plus_one"])
    n_real_plus_one = int(shard_index["n_real_plus_one"])
    n_synth = n_total_plus_one - n_real_plus_one
    method = shard_index.get("method", "?")
    print(
        f"loaded shard_index: n_total={n_total_plus_one - 1:,} n_real={n_real_plus_one - 1:,} "
        f"n_synth={n_synth:,} method={method}",
        flush=True,
    )

    device = torch.device(args.device)

    t0 = time.monotonic()
    item_embs = _load_item_embs(data_dir, args.content_subdir, device)
    queries_all = _load_queries(data_dir, args.content_subdir)
    print(
        f"loaded catalog ({tuple(item_embs.shape)}) + queries ({tuple(queries_all.shape)}) "
        f"in {time.monotonic() - t0:.1f}s",
        flush=True,
    )

    rng = torch.Generator().manual_seed(args.seed)
    n_q_have = queries_all.shape[0]
    n_q = min(args.n_queries, n_q_have)
    q_idx = torch.randperm(n_q_have, generator=rng)[:n_q]
    queries = queries_all[q_idx]

    K = 100
    print(f"running brute-force oracle on {n_q} queries @ K={K} (q_batch={args.q_batch})", flush=True)
    t1 = time.monotonic()
    oracle_topk = _brute_topk(item_embs, queries, K, q_batch=args.q_batch, device=device)
    oracle_secs = time.monotonic() - t1
    print(f"  oracle done in {oracle_secs:.1f}s ({oracle_secs / n_q * 1000:.1f} ms/query mean)", flush=True)

    n_lists = args.n_lists if args.n_lists > 0 else max(64, int(round(item_embs.shape[0] ** 0.5)))
    print(
        f"building silvertorch (n_lists={n_lists}, n_probe={args.n_probe}, K={K})",
        flush=True,
    )
    t2 = time.monotonic()
    algo = SilvertorchAlgo(
        item_embs,
        k=K,
        filter_kind="none",
        n_lists=n_lists,
        n_probe=args.n_probe,
        seed=args.seed,
    )
    print(f"  silvertorch built in {time.monotonic() - t2:.1f}s", flush=True)

    print("scoring queries through silvertorch", flush=True)
    t3 = time.monotonic()
    pred_ids_chunks = []
    chunk = max(1, args.q_batch)
    for s in range(0, n_q, chunk):
        e = min(s + chunk, n_q)
        q = queries[s:e].to(device, non_blocking=True)
        _, ids = algo(q)
        pred_ids_chunks.append(ids.to(torch.int64))
    pred_ids = torch.cat(pred_ids_chunks, dim=0)
    silvertorch_secs = time.monotonic() - t3
    print(f"  silvertorch done in {silvertorch_secs:.1f}s", flush=True)

    recall_at_10 = _recall_at_k(pred_ids, oracle_topk, 10)
    recall_at_100 = _recall_at_k(pred_ids, oracle_topk, 100)

    in_synth_oracle = (oracle_topk >= n_real_plus_one).float().mean().item()
    in_synth_pred = (pred_ids >= n_real_plus_one).float().mean().item()

    if n_synth > 0:
        cos_n = min(args.cos_sample, n_synth)
        synth_offsets = torch.randperm(n_synth, generator=rng)[:cos_n]
        synth_ids = synth_offsets + n_real_plus_one
        synth_embs = item_embs[synth_ids.to(device)]
        real_embs_t = item_embs[1:n_real_plus_one].t().contiguous()
        cos_scores = synth_embs @ real_embs_t
        nearest_cos = cos_scores.max(dim=1).values
        nearest_cos_cpu = nearest_cos.cpu()
        hist_bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.001]
        hist = torch.histc(
            nearest_cos_cpu,
            bins=len(hist_bins) - 1,
            min=hist_bins[0],
            max=hist_bins[-1],
        ).tolist()
        cos_stats = {
            "n_sampled": int(cos_n),
            "mean": float(nearest_cos_cpu.mean().item()),
            "median": float(nearest_cos_cpu.median().item()),
            "p05": float(torch.quantile(nearest_cos_cpu, 0.05).item()),
            "p95": float(torch.quantile(nearest_cos_cpu, 0.95).item()),
            "hist_bins": hist_bins,
            "hist_counts": hist,
        }
    else:
        cos_stats = None

    result = {
        "n_total": n_total_plus_one - 1,
        "n_real": n_real_plus_one - 1,
        "n_synth": n_synth,
        "method": method,
        "n_queries": n_q,
        "K": K,
        "recall_at_10": recall_at_10,
        "recall_at_100": recall_at_100,
        "synth_fraction_in_oracle_top_K": in_synth_oracle,
        "synth_fraction_in_silvertorch_top_K": in_synth_pred,
        "cos_to_nearest_real": cos_stats,
        "oracle_secs": oracle_secs,
        "silvertorch_secs": silvertorch_secs,
        "silvertorch_n_lists": n_lists,
        "silvertorch_n_probe": args.n_probe,
        "seed": args.seed,
    }

    out_path = Path(args.out) if args.out else data_dir / "synth_check.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("---------------------------------------------------------------")
    print(f"recall@10  silvertorch vs oracle = {recall_at_10:.4f}")
    print(f"recall@100 silvertorch vs oracle = {recall_at_100:.4f}")
    print(f"synth fraction in oracle top-100      = {in_synth_oracle:.4f}")
    print(f"synth fraction in silvertorch top-100 = {in_synth_pred:.4f}")
    if cos_stats is not None:
        print(
            f"cos_to_nearest_real: mean={cos_stats['mean']:.4f} "
            f"median={cos_stats['median']:.4f} p05={cos_stats['p05']:.4f} "
            f"p95={cos_stats['p95']:.4f}"
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
