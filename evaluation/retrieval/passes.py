"""Quality and perf passes over the cached query set.

The sweep driver (``retrieval.sweep``, one ``cli/evaluate.py`` subprocess
per algo) streams cached queries through each cell's algo here. The
``forward`` contract is uniform across all algos:

  ``forward(q, qa_narrow=None) -> (ids, scores)``

Unfiltered cells pass ``qa_narrow=None`` and the algo runs without
masking. Filter cells pass the synthesised per-sweep narrow attrs.
``quality_pass_cached`` / ``perf_pass_cached`` only pass the kwarg
when the caller supplied a tensor — bare-``q`` callables keep
working unmodified.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import torch
from tqdm import tqdm

from retrieval.algos import RetrievalAlgo
from retrieval.measure import PerfStats, measure_forward_cuda
from retrieval.metrics import accumulate_metrics, finalize_metrics

Forward = RetrievalAlgo | Callable[..., tuple[torch.Tensor, torch.Tensor]]

QUALITY_BATCH_SIZE = 16  # was 64; reduced because PrefilterKNN[backend="torch"]
# on arxiv/d256 (N≈3M, dim=256) materialises ``[B, P, D]`` fp32 = ~110 GiB at
# bs=64 on loose clause filters (e.g. c3_nversions, P≈1.7M items) and OOMs an
# 80 GB card. bs=16 keeps the same allocation shape as the perf-pass bs=16
# rows which already fit. The 4× chunking reduction lengthens the quality
# stream proportionally but it's a small fraction of total wall-clock.


@dataclass(frozen=True)
class QualityStats:
    """Mean quality metrics at one k. ``metrics.accumulate_metrics`` already
    computes all four per batch; emit them all instead of discarding
    precision/mrr."""

    recall: float
    ndcg: float
    precision: float
    mrr: float


@torch.inference_mode()
def quality_pass_cached(
    forward: Forward,
    queries: torch.Tensor,  # [N, D] on cpu
    targets: torch.Tensor,  # [N, T] on cpu
    num_targets: torch.Tensor,  # [N]   on cpu
    *,
    k: int,
    device: torch.device,
    desc: str,
    qa_narrow: torch.Tensor | None = None,  # [N, C_narrow] on cpu, optional
    skip_mask: torch.Tensor | None = None,  # [N] bool on cpu, optional
    batch_size: int = QUALITY_BATCH_SIZE,
) -> QualityStats:
    """Stream cached queries through the index in batches and accumulate metrics.

    Quality is invariant to batch size (same per-row scoring math), so we
    can amortise the per-call CUDA-launch + Python overhead across a wide
    batch — at bs=1 the launch overhead dominates wall on the 313k goodreads
    test stream; batching at ``QUALITY_BATCH_SIZE`` cut per-cell wall ~50×
    in benchmarking. Perf rows still report the bs=1/8/16 latencies
    separately via ``perf_pass_cached``.

    For filter-bench rows: pass ``qa_narrow`` on CPU and it's routed to
    ``forward(q, qa_narrow=...)``. ``skip_mask`` drops users whose
    synthesised qa was all -1 for the active clauses.
    """
    accum = None
    n = queries.shape[0]
    for s in tqdm(
        range(0, n, batch_size),
        desc=desc,
        leave=False,
        disable=not sys.stderr.isatty(),
    ):
        e = min(s + batch_size, n)
        # Build a per-batch kept index: ranges are contiguous, but skip_mask
        # may punch holes. Materialise the surviving rows once per chunk.
        if skip_mask is not None:
            keep_local = ~skip_mask[s:e].bool()
            if not bool(keep_local.any().item()):
                continue
            sel = keep_local.nonzero(as_tuple=False).reshape(-1) + s
        else:
            sel = torch.arange(s, e)
        q = queries[sel].to(device, non_blocking=True)
        t = targets[sel].to(device, non_blocking=True)
        nt = num_targets[sel].to(device, non_blocking=True)
        kw: dict = {}
        if qa_narrow is not None:
            kw["qa_narrow"] = qa_narrow[sel].to(device, non_blocking=True)
        topk_ids, _ = forward(q, **kw)
        accum = accumulate_metrics(topk_ids, t, nt, [k], accum)
    metrics = finalize_metrics(accum) if accum is not None else {}
    return QualityStats(
        recall=metrics.get(f"recall@{k}", 0.0),
        ndcg=metrics.get(f"ndcg@{k}", 0.0),
        precision=metrics.get(f"precision@{k}", 0.0),
        mrr=metrics.get(f"mrr@{k}", 0.0),
    )


def perf_pass_cached(
    forward: Forward,
    queries: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    n_pool: int = 4096,
    qa_narrow: torch.Tensor | None = None,  # [N, C_narrow] on cpu, optional
    skip_mask: torch.Tensor | None = None,  # [N] bool on cpu, optional
) -> PerfStats:
    """Time the index forward at the given ``batch_size`` over a query pool.

    Single fixed query collapses p20/p80 to one cluster's traversal cost for
    IVF-style algorithms. Sample ``n_pool`` query batches with a fixed seed
    and round-robin through them inside the timing loop so each iteration
    sees a different cluster.

    For filter-bench rows: pass ``qa_narrow`` and a matching pool of
    attribute batches is sampled from the same row indices, so the scored
    items are realistic for the perf measurement.

    Returns a ``PerfStats``.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = queries.shape[0]
    if skip_mask is not None:
        keep_idx = (~skip_mask.bool()).nonzero(as_tuple=False).reshape(-1)
        if keep_idx.numel() == 0:
            return PerfStats(0.0, 0.0, 0.0, 0.0, 0.0)
        rows_local = torch.randint(0, keep_idx.numel(), (n_pool, batch_size), generator=g)
        rows = keep_idx[rows_local.reshape(-1)].reshape(n_pool, batch_size)
    else:
        rows = torch.randint(0, n, (n_pool, batch_size), generator=g)
    flat = rows.reshape(-1)
    pool = queries[flat].reshape(n_pool, batch_size, -1).to(device).contiguous()

    qa_narrow_pool = None
    if qa_narrow is not None:
        qa_narrow_pool = (
            qa_narrow[flat]
            .reshape(n_pool, batch_size, -1)
            .to(device)
            .contiguous()
        )

    counter = {"i": 0}

    def perf_fn():
        idx = counter["i"] % n_pool
        counter["i"] += 1
        q = pool[idx]
        kw: dict = {}
        if qa_narrow_pool is not None:
            kw["qa_narrow"] = qa_narrow_pool[idx]
        with torch.inference_mode():
            # Returning the (ids, scores) tuple lets the memory window capture
            # the marginal cost of one forward including its outputs, not just
            # the in-kernel scratch. ``do_bench`` discards the return value.
            return forward(q, **kw)

    return measure_forward_cuda(perf_fn)


__all__ = [
    "QUALITY_BATCH_SIZE",
    "QualityStats",
    "perf_pass_cached",
    "quality_pass_cached",
]
