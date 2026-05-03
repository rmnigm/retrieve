"""Shared primitives for the unified retrieval benchmark.

The single driver in [evaluate.py](evaluate.py) imports from here. The
`forward` contract supported by the passes is the union of:

  - bare yambda: ``forward(q) -> (ids, scores)``
  - filtered:    ``forward(q, qa_narrow=..., qa_wide=...) -> (ids, scores)``

`quality_pass_cached` / `perf_pass_cached` only pass `qa_*` kwargs when
the caller supplied corresponding pool tensors — yambda's bare-`q`
lambdas keep working unmodified.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import triton.testing as ttesting
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from retrieval.metrics import accumulate_metrics, finalize_metrics
from training.evaluate import EvalDataset, collate_eval
from training.model import GSASRec

# Long enough to settle triton autotune, short enough to keep cells <60 s of overhead.
WARMUP_ITERS = 20
DEFAULT_REP_MS = 200.0


# ----- perf primitives --------------------------------------------------------


def allocated_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_allocated())


def peak_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def measure_forward_cuda(
    fn,
    *,
    rep_ms: float = DEFAULT_REP_MS,
    warmup_iters: int = WARMUP_ITERS,
    mem_reps: int = 5,
) -> tuple[float, float, float, float, float]:
    """Warmup, capture transient peak in a clean window, then time via do_bench.

    The peak window does NOT use ``do_bench`` because do_bench allocates a ~256 MiB
    L2 cache-buster each call, which would dominate the reported transient peak
    for any small kernel. We measure memory in isolation, then time separately.

    Returns ``(median_ms, p20_ms, p80_ms, peak_mib, transient_mib)``.
    """
    # Warmup must happen BEFORE the peak counter reset, otherwise autotune
    # compile-window peak pollutes the metric.
    for _ in range(warmup_iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    baseline = allocated_bytes()
    for _ in range(mem_reps):
        out = fn()
        del out
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    peak = peak_bytes()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    median, p20, p80 = ttesting.do_bench(fn, quantiles=[0.5, 0.2, 0.8], rep=rep_ms, warmup=50)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    peak_mib = peak / (1024 * 1024)
    transient_mib = max(0, peak - baseline) / (1024 * 1024)
    return float(median), float(p20), float(p80), peak_mib, transient_mib


def measure_forward_cpu(
    fn,
    *,
    rep_ms: float = DEFAULT_REP_MS,
    warmup_iters: int = 3,
) -> tuple[float, float, float, float, float]:
    """Time a CPU forward via wall-clock samples within a ``rep_ms`` budget.

    Returns ``(median_ms, p20_ms, p80_ms, 0.0, 0.0)`` — peak/transient memory
    are GPU-only and reported as 0 for CPU rows.
    """
    for _ in range(warmup_iters):
        fn()
    times: list[float] = []
    deadline = time.perf_counter() + rep_ms / 1000.0
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    if not times:
        # Pathological case: rep_ms < single-call latency.
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    n = len(times)
    median = times[n // 2]
    p20 = times[max(0, (n * 20) // 100)]
    p80 = times[min(n - 1, (n * 80) // 100)]
    return float(median), float(p20), float(p80), 0.0, 0.0


def cuda_allocated_mib() -> float:
    """Currently-allocated CUDA memory in MiB. Deltas around ``build_algorithm``
    isolate the index's marginal cost (excludes shared ``item_embs`` and any
    transient build-time scratch the caching allocator has since released)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 * 1024)


# ----- model load -------------------------------------------------------------

# Fallback hyperparams for the legacy 500M ckpts that don't ship a config.json
# alongside the .pt — they all happen to share these.
D128_DROP05_DEFAULTS = {
    "max_seq_length": 200,
    "embedding_dim": 128,
    "num_heads": 2,
    "num_blocks": 2,
    "ffn_hidden_dim": 512,
    "dropout": 0.5,
    "reuse_item_embeddings": False,
}


def load_model_for_eval(
    checkpoint_path: Path, num_items: int, device: torch.device
) -> GSASRec:
    """Load a `GSASRec` from disk for retrieval eval.

    Reads the sibling ``config.json`` when present (5B / freshly-trained
    ckpts have it via ``GSASRecConfig.save``); falls back to the
    d128-drop0.5 hyperparams for the legacy 500M ckpts that don't ship one.
    """
    cfg_path = checkpoint_path.parent / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        params = {k: cfg[k] for k in D128_DROP05_DEFAULTS if k in cfg}
    else:
        params = dict(D128_DROP05_DEFAULTS)
    logger.info("model: GSASRec params={}", params)
    model = GSASRec(num_items=num_items, **params).to(device).eval()
    state = torch.load(str(checkpoint_path), map_location=str(device), weights_only=True)
    model.load_state_dict(state)
    return model


# ----- query cache (encode the test set once) --------------------------------


@torch.inference_mode()
def encode_queries(
    model: nn.Module,
    data_path: Path,
    *,
    max_length: int,
    encode_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run ``model.predict_last`` over the entire eval split once.

    Returns ``(queries [N, D], targets [N, T_max], num_targets [N])`` on CPU.
    Encoding batch size is independent of the per-algo perf batch.
    """
    dataset = EvalDataset(str(data_path), max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=encode_batch_size,
        shuffle=False,
        collate_fn=collate_eval,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    q_chunks: list[torch.Tensor] = []
    t_lists: list[list[int]] = []
    for item_seqs, targets, num_targets in tqdm(loader, desc="encode queries"):
        item_seqs = item_seqs.to(device, non_blocking=True)
        q = model.predict_last(item_seqs).detach().cpu()
        q_chunks.append(q)
        for row, n in zip(targets, num_targets, strict=True):
            t_lists.append(row[:n].tolist())

    queries = torch.cat(q_chunks, dim=0)
    n_users = queries.shape[0]
    n_targets = torch.tensor([len(t) for t in t_lists], dtype=torch.long)
    t_max = max(int(n_targets.max().item()), 1)
    targets = torch.full((n_users, t_max), -1, dtype=torch.long)
    for i, t in enumerate(t_lists):
        if t:
            targets[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    return queries, targets, n_targets


# ----- passes -----------------------------------------------------------------


@torch.inference_mode()
def quality_pass_cached(
    forward,
    queries: torch.Tensor,  # [N, D] on cpu
    targets: torch.Tensor,  # [N, T] on cpu
    num_targets: torch.Tensor,  # [N]   on cpu
    *,
    k: int,
    device: torch.device,
    desc: str,
    qa_narrow: torch.Tensor | None = None,  # [N, C_narrow] on cpu, optional
    qa_wide: torch.Tensor | None = None,  # [N, Q_wide]    on cpu, optional
    skip_mask: torch.Tensor | None = None,  # [N] bool on cpu, optional
) -> tuple[float, float]:
    """Stream cached queries through the index at bs=1 and accumulate metrics.

    Quality is invariant to perf batch size (same scoring math), so we pay
    the bs=1 stream once per ``(algo, k)`` cell and attach the result to every
    bs row of that cell.

    For filter-bench rows: pass ``qa_narrow`` and/or ``qa_wide`` on CPU and
    they're routed to ``forward(q, qa_narrow=..., qa_wide=...)``. ``skip_mask``
    drops users (e.g. wide eval rows where the target had zero surviving
    shelves).
    """
    accum = None
    n = queries.shape[0]
    for i in tqdm(range(n), desc=desc, leave=False):
        if skip_mask is not None and bool(skip_mask[i].item()):
            continue
        q = queries[i : i + 1].to(device, non_blocking=True)
        t = targets[i : i + 1].to(device, non_blocking=True)
        nt = num_targets[i : i + 1].to(device, non_blocking=True)
        kw: dict = {}
        if qa_narrow is not None:
            kw["qa_narrow"] = qa_narrow[i : i + 1].to(device, non_blocking=True)
        if qa_wide is not None:
            kw["qa_wide"] = qa_wide[i : i + 1].to(device, non_blocking=True)
        topk_ids, _ = forward(q, **kw)
        accum = accumulate_metrics(topk_ids, t, nt, [k], accum)
    metrics = finalize_metrics(accum) if accum is not None else {}
    return metrics.get(f"recall@{k}", 0.0), metrics.get(f"ndcg@{k}", 0.0)


def perf_pass_cached(
    forward,
    queries: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    is_cpu: bool,
    seed: int,
    n_pool: int = 64,
    qa_narrow: torch.Tensor | None = None,  # [N, C_narrow] on cpu, optional
    qa_wide: torch.Tensor | None = None,  # [N, Q_wide]    on cpu, optional
    skip_mask: torch.Tensor | None = None,  # [N] bool on cpu, optional
) -> tuple[float, float, float, float, float]:
    """Time the index forward at the given ``batch_size`` over a query pool.

    Single fixed query collapses p20/p80 to one cluster's traversal cost for
    IVF-style algorithms. Sample ``n_pool`` query batches with a fixed seed
    and round-robin through them inside the timing loop so each iteration
    sees a different cluster.

    For filter-bench rows: pass ``qa_narrow`` and/or ``qa_wide`` and a
    matching pool of attribute batches is sampled from the same row indices,
    so the scored items are realistic for the perf measurement.

    Returns ``(median_ms, p20_ms, p80_ms, peak_mib, transient_mib)``.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = queries.shape[0]
    if skip_mask is not None:
        keep_idx = (~skip_mask.bool()).nonzero(as_tuple=False).reshape(-1)
        if keep_idx.numel() == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        rows_local = torch.randint(0, keep_idx.numel(), (n_pool, batch_size), generator=g)
        rows = keep_idx[rows_local.reshape(-1)].reshape(n_pool, batch_size)
    else:
        rows = torch.randint(0, n, (n_pool, batch_size), generator=g)
    flat = rows.reshape(-1)
    pool = queries[flat].reshape(n_pool, batch_size, -1).to(device).contiguous()

    qa_narrow_pool = None
    qa_wide_pool = None
    if qa_narrow is not None:
        qa_narrow_pool = (
            qa_narrow[flat]
            .reshape(n_pool, batch_size, -1)
            .to(device)
            .contiguous()
        )
    if qa_wide is not None:
        qa_wide_pool = (
            qa_wide[flat]
            .reshape(n_pool, batch_size, -1)
            .to(device)
            .contiguous()
        )

    counter = {"i": 0}

    def perf_fn() -> None:
        idx = counter["i"] % n_pool
        counter["i"] += 1
        q = pool[idx]
        kw: dict = {}
        if qa_narrow_pool is not None:
            kw["qa_narrow"] = qa_narrow_pool[idx]
        if qa_wide_pool is not None:
            kw["qa_wide"] = qa_wide_pool[idx]
        with torch.inference_mode():
            forward(q, **kw)

    if is_cpu:
        return measure_forward_cpu(perf_fn)
    return measure_forward_cuda(perf_fn)


__all__ = [
    "cuda_allocated_mib",
    "encode_queries",
    "load_model_for_eval",
    "perf_pass_cached",
    "quality_pass_cached",
]
