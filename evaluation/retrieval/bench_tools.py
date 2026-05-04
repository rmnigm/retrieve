"""Shared primitives for the unified retrieval benchmark.

The single driver in [evaluate.py](evaluate.py) imports from here. The
`forward` contract is uniform across all algos:

  ``forward(q, qa_narrow=None) -> (ids, scores)``

Unfiltered cells pass ``qa_narrow=None`` and the algo runs without
masking. Filter cells pass the synthesised per-sweep narrow attrs.
``quality_pass_cached`` / ``perf_pass_cached`` only pass the kwarg
when the caller supplied a tensor — yambda's bare-`q` lambdas keep
working unmodified.
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
# Sample count below which median == p20 == p80 starts collapsing into a single
# cold-start point. Auto-extending the rep budget guarantees enough samples for
# meaningful quantiles, capped at MAX_REP_MS so any genuinely-slow kernel still
# terminates.
MIN_SAMPLES = 30
MAX_REP_MS = 3000.0
# Repeat count for the memory-window. Wider than 5 so filter-bench cells with
# variable mask cardinality don't undersample the worst-case transient.
MEM_REPS = 16


# ----- perf primitives --------------------------------------------------------


def allocated_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_allocated())


def peak_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def pin_precision_globals() -> None:
    """Pin TF32/matmul-precision flags so two runs on the same box don't drift.

    Without this, anything earlier in the process (an upstream import, an
    unrelated model load) could flip ``allow_tf32`` and silently change both
    numerics and throughput. Eval rows must be comparable across runs, so we
    pin to the strict path. Callers that explicitly want TF32 can override
    after this returns.
    """
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def warm_gpu_once(device: torch.device) -> None:
    """One-shot pre-warm to absorb cuBLAS / kernel-loader / pinned-mem init.

    The very first GPU op in a process pays cuBLAS-handle init, kernel-module
    load, and pinned-memory allocator setup — typically 0.5–1.5 s. Per-cell
    warmup loops absorb this on whichever cell happens to run first, which
    is unfair to that cell. Run a tiny mat-mul once at top-of-suite so the
    cost is paid outside any measured window.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    a = torch.randn(64, 64, device=device)
    b = torch.randn(64, 64, device=device)
    for _ in range(3):
        (a @ b).sum().item()  # .item() forces a sync per iter
    torch.cuda.synchronize()
    del a, b


def measure_forward_cuda(
    fn,
    *,
    rep_ms: float = DEFAULT_REP_MS,
    warmup_iters: int = WARMUP_ITERS,
    mem_reps: int = MEM_REPS,
    min_samples: int = MIN_SAMPLES,
    max_rep_ms: float = MAX_REP_MS,
) -> tuple[float, float, float, float, float]:
    """Warmup, capture transient peak in a clean window, then time via do_bench.

    The peak window does NOT use ``do_bench`` because do_bench allocates a ~256 MiB
    L2 cache-buster each call, which would dominate the reported transient peak
    for any small kernel. We measure memory in isolation, then time separately.

    The timing window auto-extends ``rep_ms`` (capped at ``max_rep_ms``) until
    at least ``min_samples`` complete iterations fit. Without this, kernels
    whose single-call latency exceeds ``rep_ms`` collapse to one sample and
    return ``median == p20 == p80`` — a cold-start fingerprint that's
    indistinguishable from a real measurement. The do_bench ``warmup`` budget
    is set to ~half the rep so any cold-start cost (e.g. a last-mile autotune
    config that escaped the upstream warmup) is absorbed before timing.

    No ``empty_cache()`` between warmup and timing: it would force a
    ``cudaMalloc`` on the first measured call, which lands inside do_bench's
    window. Pool reuse across warmup → mem-window → timing is the desired
    behaviour for steady-state numbers.

    Returns ``(median_ms, p20_ms, p80_ms, peak_mib, transient_mib)``.
    """
    # Warmup must happen BEFORE the peak counter reset, otherwise autotune
    # compile-window peak pollutes the metric.
    for _ in range(warmup_iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Memory window. ``empty_cache()`` is intentionally NOT called here:
    # ``memory_allocated()`` is unaffected by it (it only releases pooled-free
    # blocks), and dropping the pool would conflate first-malloc latency into
    # the next mem-window iteration's transient.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    baseline = allocated_bytes()
    keep_alive: list = []  # hold last out across iters so peak captures result+scratch
    for _ in range(mem_reps):
        out = fn()
        if out is not None:
            keep_alive.append(out)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    peak = peak_bytes()
    keep_alive.clear()

    # Timing window. Auto-extend rep_ms until ≥ min_samples are collected.
    cur_rep = float(rep_ms)
    median = p20 = p80 = 0.0
    while True:
        warmup_ms = max(50.0, cur_rep * 0.5)
        median, p20, p80 = ttesting.do_bench(
            fn, quantiles=[0.5, 0.2, 0.8], rep=cur_rep, warmup=warmup_ms
        )
        if median <= 0 or cur_rep >= max_rep_ms:
            break
        if cur_rep / median >= min_samples:
            break
        cur_rep = min(max_rep_ms, median * min_samples * 1.2)
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
    min_samples: int = MIN_SAMPLES,
    max_rep_ms: float = MAX_REP_MS,
) -> tuple[float, float, float, float, float]:
    """Time a CPU forward via wall-clock samples within a ``rep_ms`` budget.

    The deadline is auto-extended (capped at ``max_rep_ms``) when fewer than
    ``min_samples`` iterations fit, mirroring ``measure_forward_cuda`` so a
    slow CPU baseline can't collapse to a single-sample measurement.

    Returns ``(median_ms, p20_ms, p80_ms, 0.0, 0.0)`` — peak/transient memory
    are GPU-only and reported as 0 for CPU rows.
    """
    for _ in range(warmup_iters):
        fn()

    times: list[float] = []
    budget = float(rep_ms)
    while True:
        deadline = time.perf_counter() + budget / 1000.0
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
        if not times:
            # rep_ms shorter than a single-call latency; force one sample
            # and let the convergence check below decide on extending.
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
        if len(times) >= min_samples or budget >= max_rep_ms:
            break
        # Estimate next budget from current median so we converge in one extension.
        cur_med = sorted(times)[len(times) // 2]
        budget = min(max_rep_ms, max(budget * 2, cur_med * min_samples * 1.2))
        times.clear()  # collected times under the old budget aren't representative

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
    amp_enabled = device.type == "cuda"
    for item_seqs, targets, num_targets in tqdm(loader, desc="encode queries"):
        item_seqs = item_seqs.to(device, non_blocking=True)
        # Match training/evaluate.py: fp32 attention NaNs out on left-padded
        # rows where the first positions are fully masked; autocast dispatches
        # to a SDPA kernel that handles the all-masked-keys case.
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            q = model.predict_last(item_seqs)
        q = q.float().detach().cpu()
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


QUALITY_BATCH_SIZE = 64


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
    skip_mask: torch.Tensor | None = None,  # [N] bool on cpu, optional
    batch_size: int = QUALITY_BATCH_SIZE,
) -> tuple[float, float]:
    """Stream cached queries through the index in batches and accumulate metrics.

    Quality is invariant to batch size (same per-row scoring math), so we
    can amortise the per-call CUDA-launch + Python overhead across a wide
    batch — at bs=1 the launch overhead dominates wall on the 313k goodreads
    test stream. Default ``batch_size=64`` cut per-cell wall ~50× in
    benchmarking. Perf rows still report the bs=1/8/16 latencies separately
    via ``perf_pass_cached``.

    For filter-bench rows: pass ``qa_narrow`` on CPU and it's routed to
    ``forward(q, qa_narrow=...)``. ``skip_mask`` drops users whose
    synthesised qa was all -1 for the active clauses.
    """
    accum = None
    n = queries.shape[0]
    for s in tqdm(range(0, n, batch_size), desc=desc, leave=False):
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
    return metrics.get(f"recall@{k}", 0.0), metrics.get(f"ndcg@{k}", 0.0)


def perf_pass_cached(
    forward,
    queries: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    is_cpu: bool,
    seed: int,
    n_pool: int = 4096,
    qa_narrow: torch.Tensor | None = None,  # [N, C_narrow] on cpu, optional
    skip_mask: torch.Tensor | None = None,  # [N] bool on cpu, optional
) -> tuple[float, float, float, float, float]:
    """Time the index forward at the given ``batch_size`` over a query pool.

    Single fixed query collapses p20/p80 to one cluster's traversal cost for
    IVF-style algorithms. Sample ``n_pool`` query batches with a fixed seed
    and round-robin through them inside the timing loop so each iteration
    sees a different cluster.

    For filter-bench rows: pass ``qa_narrow`` and a matching pool of
    attribute batches is sampled from the same row indices, so the scored
    items are realistic for the perf measurement.

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

    if is_cpu:
        return measure_forward_cpu(perf_fn)
    return measure_forward_cuda(perf_fn)


__all__ = [
    "cuda_allocated_mib",
    "encode_queries",
    "load_model_for_eval",
    "perf_pass_cached",
    "pin_precision_globals",
    "quality_pass_cached",
    "warm_gpu_once",
]
