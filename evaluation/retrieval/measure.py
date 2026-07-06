"""CUDA measurement primitives for the retrieval benchmark.

Timing/memory-window machinery only. Import rule: this module may import
``torch`` and ``triton.testing`` (plus stdlib) and NOTHING else — no
model, training, or harness imports, ever. Model loading lives in
``retrieval.encode``; the quality/perf passes that consume these
primitives live in ``retrieval.passes``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton.testing as ttesting

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


@dataclass(frozen=True)
class PerfStats:
    """One perf measurement: latency quantiles + memory windows."""

    median_ms: float
    p20_ms: float
    p80_ms: float
    peak_mib: float
    transient_mib: float
    # Stage 4a fields land here (mean_ms, p99_ms, throughput_qps, samples) —
    # additive, no call-site churn.

    def as_row_fields(self) -> dict[str, float]:
        return {
            "median_ms": self.median_ms,
            "p20_ms": self.p20_ms,
            "p80_ms": self.p80_ms,
            "peak_mem_mib": self.peak_mib,
            "fwd_scratch_mib": self.transient_mib,
        }


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
) -> PerfStats:
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

    Returns a ``PerfStats``.
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
    return PerfStats(
        median_ms=float(median),
        p20_ms=float(p20),
        p80_ms=float(p80),
        peak_mib=peak_mib,
        transient_mib=transient_mib,
    )


def cuda_allocated_mib() -> float:
    """Currently-allocated CUDA memory in MiB. Deltas around ``build_algorithm``
    isolate the index's marginal cost (excludes shared ``item_embs`` and any
    transient build-time scratch the caching allocator has since released)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 * 1024)


__all__ = [
    "DEFAULT_REP_MS",
    "MAX_REP_MS",
    "MEM_REPS",
    "MIN_SAMPLES",
    "WARMUP_ITERS",
    "PerfStats",
    "allocated_bytes",
    "cuda_allocated_mib",
    "measure_forward_cuda",
    "peak_bytes",
    "pin_precision_globals",
    "warm_gpu_once",
]
