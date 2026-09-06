"""Measurement primitives for harness v2 (H §2.1, §2.3, §2.5, §8.2 E/F).

Imports torch and the stdlib only (``triton`` for its version string) — no data, model or
harness imports. ``run.py`` (C3) composes these: ``setup`` → ``warm_gpu_once`` →
``provenance`` / ``clocks`` → ``timed_build`` → ``index_bytes`` → per ``(k, bs, mode)``
``latency`` (with ``graph_callable`` for ``mode="graph"``) → optional ``profile_once``.

Timing protocol (§2.5): 50 warm-up calls → sync → ``N = clamp(2 s / median_est, 1000,
5000)`` → 3 windows of N calls, each call bracketed by CUDA events on the current stream,
wall clock around the window with one sync at the end. The reported dict comes from the
window with the median median; ``spread`` is over the three window medians. No L2 flush:
the caller's pool rotation keeps index reads naturally cold. Closed-loop, one client.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import platform
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
LIB_SUBTREE = "retrieve/src/retrieve"  # code_version = tree hash of this (H §8.2 B)
MiB = 1024 * 1024


class NotCapturable(RuntimeError):
    """The cell cannot run ``mode="graph"``; ``str(exc)`` is the record's ``reason``."""


# ----- environment ------------------------------------------------------------


def setup(seed: int) -> None:
    """Once per process: seeds, TF32 off, exact fp32 matmuls (§2.1)."""
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def warm_gpu_once() -> None:
    """Pay context / cuBLAS / allocator init outside any measured window."""
    if not torch.cuda.is_available():
        return
    a = torch.randn(64, 64, device="cuda")
    for _ in range(3):
        (a @ a).sum().item()
    torch.cuda.synchronize()


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def code_version() -> str:
    """The library subtree's tree hash at HEAD (H §8.2 B) — the resume key's code component.
    Outside a git checkout: ``files:<sha256>`` over the installed ``retrieve`` sources, a
    disjoint namespace so the two can never be mistaken for each other."""
    tree = _git("rev-parse", f"HEAD:{LIB_SUBTREE}")
    if tree:
        return tree
    import retrieve  # noqa: PLC0415

    root = Path(retrieve.__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return "files:" + h.hexdigest()[:40]


def _nvidia_smi(query: str) -> list[str] | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", "0"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [v.strip() for v in out.strip().split(",")]


def provenance() -> dict[str, Any]:
    """The record's ``env`` block minus clocks (§3.2, §8.2 B/F). ``commit`` is the repo HEAD;
    ``code_version`` is the library subtree's tree hash, so doc churn never invalidates a
    campaign but a kernel edit does. ``dirty`` covers the whole tree."""
    try:
        import triton  # noqa: PLC0415

        triton_v: str | None = triton.__version__
    except ImportError:
        triton_v = None
    driver = _nvidia_smi("driver_version")
    return {
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "driver": driver[0] if driver else None,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton_v,
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "code_version": code_version(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def clocks(expected_sm_mhz: int | None = None) -> dict[str, Any]:
    """Sample SM/mem clocks and the power limit (call *after* warm-up, §2.1).
    ``clocks_locked`` is a recorded observation, not a query: nvidia-smi exposes no lock
    flag, so it is ``sm_mhz`` within 2 % of ``expected_sm_mhz`` (the runbook's
    ``nvidia-smi -lgc`` value); ``None`` when no expectation or no sample."""
    vals = _nvidia_smi("clocks.sm,clocks.mem,clocks.max.sm,power.limit")
    out: dict[str, Any] = dict.fromkeys(("sm_mhz", "mem_mhz", "sm_max_mhz", "power_limit_w"))
    if vals and len(vals) == 4:
        for key, v in zip(out, vals):
            out[key] = float(v) if v.replace(".", "", 1).isdigit() else None
    sm = out["sm_mhz"]
    out["expected_sm_mhz"] = expected_sm_mhz
    out["clocks_locked"] = (
        None
        if sm is None or expected_sm_mhz is None
        else abs(sm - expected_sm_mhz) <= 0.02 * expected_sm_mhz
    )
    return out


# ----- build / memory ---------------------------------------------------------


def timed_build(fn: Callable[[], Any]) -> tuple[Any, float]:
    """``(fn(), seconds)`` with a device sync on each side (§2.3 ``build_s``)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def index_bytes(module: nn.Module | None) -> int:
    """Σ numel·itemsize over ``module.buffers()`` (deduplicated, submodules included)."""
    if module is None:
        return 0
    return sum(b.numel() * b.element_size() for b in module.buffers())


# ----- latency ----------------------------------------------------------------


def _time_calls(fn: Callable[[], Any], n: int) -> tuple[list[float], float]:
    """Per-call ms (CUDA events on the current stream; ``perf_counter`` without CUDA) and
    the window's wall seconds, one sync at the end."""
    if not torch.cuda.is_available():
        ms = []
        t0 = time.perf_counter()
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ms.append((time.perf_counter() - t) * 1e3)
        return ms, time.perf_counter() - t0
    event = torch.cuda.Event
    ev = [(event(enable_timing=True), event(enable_timing=True)) for _ in range(n)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return [s.elapsed_time(e) for s, e in ev], wall


def stats(ms: list[float], wall_s: float, bs: int) -> dict[str, Any]:
    """Robust summary of one window (§2.5 + §8.2 E). Quantiles are linear-interpolated."""
    t = torch.tensor(ms, dtype=torch.float64)
    n = t.numel()
    q1, med, q3, p95, p99 = torch.quantile(
        t, torch.tensor([0.25, 0.5, 0.75, 0.95, 0.99], dtype=torch.float64)
    ).tolist()
    mean, std = t.mean().item(), t.std(correction=0).item()
    iqr = q3 - q1
    return {
        "n": n,
        "median_ms": med,
        "mean_ms": mean,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": t.min().item(),
        "iqr_ms": iqr,
        "qps": n * bs / wall_s if wall_s > 0 else None,
        "host_gap_ms": wall_s * 1e3 / n - mean,
        "outliers_std": int(((t - mean).abs() > 3 * std).sum()),
        "outliers_tukey": int(((t < q1 - 1.5 * iqr) | (t > q3 + 1.5 * iqr)).sum()),
        "load": "closed_loop",
    }


def latency(
    fn: Callable[[], Any],
    *,
    bs: int,
    mode: str,
    warmup: int = 50,
    windows: int = 3,
    target_s: float = 2.0,
    n_min: int = 1000,
    n_max: int = 5000,
) -> tuple[dict[str, Any], list[float]]:
    """§2.5 for one ``(k, bs, mode)`` variant. ``fn`` is the zero-arg call that rotates the
    pool (eager forward, or ``graph_callable``'s replay). Returns ``(perf dict, per-call ms
    of the chosen window)``. Eager only: the first call runs under
    ``set_sync_debug_mode("warn")`` and ``peak_fwd_mib`` is taken over the first window."""
    if mode not in ("eager", "graph"):
        raise ValueError(f"mode must be 'eager' or 'graph', got {mode!r}")
    cuda = torch.cuda.is_available()
    if mode == "eager" and cuda:
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    est, _ = _time_calls(fn, 20)
    n = int(min(max(target_s * 1e3 / max(torch.tensor(est).median().item(), 1e-6), n_min), n_max))

    peak_fwd_mib = None
    runs: list[tuple[list[float], float]] = []
    for w in range(windows):
        measure = w == 0 and mode == "eager" and cuda
        if measure:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
        runs.append(_time_calls(fn, n))
        if measure:
            peak_fwd_mib = (torch.cuda.max_memory_allocated() - before) / MiB
    medians = [torch.tensor(ms).median().item() for ms, _ in runs]
    pick = sorted(range(windows), key=lambda i: medians[i])[windows // 2]
    out = stats(*runs[pick], bs)
    spread = (max(medians) - min(medians)) / medians[pick] if medians[pick] > 0 else 0.0
    out.update(
        mode=mode,
        bs=bs,
        spread=spread,
        unstable=spread > 0.05,
        peak_fwd_mib=peak_fwd_mib,
        window_medians_ms=medians,
    )
    return out, runs[pick][0]


# ----- graph mode / profiling -------------------------------------------------


def graph_callable(module: nn.Module, *example_args: Any, warmup: int = 5) -> Callable[..., Any]:
    """``torch.compile(mode="reduce-overhead", dynamic=False, fullgraph=True)`` of ``module``,
    warmed on ``example_args`` (one static shape = one capture per bs). Raises
    ``NotCapturable`` unless the warm-up recorded ``cudagraph_skips == 0`` and one call is
    exactly one ``cudaGraphLaunch``; modules flagged ``capturable = False`` (the ``official``
    backend, O D7) raise before compiling. Call with the same-shaped pool batches."""
    if not getattr(module, "capturable", True):
        raise NotCapturable("not_capturable")
    if not torch.cuda.is_available():
        raise NotCapturable("cuda_unavailable")
    from torch._dynamo.utils import counters  # noqa: PLC0415
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    torch._dynamo.reset()
    skips_before = int(counters["inductor"]["cudagraph_skips"])
    compiled = torch.compile(module, mode="reduce-overhead", dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for _ in range(warmup):
            compiled(*example_args)
        torch.cuda.synchronize()
        skips = int(counters["inductor"]["cudagraph_skips"]) - skips_before
        if skips:
            raise NotCapturable(f"cudagraph_skips={skips}")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            compiled(*example_args)
            torch.cuda.synchronize()
    launches = sum(e.count for e in prof.key_averages() if e.key == "cudaGraphLaunch")
    if launches != 1:
        raise NotCapturable(f"cudaGraphLaunch per call = {launches}, expected 1")
    return compiled


def profile_once(fn: Callable[[], Any], top: int = 8) -> list[dict[str, Any]]:
    """One eager call under ``torch.profiler``; the ``top`` device kernels by self time
    (``{"kernel", "us", "calls"}``) — the wp4 ``kernel_only.py`` split. Empty without CUDA."""
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    if not torch.cuda.is_available():
        return []
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    kernels = [
        e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    kernels.sort(key=lambda e: e.self_device_time_total, reverse=True)
    return [
        {"kernel": e.key, "us": float(e.self_device_time_total), "calls": int(e.count)}
        for e in kernels[:top]
    ]


__all__ = [
    "LIB_SUBTREE",
    "NotCapturable",
    "clocks",
    "code_version",
    "graph_callable",
    "index_bytes",
    "latency",
    "profile_once",
    "provenance",
    "setup",
    "stats",
    "timed_build",
    "warm_gpu_once",
]
