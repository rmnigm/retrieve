"""Measurement primitives (docs/system/evaluation.md § Measurement protocol): environment,
provenance, clocks, build timing, index size, the latency windows, graph capture, profiling."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import socket
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import triton
from torch import nn
from torch._dynamo.utils import counters
from torch.profiler import ProfilerActivity, profile

import retrieve

ROOT = Path(__file__).resolve().parents[2]
LIB = Path(retrieve.__file__).resolve().parent
MiB = 1024 * 1024


class NotCapturable(RuntimeError):
    """The cell cannot run ``mode="graph"``; ``str(exc)`` is the record's ``reason``."""


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


def _git(*args: str, cwd: Path = ROOT) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def subtree_dirty() -> bool | None:
    """Uncommitted changes (untracked files included) under ``LIB``; ``None`` outside git."""
    out = _git("status", "--porcelain", "--", ".", cwd=LIB)
    return None if out is None else bool(out)


def repo_dirty() -> bool | None:
    """Uncommitted changes to tracked files anywhere in the repo; ``None`` outside git."""
    out = _git("status", "--porcelain", "--untracked-files=no")
    return None if out is None else bool(out)


def code_version() -> str:
    """The resume key's code component (docs/system/evaluation.md § Resume): the imported
    library's tree hash when clean, else ``files:<sha256>`` of its sources on disk."""
    if subtree_dirty() is False:
        tree = _git("rev-parse", "HEAD:./", cwd=LIB)
        if tree:
            return tree
    return files_hash()


def files_hash() -> str:
    """``files:<sha256>`` over every ``*.py`` under the imported ``retrieve`` package."""
    h = hashlib.sha256()
    for p in sorted(LIB.rglob("*.py")):
        h.update(p.relative_to(LIB).as_posix().encode())
        h.update(p.read_bytes())
    return "files:" + h.hexdigest()[:40]


def inductor_cache_dir(code_version: str, given: str | None) -> str:
    """One inductor cache directory per ``code_version`` under the caller's dir or the temp dir
    (docs/system/evaluation.md § Inductor cache)."""
    base = Path(given) if given else Path(tempfile.gettempdir()) / "bench-inductor"
    return str(base / code_version.replace(":", "-"))


def smi_device() -> str:
    """The device for ``nvidia-smi -i``, which ignores ``CUDA_VISIBLE_DEVICES``: its first
    entry, ``"0"`` when unset."""
    return os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip() or "0"


def _nvidia_smi(query: str) -> list[str] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
                "-i",
                smi_device(),
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [v.strip() for v in out.strip().split(",")]


def official_commit() -> str | None:
    """The installed ``silvertorch``'s git commit (PEP 610 ``direct_url.json``), else ``None``."""
    try:
        dist = importlib.metadata.distribution("silvertorch")
    except importlib.metadata.PackageNotFoundError:
        return None
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    return direct.get("vcs_info", {}).get("commit_id")


# written beside the extension by a build recipe that changes Meta's compiler defaults
OFFICIAL_BUILD_FLAGS = "_build_flags.json"


def official_build(package: str = "silvertorch") -> dict[str, Any] | None:
    """The extension ``package`` would load, found without importing it: path, sha256, size
    and its build flags; ``None`` when the package is absent."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg = Path(next(iter(spec.submodule_search_locations)))
    so = next(iter(sorted(pkg.glob("_C*.so"))), None)
    flags = pkg / OFFICIAL_BUILD_FLAGS
    return {
        "so_path": so and str(so),
        "so_sha256": so and hashlib.sha256(so.read_bytes()).hexdigest(),
        "so_bytes": so and so.stat().st_size,
        "nvcc_append_flags": (
            json.loads(flags.read_text()).get("nvcc_append_flags") if flags.exists() else None
        ),
    }


def provenance() -> dict[str, Any]:
    """The record's ``env`` block minus the clocks (docs/system/evaluation.md § Output)."""
    driver = _nvidia_smi("driver_version")
    return {
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "driver": driver[0] if driver else None,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "official_commit": official_commit(),
        "official_build": official_build(),
        "official_fork_build": official_build("silvertorch_fork"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": subtree_dirty(),
        "repo_dirty": repo_dirty(),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "code_version": code_version(),
        "lib_dir": str(LIB),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def clocks() -> dict[str, Any]:
    """One ``nvidia-smi`` sample of the SM / memory clocks and the power limit; ``None`` values
    without the tool."""
    vals = _nvidia_smi("clocks.sm,clocks.mem,clocks.max.sm,power.limit")
    out: dict[str, Any] = dict.fromkeys(("sm_mhz", "mem_mhz", "sm_max_mhz", "power_limit_w"))
    if vals and len(vals) == 4:
        for key, v in zip(out, vals, strict=True):
            out[key] = float(v) if v.replace(".", "", 1).isdigit() else None
    return out


def clock_report() -> str:
    """``nvidia-smi -q -d CLOCK``, the job log's clock block."""
    try:
        return subprocess.check_output(
            ["nvidia-smi", "-q", "-d", "CLOCK", "-i", smi_device()],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"nvidia-smi unavailable: {exc}\n"


def clock_histogram(samples: Sequence[float]) -> str:
    """One line: ``n`` under-load SM clock samples as ``MHz×count``, highest first."""
    counts = sorted(Counter(int(s) for s in samples).items(), reverse=True)
    return f"sm_mhz under load (n={len(samples)}): " + (
        ", ".join(f"{mhz}×{n}" for mhz, n in counts) or "no samples"
    )


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


def _time_calls(fn: Callable[[], Any], n: int) -> tuple[list[float], float]:
    """Per-call ms (CUDA events; ``perf_counter`` without CUDA) and the window's wall seconds."""
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
    """The summary of one window (§2.5): linear-interpolated quantiles, the trimmed mean over
    ``n // 10`` calls dropped from each end."""
    t = torch.tensor(ms, dtype=torch.float64)
    n = t.numel()
    q1, med, q3, p95, p99 = torch.quantile(
        t, torch.tensor([0.25, 0.5, 0.75, 0.95, 0.99], dtype=torch.float64)
    ).tolist()
    mean, std = t.mean().item(), t.std(correction=0).item()
    iqr = q3 - q1
    cut = n // 10
    return {
        "n": n,
        "median_ms": med,
        "mean_ms": mean,
        "trimmed_mean_ms": t.sort().values[cut : n - cut].mean().item(),
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


def latency(fn: Callable[[], Any], **kw: Any) -> tuple[dict[str, Any], list[float]]:
    """§2.5 for one ``(k, bs, mode)`` variant: :func:`latency_group` of one arm."""
    return latency_group([fn], **kw)[0]


def latency_group(
    fns: Sequence[Callable[[], Any]],
    *,
    bs: int,
    mode: str,
    warmup: int = 50,
    windows: int = 3,
    target_s: float = 2.0,
    n_min: int = 1000,
    n_max: int = 5000,
) -> list[tuple[dict[str, Any], list[float]]]:
    """§2.5 for the arms of one ``(k, bs, mode)`` variant, timed round-robin (window ``i`` of
    every arm before window ``i + 1``, so clock drift lands on every arm alike). Returns, per
    arm, ``(perf dict, per-call ms of its chosen window)``."""
    if mode not in ("eager", "graph"):
        raise ValueError(f"mode must be 'eager' or 'graph', got {mode!r}")
    cuda = torch.cuda.is_available()
    ns = []
    for fn in fns:
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
        med = max(torch.tensor(est).median().item(), 1e-6)
        ns.append(int(min(max(target_s * 1e3 / med, n_min), n_max)))

    peak_fwd_mib: list[float | None] = [None] * len(fns)
    runs: list[list[tuple[list[float], float]]] = [[] for _ in fns]
    window_sm_mhz: list[list[float | None]] = [[] for _ in fns]
    for w in range(windows):
        for i, fn in enumerate(fns):
            measure = w == 0 and mode == "eager" and cuda
            if measure:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
            runs[i].append(_time_calls(fn, ns[i]))
            if measure:
                peak_fwd_mib[i] = (torch.cuda.max_memory_allocated() - before) / MiB
            window_sm_mhz[i].append(clocks()["sm_mhz"] if cuda else None)  # under load
    out = []
    for i in range(len(fns)):
        medians = [torch.tensor(ms).median().item() for ms, _ in runs[i]]
        pick = sorted(range(windows), key=medians.__getitem__)[windows // 2]
        d = stats(*runs[i][pick], bs)
        spread = (max(medians) - min(medians)) / medians[pick] if medians[pick] > 0 else 0.0
        d.update(
            mode=mode,
            bs=bs,
            spread=spread,
            unstable=spread > 0.05,
            peak_fwd_mib=peak_fwd_mib[i],
            window_medians_ms=medians,
            window_sm_mhz=window_sm_mhz[i],
            sm_mhz=window_sm_mhz[i][-1],
        )
        out.append((d, runs[i][pick][0]))
    return out


def graph_callable(module: nn.Module, *example_args: Any, warmup: int = 5) -> Callable[..., Any]:
    """The ``graph`` mode callee: ``torch.compile(mode="reduce-overhead")`` warmed on
    ``example_args``; raises ``NotCapturable`` unless one call is exactly one
    ``cudaGraphLaunch`` with no cudagraph skips. No dynamo reset here: an interleaved group's
    arms stay captured side by side."""
    if not getattr(module, "capturable", True):
        raise NotCapturable("not_capturable")
    if not torch.cuda.is_available():
        raise NotCapturable("cuda_unavailable")

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


PROFILE_PADS_S = (0.0, 0.01, 0.1, 1.0, 5.0)
SENTINEL = "spin_kernel"  # torch.cuda._sleep's kernel; nothing in a forward launches it
RANGE_PREFIX = "## "  # profiler range annotations booked as device events (compiled graph calls)


# Like-for-like kernel groups for T3; a kernel's group is the first whose substrings its
# name contains (evaluation.md § Measurement protocol). Meta's scoring is the process_cluster +
# payload kernels, its bloom test the bloom_search kernels: all "scorer".
KERNEL_SCOPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scorer", ("::fused_kmean_ann::", "::bloom_search::")),
    ("topk", ("::mbtopk::", "::sbtopk::", "radixSortKVInPlace", "bitonicSortKVInPlace")),
    ("epilogue", ("gpu_index_kernel", "_scatter_gather_elementwise_kernel")),
)


def _range(event: Any) -> bool:
    return event.key.startswith(RANGE_PREFIX)


def kernel_scope(name: str) -> str:
    """The ``KERNEL_SCOPES`` group of one kernel name, else ``"other"``."""
    return next((s for s, subs in KERNEL_SCOPES if any(x in name for x in subs)), "other")


def kernel_summary(events: Sequence[Any], top: int = 8) -> dict[str, Any]:
    """``kernels`` (the ``top`` by self time), ``kernels_us`` / ``kernels_calls`` (every kernel)
    and ``kernel_scopes`` of one profiled call. The profiler books a compiled graph's ``## …``
    range as a device event spanning its kernels, so ranges are skipped or they count twice."""
    kernels = sorted(
        (e for e in events if SENTINEL not in e.key),
        key=lambda e: e.self_device_time_total,
        reverse=True,
    )
    scopes = {s: {"us": 0.0, "calls": 0} for s in (*(s for s, _ in KERNEL_SCOPES), "other")}
    for e in kernels:
        if not _range(e):
            sc = scopes[kernel_scope(e.key)]
            sc["us"] += float(e.self_device_time_total)
            sc["calls"] += int(e.count)
    return {
        "kernels": [
            {"kernel": e.key, "us": float(e.self_device_time_total), "calls": int(e.count)}
            for e in kernels[:top]
        ],
        "kernels_us": float(sum(e.self_device_time_total for e in kernels if not _range(e))),
        "kernels_calls": int(sum(e.count for e in kernels if not _range(e))),
        "kernel_scopes": scopes,
    }


def profile_once(fn: Callable[[], Any], top: int = 8) -> dict[str, Any]:
    """``kernel_summary`` of one eager call under ``torch.profiler``, retried with longer idle
    pads until both sentinel kernels are recorded (docs/system/evaluation.md § Measurement
    protocol)."""
    if not torch.cuda.is_available():
        return kernel_summary([], top)
    fn()
    torch.cuda.synchronize()
    for pad in PROFILE_PADS_S:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            time.sleep(pad)
            torch.cuda._sleep(1)
            fn()
            torch.cuda._sleep(1)
            torch.cuda.synchronize()
            time.sleep(pad)
        events = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
        sentinels = sum(e.count for e in events if SENTINEL in e.key)
        if sentinels == 2:
            break
    else:
        raise RuntimeError(
            f"profile_once: {sentinels} of 2 sentinel kernels recorded at pad {pad} s; "
            "the profiler dropped device activities at the window edges"
        )
    return kernel_summary(events, top)
