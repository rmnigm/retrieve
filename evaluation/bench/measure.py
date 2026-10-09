"""Measurement primitives (H §2.1, §2.3, §2.5, §8.2 E/F): environment, provenance, clocks,
build timing, index size, the latency windows, graph capture and the profiler split.

Imports torch, the stdlib, ``triton`` (version) and ``retrieve`` (files). ``run.py`` composes
these: ``setup`` → ``warm_gpu_once`` → ``provenance`` / ``clocks`` → ``timed_build`` →
``index_bytes`` → per ``(k, bs, mode)`` ``latency`` / ``latency_group`` (with ``graph_callable``
for ``mode="graph"``) → optional ``profile_once``.

Timing protocol (§2.5): 50 warm-up calls → sync → ``N = clamp(2 s / median_est, 1000,
5000)`` → 3 windows of N calls, each call bracketed by CUDA events on the current stream,
wall clock around the window with one sync at the end, one ``nvidia-smi`` SM clock sample
after each window; an interleaved group runs window ``i`` of every arm before window
``i + 1``. The reported dict comes from the window with the median median; ``spread`` is
over the three window medians. No L2 flush: the caller's pool rotation keeps
index reads naturally cold. Closed-loop, one client.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
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
# code_version names the library this process imports, wherever its checkout is (H §8.2 B)
LIB = Path(retrieve.__file__).resolve().parent
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


def _git(*args: str, cwd: Path = ROOT) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def subtree_dirty() -> bool | None:
    """Uncommitted changes (untracked files included) under ``LIB``, in the checkout that holds
    it — the code the harness measures. ``None`` when ``LIB`` is not in a git checkout."""
    out = _git("status", "--porcelain", "--", ".", cwd=LIB)
    return None if out is None else bool(out)


def repo_dirty() -> bool | None:
    """Uncommitted changes to *tracked* files anywhere; informational (docs, plans, harness
    code). ``None`` outside a git checkout. ``evaluation/results/`` is gitignored, so a
    campaign appending to its records never flips it."""
    out = _git("status", "--porcelain", "--untracked-files=no")
    return None if out is None else bool(out)


def code_version() -> str:
    """The resume key's code component (H §8.2 B): the tree hash at HEAD of the imported
    package's directory, in its own checkout, when that subtree is clean; else
    ``files:<sha256>`` over the sources actually on disk (also the value outside a git
    checkout). Read from ``LIB``, not from the checkout that launched ``bench``, so an editable
    install pointing at another tree is stamped as what it runs. The two namespaces are
    disjoint, and a dirty subtree never reuses a cell measured at the committed tree."""
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
    """Roadmap H4 / H-INDCACHE: the inductor cache ``bench run`` uses, always one directory per
    ``code_version``: under the caller's ``TORCHINDUCTOR_CACHE_DIR`` when set, else under the
    temp dir. The FX-graph and AOT-autograd caches key a graph on the custom op, not on its
    ``@triton_op`` body, so a cache shared across library edits replays a stale kernel in graph
    mode (testing.md § Running); a reused caller dir must not serve another tree's graphs."""
    base = Path(given) if given else Path(tempfile.gettempdir()) / "bench-inductor"
    return str(base / code_version.replace(":", "-"))


def smi_device() -> str:
    """The process's device for ``nvidia-smi -i``, which ignores ``CUDA_VISIBLE_DEVICES``: its
    first entry (an index or a UUID), ``"0"`` when unset or empty."""
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
    """``vcs_info.commit_id`` of the installed ``silvertorch`` (its PEP 610
    ``direct_url.json``); ``None`` when it is not installed or not installed from git."""
    try:
        dist = importlib.metadata.distribution("silvertorch")
    except importlib.metadata.PackageNotFoundError:
        return None
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    return direct.get("vcs_info", {}).get("commit_id")


def provenance() -> dict[str, Any]:
    """The record's ``env`` block minus clocks (§3.2, §8.2 B/F). ``commit`` is the repo HEAD;
    ``dirty`` is ``subtree_dirty()`` — the flag ``report.py`` enforces (§8.2 F) — and
    ``repo_dirty`` the informational whole-tree one; ``code_version`` follows ``dirty`` (a
    kernel edit, committed or not, invalidates a campaign; doc churn never does).
    ``official_commit`` is the installed ``silvertorch`` build's git commit."""
    driver = _nvidia_smi("driver_version")
    return {
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "driver": driver[0] if driver else None,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "official_commit": official_commit(),
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
    """One ``nvidia-smi`` sample of the SM / memory clocks and the power limit; every value
    ``None`` without the tool. Whether the GPU is loaded when this runs is the caller's
    business: ``run.py`` records the process-start sample as ``env.sm_mhz_idle`` and
    ``latency`` samples again right after each window, under load."""
    vals = _nvidia_smi("clocks.sm,clocks.mem,clocks.max.sm,power.limit")
    out: dict[str, Any] = dict.fromkeys(("sm_mhz", "mem_mhz", "sm_max_mhz", "power_limit_w"))
    if vals and len(vals) == 4:
        for key, v in zip(out, vals, strict=True):
            out[key] = float(v) if v.replace(".", "", 1).isdigit() else None
    return out


def clock_report() -> str:
    """``nvidia-smi -q -d CLOCK``: the job log's start / end clock block (current, application,
    default and max clocks, clock policy)."""
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
    """Robust summary of one window (§2.5 + §8.2 E). Quantiles are linear-interpolated;
    ``trimmed_mean_ms`` drops ``n // 10`` calls from each end."""
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
    """§2.5 for one ``(k, bs, mode)`` variant: :func:`latency_group` of one arm. The defaults
    are :func:`latency_group`'s, repeated because ``report.methodology`` prints them from this
    signature."""
    kw = {"warmup": warmup, "windows": windows, "target_s": target_s}
    return latency_group([fn], bs=bs, mode=mode, n_min=n_min, n_max=n_max, **kw)[0]


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
    """§2.5 for the arms of one ``(k, bs, mode)`` variant, timed round-robin: every arm is
    warmed up and calibrated (its own ``N``), then round ``i`` runs window ``i`` of each arm in
    order (A, B, A, B, …), so clock drift lands on every arm alike. Each ``fn`` is a zero-arg
    call that rotates the pool (eager forward, or ``graph_callable``'s replay). Returns, per
    arm, ``(perf dict, per-call ms of its chosen window)``. Eager only: the first call runs
    under ``set_sync_debug_mode("warn")`` and ``peak_fwd_mib`` is taken over the arm's first
    window. ``window_sm_mhz`` is the SM clock sampled right after each window's sync, while
    the GPU is still at its load clock; ``sm_mhz`` is its last element — the per-variant value
    H §7's unlocked-clock fallback needs and the only clock sample ``clocks_drift`` compares.
    Samples are ``None`` without CUDA."""
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


# ----- graph mode / profiling -------------------------------------------------


def graph_callable(module: nn.Module, *example_args: Any, warmup: int = 5) -> Callable[..., Any]:
    """``torch.compile(mode="reduce-overhead", dynamic=False, fullgraph=True)`` of ``module``,
    warmed on ``example_args`` (one static shape = one capture per bs). It does not reset
    dynamo, so the arms of an interleaved group stay captured side by side; the caller resets
    before the first capture of a variant. Raises
    ``NotCapturable`` unless the warm-up recorded ``cudagraph_skips == 0`` and one call is
    exactly one ``cudaGraphLaunch``; modules flagged ``capturable = False`` (the ``official``
    backend, O D7) raise before compiling. Call with the same-shaped pool batches."""
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


# H-SCOPE: like-for-like kernel groups for T3; a kernel's group is the first whose substrings its
# name contains (evaluation.md § Measurement protocol). Our scorer fuses the bloom test; Meta's
# scoring is process_cluster + payload kernels, its bloom test bloom_search kernels: all "scorer".
KERNEL_SCOPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scorer", ("_codesigned_probe_score", "::fused_kmean_ann::", "::bloom_search::")),
    ("topk", ("::mbtopk::", "::sbtopk::", "radixSortKVInPlace", "bitonicSortKVInPlace")),
    ("epilogue", ("gpu_index_kernel", "_scatter_gather_elementwise_kernel")),
)


def _range(event: Any) -> bool:
    return event.key.startswith(RANGE_PREFIX)


def kernel_scope(name: str) -> str:
    """The ``KERNEL_SCOPES`` group of one kernel name, else ``"other"``."""
    return next((s for s, subs in KERNEL_SCOPES if any(x in name for x in subs)), "other")


def kernel_summary(events: Sequence[Any], top: int = 8) -> dict[str, Any]:
    """The entry fields of one profiled call from its device events (``key``,
    ``self_device_time_total``, ``count``), sentinels dropped: ``kernels``, the ``top`` by self
    time (``{"kernel", "us", "calls"}``), and ``kernels_us`` / ``kernels_calls``, the device time
    and launches summed over every kernel (H-KSUM: the top-8 sum is only a lower bound). The sums
    skip ``## …`` events: the profiler books a compiled graph's ``## Call CompiledFxGraph …`` range
    as a device event spanning that graph's kernels, which would count them twice.
    ``kernel_scopes`` splits the same sums by ``kernel_scope`` (H-SCOPE): ``{scope: {"us",
    "calls"}}`` for scorer, topk, epilogue and other, adding up to ``kernels_us`` /
    ``kernels_calls``."""
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
    """One eager call under ``torch.profiler``: ``kernel_summary`` of its device kernels — the
    wp4 ``kernel_only.py`` split plus the full sum. Empty kernels and zero sums without CUDA.
    The call is bracketed by two sentinel kernels and the profile is retried with a longer
    idle pad on both sides of the window until both sentinels are recorded; raises if they
    never are (evaluation.md § Measurement protocol)."""
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


__all__ = [
    "LIB",
    "NotCapturable",
    "clock_histogram",
    "clock_report",
    "clocks",
    "code_version",
    "files_hash",
    "graph_callable",
    "index_bytes",
    "kernel_scope",
    "kernel_summary",
    "latency",
    "latency_group",
    "profile_once",
    "provenance",
    "repo_dirty",
    "setup",
    "smi_device",
    "stats",
    "subtree_dirty",
    "timed_build",
    "warm_gpu_once",
]
