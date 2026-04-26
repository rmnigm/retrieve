"""Random-vector benchmark harness.

Activated only with ``--bench``. Each cell measures one implementation in
isolation, captures **index-only persistent memory** and **forward-pass
transient peak memory**, and writes a single JSON record to
``bench_results/<run_id>/<algo>__<cell_slug>__<impl>.json``.

Per-algo isolation: callers (e.g. ``tests/bench/run.py``) spawn one pytest
subprocess per ``bench_*.py`` file. Within a subprocess, the ``_isolate_cuda``
autouse fixture clears Python GC, the CUDA cache, and peak-memory counters
between cells. Triton autotune cache is reused across cells of the same algo
within a subprocess (it has to be — that's the whole point of caching).

Markdown rendering is offline. Use ``python -m tests.bench.render`` against the
JSON directory.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
import triton
import triton.testing as ttesting

from tests.conftest import recall_at_k as recall_at_k  # re-export for bench files


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

WARMUP_ITERS = 20  # enough for a 16-config @triton.autotune to settle
DEFAULT_REP_MS = 200.0


# ---------------------------------------------------------------------------
# pytest plumbing
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--bench",
        action="store_true",
        default=False,
        help="run the random-vector benchmark suite",
    )
    parser.addoption(
        "--bench-run-id",
        default=None,
        help="run id (default: env BENCH_RUN_ID, else UTC timestamp)",
    )
    parser.addoption(
        "--bench-out-dir",
        default=None,
        help="output dir (default: env BENCH_OUT_DIR, else bench_results/<run_id>)",
    )
    parser.addoption(
        "--profile",
        action="store_true",
        default=False,
        help="run torch.profiler on the slowest cell of each bench module",
    )
    parser.addoption(
        "--bench-rep",
        type=float,
        default=DEFAULT_REP_MS,
        help=f"ms target per do_bench measurement (default {DEFAULT_REP_MS})",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--bench"):
        return
    skip_bench = pytest.mark.skip(reason="bench tests run only with --bench")
    bench_dir = os.path.join("tests", "bench")
    for it in items:
        if bench_dir in str(it.fspath):
            it.add_marker(skip_bench)


@pytest.fixture(autouse=True)
def _isolate_cuda():
    """Run between cells: drop Python refs, free cached blocks, reset peak counters.

    Triton's compiled kernel cache lives on the autotuner's ``cache`` dict, NOT
    in the CUDA allocator — it survives ``empty_cache`` and is intentionally
    reused across cells of the same algo.
    """
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Per-run output directory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bench_run_id(request) -> str:
    rid = request.config.getoption("--bench-run-id") or os.environ.get("BENCH_RUN_ID")
    if rid:
        return rid
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


@pytest.fixture(scope="session")
def bench_out_dir(request, bench_run_id) -> Path:
    explicit = request.config.getoption("--bench-out-dir") or os.environ.get(
        "BENCH_OUT_DIR"
    )
    out = Path(explicit) if explicit else Path("bench_results") / bench_run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "refs").mkdir(exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------


@dataclass
class BenchRecord:
    run_id: str
    timestamp: str
    algo: str
    impl: str
    cell: str
    params: dict[str, Any]
    memory: dict[str, int]
    timing: dict[str, float]
    correctness: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)


_DEVICE_INFO_CACHE: dict[str, Any] | None = None


def _device_info() -> dict[str, Any]:
    global _DEVICE_INFO_CACHE
    if _DEVICE_INFO_CACHE is not None:
        return _DEVICE_INFO_CACHE
    if not torch.cuda.is_available():
        _DEVICE_INFO_CACHE = {"name": "cpu", "memory_total_mib": 0}
        return _DEVICE_INFO_CACHE
    props = torch.cuda.get_device_properties(0)
    _DEVICE_INFO_CACHE = {
        "name": props.name,
        "memory_total_mib": props.total_memory // (1024 * 1024),
        "capability": f"{props.major}.{props.minor}",
    }
    return _DEVICE_INFO_CACHE


_VERSIONS_CACHE: dict[str, str] | None = None


def _versions() -> dict[str, str]:
    global _VERSIONS_CACHE
    if _VERSIONS_CACHE is not None:
        return _VERSIONS_CACHE
    git_sha = ""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    _VERSIONS_CACHE = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "git_sha": git_sha,
    }
    return _VERSIONS_CACHE


_SLUG_RE = re.compile(r"[^A-Za-z0-9._=+-]+")


def _cell_slug(cell: str) -> str:
    return _SLUG_RE.sub("_", cell).strip("_")


def write_record(rec: BenchRecord, out_dir: Path) -> Path:
    fname = f"{rec.algo}__{_cell_slug(rec.cell)}__{rec.impl}.json"
    path = out_dir / fname
    with path.open("w") as f:
        json.dump(dataclasses.asdict(rec), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


# ---------------------------------------------------------------------------
# Memory measurement primitives
# ---------------------------------------------------------------------------


def _sync_and_clean() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _allocated() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_allocated())


def _peak() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def measure_index(build_and_register: Callable[[], Any]) -> tuple[Any, dict[str, int]]:
    """Build the structure under test; return ``(module, {input_data_bytes, index_bytes})``.

    The caller passes a closure that:
      1. allocates raw input tensors,
      2. instantiates the module,
      3. calls ``module.register_index(...)``,
      4. returns ``(module, [refs_to_inputs])``.

    We snapshot the allocator before, after-input, after-register, and
    after-cleanup. ``index_bytes`` is the delta from baseline to "all inputs
    dropped + transients reclaimed". For ``register_buffer``-by-reference
    modules this still includes the embeddings (they ARE the index). For
    consume-and-quantize modules (V3 / SilverTorch / IVF / Bloom) ``del`` of
    inputs frees the originals, leaving only the structure's own buffers.
    """
    _sync_and_clean()
    baseline = _allocated()

    module, inputs = build_and_register()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    after_register = _allocated()
    input_data_bytes = max(0, after_register - baseline)  # not strictly clean — may include transients

    # Drop the local refs the caller created. The caller is expected to clear
    # any closure-captured refs *before* returning, so this is the last live
    # local on the input tensors.
    for ref in inputs:
        del ref
    del inputs
    _sync_and_clean()

    index_bytes = max(0, _allocated() - baseline)
    return module, {
        "input_data_bytes": input_data_bytes,
        "index_bytes": index_bytes,
    }


def measure_forward(
    fn: Callable[[], Any],
    *,
    rep_ms: float = DEFAULT_REP_MS,
    warmup_iters: int = WARMUP_ITERS,
    mem_reps: int = 5,
) -> dict[str, float | int]:
    """Warmup, capture transient peak in a clean window, then time via do_bench.

    The peak window does NOT use ``triton.testing.do_bench`` because do_bench
    allocates a ~256 MiB L2 cache-buster tensor each call — which would dominate
    the reported transient peak for any small kernel. We measure the kernel's
    own memory footprint by running ``fn`` ``mem_reps`` times in isolation,
    then time it separately.

    Returns: ``median_ms``, ``p20_ms``, ``p80_ms``, ``warmup_iters``, ``rep_ms``,
    ``forward_baseline_bytes``, ``forward_peak_bytes``, ``transient_peak_bytes``.
    """
    # Warmup — fires autotune, settles JIT, primes caches. Must happen BEFORE
    # we reset the peak counter, otherwise autotune compile-window peak pollutes
    # the metric.
    for _ in range(warmup_iters):
        fn()
    _sync_and_clean()

    # Memory measurement window: clean, no do_bench overhead.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    baseline = _allocated()
    for _ in range(mem_reps):
        out = fn()
        del out
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    peak = _peak()

    # Timing window — do_bench's L2 flush is fine here, it's what we want for
    # stable per-call latency.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # release any output tensors held by allocator pool
    median, p20, p80 = ttesting.do_bench(
        fn, quantiles=[0.5, 0.2, 0.8], rep=rep_ms, warmup=50
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return {
        "median_ms": float(median),
        "p20_ms": float(p20),
        "p80_ms": float(p80),
        "warmup_iters": warmup_iters,
        "rep_ms": rep_ms,
        "forward_baseline_bytes": baseline,
        "forward_peak_bytes": peak,
        "transient_peak_bytes": max(0, peak - baseline),
    }


# ---------------------------------------------------------------------------
# Top-level cell helper
# ---------------------------------------------------------------------------


def make_record(
    *,
    run_id: str,
    algo: str,
    impl: str,
    cell: str,
    params: dict[str, Any],
    mem: dict[str, int],
    fwd: dict[str, float | int],
    correctness: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> BenchRecord:
    """Assemble a BenchRecord from measure_index + measure_forward outputs."""
    return BenchRecord(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        algo=algo,
        impl=impl,
        cell=cell,
        params=params,
        memory={
            "input_data_bytes": int(mem["input_data_bytes"]),
            "index_bytes": int(mem["index_bytes"]),
            "forward_baseline_bytes": int(fwd["forward_baseline_bytes"]),
            "forward_peak_bytes": int(fwd["forward_peak_bytes"]),
            "transient_peak_bytes": int(fwd["transient_peak_bytes"]),
        },
        timing={
            "median_ms": float(fwd["median_ms"]),
            "p20_ms": float(fwd["p20_ms"]),
            "p80_ms": float(fwd["p80_ms"]),
            "warmup_iters": int(fwd["warmup_iters"]),
            "rep_ms": float(fwd["rep_ms"]),
        },
        correctness=correctness or {},
        extra=extra or {},
        device=_device_info(),
        versions=_versions(),
    )


def measure_cell(
    *,
    algo: str,
    impl: str,
    cell: str,
    params: dict[str, Any],
    build_and_register: Callable[[], Any],
    forward_fn_factory: Callable[[Any], Callable[[], Any]],
    out_dir: Path,
    run_id: str,
    rep_ms: float,
    correctness: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> BenchRecord:
    """Run one cell end-to-end: build, measure index, build forward fn, measure forward, write JSON.

    ``forward_fn_factory(module)`` returns the ``fn`` that ``do_bench`` will time.
    Keep its closure tight — anything captured here counts toward forward peak.
    """
    module, mem = measure_index(build_and_register)
    fn = forward_fn_factory(module)

    fwd = measure_forward(fn, rep_ms=rep_ms)

    rec = BenchRecord(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        algo=algo,
        impl=impl,
        cell=cell,
        params=params,
        memory={
            "input_data_bytes": int(mem["input_data_bytes"]),
            "index_bytes": int(mem["index_bytes"]),
            "forward_baseline_bytes": int(fwd["forward_baseline_bytes"]),
            "forward_peak_bytes": int(fwd["forward_peak_bytes"]),
            "transient_peak_bytes": int(fwd["transient_peak_bytes"]),
        },
        timing={
            "median_ms": fwd["median_ms"],
            "p20_ms": fwd["p20_ms"],
            "p80_ms": fwd["p80_ms"],
            "warmup_iters": fwd["warmup_iters"],
            "rep_ms": fwd["rep_ms"],
        },
        correctness=correctness or {},
        extra=extra or {},
        device=_device_info(),
        versions=_versions(),
    )
    write_record(rec, out_dir)

    # Drop the module + fn refs so the autouse fixture's empty_cache actually frees them.
    del fn, module
    return rec


# ---------------------------------------------------------------------------
# Correctness helpers (kept; used by bench files)
# ---------------------------------------------------------------------------


def topk_matches(
    out_ids: torch.Tensor,
    out_scores: torch.Tensor,
    ref_ids: torch.Tensor,
    ref_scores: torch.Tensor,
    *,
    score_atol: float = 1e-3,
    score_rtol: float = 1e-3,
) -> bool:
    """Numerics-tolerant top-K equivalence check for benches."""
    common_k = min(out_scores.shape[1], ref_scores.shape[1])
    out_sorted, _ = out_scores.sort(dim=1, descending=True)
    ref_sorted, _ = ref_scores.sort(dim=1, descending=True)
    out_sorted = out_sorted[:, :common_k]
    ref_sorted = ref_sorted[:, :common_k]
    out_finite = torch.where(
        torch.isfinite(out_sorted), out_sorted, torch.zeros_like(out_sorted)
    )
    ref_finite = torch.where(
        torch.isfinite(ref_sorted), ref_sorted, torch.zeros_like(ref_sorted)
    )
    if not torch.allclose(out_finite, ref_finite, atol=score_atol, rtol=score_rtol):
        return False

    b = out_ids.shape[0]
    out_k = out_ids.shape[1]
    ref_k = ref_ids.shape[1]
    for bi in range(b):
        out_set = {
            out_ids[bi, j].item()
            for j in range(out_k)
            if torch.isfinite(out_scores[bi, j])
        }
        ref_set = {
            ref_ids[bi, j].item()
            for j in range(ref_k)
            if torch.isfinite(ref_scores[bi, j])
        }
        diff = out_set.symmetric_difference(ref_set)
        if diff:
            ref_min = (
                ref_scores[bi][torch.isfinite(ref_scores[bi])].min().item()
                if torch.isfinite(ref_scores[bi]).any()
                else float("-inf")
            )
            out_min = (
                out_scores[bi][torch.isfinite(out_scores[bi])].min().item()
                if torch.isfinite(out_scores[bi]).any()
                else float("-inf")
            )
            tol = score_atol + score_rtol * max(abs(ref_min), abs(out_min), 1.0)
            if abs(ref_min - out_min) > tol:
                return False
    return True


def skip_if_insufficient_memory(required_bytes: int, headroom_frac: float = 0.15) -> None:
    if not torch.cuda.is_available():
        return
    free, _ = torch.cuda.mem_get_info()
    budget = free * (1 - headroom_frac)
    if required_bytes > budget:
        pytest.skip(
            f"need ~{required_bytes / 2**30:.1f} GiB, only {free / 2**30:.1f} GiB free"
        )


# ---------------------------------------------------------------------------
# Profiler drill-down (opt-in via --profile)
# ---------------------------------------------------------------------------


def profile_cell(fn, *, label: str, trace_dir: str = "traces") -> None:
    os.makedirs(trace_dir, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for _ in range(5):
            fn()
            prof.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    print(f"\n[profile/{label}] " + "=" * 60, file=sys.stderr)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), file=sys.stderr)
    chrome_path = os.path.join(trace_dir, f"{label}.json")
    prof.export_chrome_trace(chrome_path)
    print(f"[profile/{label}] chrome trace: {chrome_path}", file=sys.stderr)
