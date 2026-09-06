"""CPU-only tests for ``retrieval.bench`` (harness v2 WP-1).

What can be checked without a GPU: the window statistics against a numpy reference, the
control flow of ``latency`` (window count, chosen window, keys, spread), ``index_bytes``
over shared submodules, ``provenance``'s git fields, the ``clocks`` record shape, and that
``graph_callable`` refuses (with the record's ``reason``) before compiling anything. The
CUDA-event timing path, the cudagraph skip / launch assertions and ``profile_once``'s kernel
table are exercised on the A100 in roadmap C4.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
import torch
from torch import nn

from retrieval import bench


def test_stats_matches_numpy_reference():
    rng = np.random.default_rng(0)
    ms = rng.lognormal(mean=0.0, sigma=0.4, size=500).tolist()
    d = bench.stats(ms, wall_s=1.25, bs=4)
    x = np.asarray(ms)
    q1, med, q3, p95, p99 = np.percentile(x, [25, 50, 75, 95, 99])
    assert d["n"] == 500
    assert d["median_ms"] == pytest.approx(med, abs=1e-9)
    assert d["mean_ms"] == pytest.approx(x.mean(), abs=1e-9)
    assert d["p95_ms"] == pytest.approx(p95, abs=1e-9)
    assert d["p99_ms"] == pytest.approx(p99, abs=1e-9)
    assert d["min_ms"] == pytest.approx(x.min(), abs=1e-9)
    assert d["iqr_ms"] == pytest.approx(q3 - q1, abs=1e-9)
    assert d["qps"] == pytest.approx(500 * 4 / 1.25)
    assert d["host_gap_ms"] == pytest.approx(1.25e3 / 500 - x.mean(), abs=1e-9)
    assert d["outliers_std"] == int((np.abs(x - x.mean()) > 3 * x.std()).sum())
    iqr = q3 - q1
    assert d["outliers_tukey"] == int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
    assert d["load"] == "closed_loop"


def test_latency_windows_and_keys():
    a = torch.randn(16, 16)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return a @ a

    d, samples = bench.latency(fn, bs=2, mode="eager", warmup=3, n_min=25, n_max=25)
    # 1 sync-debug call + 3 warm-up + 20 estimate + 3 windows × 25 (no CUDA: no first call).
    assert calls["n"] == 3 + 20 + 3 * 25
    assert d["n"] == 25 == len(samples)
    assert d["mode"] == "eager" and d["bs"] == 2 and d["load"] == "closed_loop"
    assert len(d["window_medians_ms"]) == 3
    # The reported window is the one with the median median.
    assert d["median_ms"] == sorted(d["window_medians_ms"])[1]
    assert d["spread"] >= 0.0 and isinstance(d["unstable"], bool)
    assert d["peak_fwd_mib"] is None  # no CUDA allocator to read on this box
    floats = ("median_ms", "mean_ms", "p95_ms", "p99_ms", "min_ms", "iqr_ms", "qps", "host_gap_ms")
    assert all(isinstance(d[key], float) for key in floats)


def test_latency_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        bench.latency(lambda: None, bs=1, mode="compiled")


def test_timed_build_returns_value_and_seconds():
    out, s = bench.timed_build(lambda: torch.ones(3))
    assert torch.equal(out, torch.ones(3)) and s > 0.0


class _Leaf(nn.Module):
    def __init__(self, n: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("buf", torch.zeros(n, dtype=dtype))


def test_index_bytes_sums_buffers_once_per_tensor():
    shared = _Leaf(10, torch.int64)  # 80 B
    m = nn.Module()
    m.register_buffer("codes", torch.zeros(7, 3, dtype=torch.int8))  # 21 B
    m.register_buffer("scale", torch.zeros((), dtype=torch.float32))  # 4 B
    m.a = shared
    m.b = shared  # the same filter registered under two names counts once
    assert bench.index_bytes(shared) == 80
    assert bench.index_bytes(m) == 21 + 4 + 80
    assert bench.index_bytes(None) == 0


def test_provenance_fields():
    p = bench.provenance()
    expected = {
        "gpu", "driver", "cuda", "torch", "triton", "commit", "dirty", "git_branch",
        "code_version", "host", "python", "started",
    }  # fmt: skip
    assert expected <= set(p)
    assert p["torch"] == torch.__version__ and isinstance(p["dirty"], bool)
    assert p["started"].endswith("+00:00")
    try:
        tree = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{bench.LIB_SUBTREE}"], cwd=bench.ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git unavailable")
    assert p["code_version"] == tree and len(tree) == 40


def test_clocks_record_shape():
    c = bench.clocks(expected_sm_mhz=1410)
    assert set(c) == {
        "sm_mhz", "mem_mhz", "sm_max_mhz", "power_limit_w", "expected_sm_mhz", "clocks_locked"
    }  # fmt: skip
    assert c["expected_sm_mhz"] == 1410
    if c["sm_mhz"] is None:
        assert c["clocks_locked"] is None  # no nvidia-smi → recorded as unknown, not False
    else:
        assert isinstance(c["clocks_locked"], bool)
    assert bench.clocks()["clocks_locked"] is None


def test_graph_callable_refuses_uncapturable_before_compiling():
    m = _Leaf(4, torch.float32)
    m.capturable = False
    with pytest.raises(bench.NotCapturable, match="not_capturable"):
        bench.graph_callable(m, torch.zeros(1, 4))


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only branch")
def test_graph_callable_refuses_without_cuda():
    with pytest.raises(bench.NotCapturable, match="cuda_unavailable"):
        bench.graph_callable(_Leaf(4, torch.float32), torch.zeros(1, 4))


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only branch")
def test_profile_once_is_empty_without_cuda():
    assert bench.profile_once(lambda: torch.ones(2) + 1) == []


def test_setup_seeds_and_pins_precision():
    bench.setup(7)
    a = torch.randn(4)
    bench.setup(7)
    assert torch.equal(a, torch.randn(4))
    assert torch.get_float32_matmul_precision() == "highest"
