"""Tests for ``bench.measure``, CPU-only but for the ``gpu``-marked ``profile_once`` one.

What can be checked without a GPU: the window statistics against a numpy reference, the
control flow of ``latency`` (window count, chosen window, keys, spread), ``index_bytes``
over shared submodules, ``provenance``'s git fields, the ``clocks`` record shape, and that
``graph_callable`` refuses (with the record's ``reason``) before compiling anything. On a GPU,
``profile_once`` returns a Triton bloom call's kernels (H-PROFILE). The CUDA-event timing path
and the cudagraph skip / launch assertions are exercised on the A100 in roadmap C4.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import retrieve
from bench import measure as bench
from eval_datasets.layout import atomic_write
from retrieve import SilverTorch


def test_stats_matches_numpy_reference():
    rng = np.random.default_rng(0)
    ms = rng.lognormal(mean=0.0, sigma=0.4, size=500).tolist()
    d = bench.stats(ms, wall_s=1.25, bs=4)
    x = np.asarray(ms)
    q1, med, q3, p95, p99 = np.percentile(x, [25, 50, 75, 95, 99])
    assert d["n"] == 500
    assert d["median_ms"] == pytest.approx(med, abs=1e-9)
    assert d["mean_ms"] == pytest.approx(x.mean(), abs=1e-9)
    assert d["trimmed_mean_ms"] == pytest.approx(np.sort(x)[50:450].mean(), abs=1e-9)
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
    # The under-load clock samples, one per window, are a CUDA-only observation.
    assert d["window_sm_mhz"] == [None] * 3 and d["sm_mhz"] is None
    floats = (
        "median_ms", "mean_ms", "trimmed_mean_ms", "p95_ms", "p99_ms", "min_ms", "iqr_ms", "qps",
        "host_gap_ms",
    )  # fmt: skip
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
        "gpu", "driver", "cuda", "torch", "triton", "commit", "dirty", "repo_dirty",
        "git_branch", "code_version", "host", "python", "started", "official_commit",
        "official_build",
    }  # fmt: skip
    assert expected <= set(p)
    assert p["torch"] == torch.__version__
    assert isinstance(p["dirty"], bool) and isinstance(p["repo_dirty"], bool)
    assert p["started"].endswith("+00:00")
    assert p["lib_dir"] == str(Path(retrieve.__file__).resolve().parent)
    try:
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD:./"], cwd=bench.LIB, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git unavailable")
    assert len(tree) == 40
    if p["dirty"]:  # an uncommitted kernel edit on this box: the content hash, not the tree
        assert p["code_version"].startswith("files:")
    else:
        assert p["code_version"] == tree


class _Dist:
    def __init__(self, direct_url: str | None):
        self.direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        return self.direct_url if name == "direct_url.json" else None


@pytest.mark.parametrize(
    ("dist", "want"),
    [
        (_Dist('{"url": "u", "vcs_info": {"vcs": "git", "commit_id": "abc123"}}'), "abc123"),
        (_Dist('{"url": "file:///x", "dir_info": {}}'), None),
        (_Dist(None), None),
        (None, None),
    ],
    ids=["git", "local_dir", "no_direct_url", "not_installed"],
)
def test_official_commit_reads_pep610(monkeypatch, dist, want):
    def fake(name):
        assert name == "silvertorch"
        if dist is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return dist

    monkeypatch.setattr(importlib.metadata, "distribution", fake)
    assert bench.official_commit() == want


def _fake_git(subtree_status: str, repo_status: str):
    """A ``bench._git`` stand-in: ``status`` answers depend on where it runs, the rest is real."""
    real = bench._git

    def fake(*args, cwd=bench.ROOT):
        if args[0] == "status":
            return subtree_status if cwd == bench.LIB else repo_status
        return real(*args, cwd=cwd)

    return fake


def test_dirty_is_scoped_to_the_library_subtree(monkeypatch):
    tree = bench._git("rev-parse", "HEAD:./", cwd=bench.LIB)
    if not tree:
        pytest.skip("git unavailable")
    # Docs / harness edits: repo dirty, library clean → the tree hash still names the code.
    monkeypatch.setattr(bench, "_git", _fake_git("", " M docs/plans/00-roadmap.md"))
    p = bench.provenance()
    assert (p["dirty"], p["repo_dirty"]) == (False, True) and p["code_version"] == tree
    # An uncommitted kernel edit: the tree hash would lie, so code_version is the content
    # hash of the files on disk — a different resume key from the committed tree's.
    monkeypatch.setattr(bench, "_git", _fake_git(" M retrieve/src/retrieve/x.py", ""))
    p = bench.provenance()
    assert (p["dirty"], p["repo_dirty"]) == (True, False)
    assert p["code_version"] == bench.files_hash() != tree
    assert p["code_version"].startswith("files:")
    # Outside a git checkout: unknown, and the content hash.
    monkeypatch.setattr(bench, "_git", lambda *a, cwd=None: None)
    p = bench.provenance()
    assert p["dirty"] is None and p["repo_dirty"] is None and p["commit"] is None
    assert p["code_version"] == bench.files_hash()


def _git_in(repo, *args):
    return subprocess.check_output(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, text=True
    ).strip()


def test_code_version_names_the_imported_library_not_the_launching_checkout(tmp_path, monkeypatch):
    """H-PROVENANCE: an editable install whose package sits in another checkout (the pods'
    shared venv) is stamped from that checkout, and moves when it moves."""
    other = tmp_path / "other"
    lib = other / "retrieve" / "src" / "retrieve"
    lib.mkdir(parents=True)
    (lib / "__init__.py").write_text("x = 1\n")
    _git_in(other, "init", "-q")
    _git_in(other, "add", ".")
    _git_in(other, "commit", "-q", "-m", "a")
    monkeypatch.setattr(bench, "LIB", lib)
    first = bench.code_version()
    assert first == _git_in(other, "rev-parse", "HEAD:retrieve/src/retrieve")
    assert first != bench._git(
        "rev-parse", "HEAD:retrieve/src/retrieve"
    )  # the launching checkout's
    assert bench.provenance()["dirty"] is False
    # A fast-forward of the imported tree mid-leg: the next child's stamp follows it.
    (lib / "__init__.py").write_text("x = 2\n")
    _git_in(other, "commit", "-q", "-am", "b")
    second = bench.code_version()
    assert second == _git_in(other, "rev-parse", "HEAD:retrieve/src/retrieve") != first
    # An uncommitted edit there: dirty, and the content hash of what is on disk there.
    (lib / "__init__.py").write_text("x = 3\n")
    assert bench.subtree_dirty() is True
    assert bench.code_version() == bench.files_hash() != second
    want = hashlib.sha256(b"__init__.py" + b"x = 3\n").hexdigest()[:40]
    assert bench.files_hash() == "files:" + want
    # Not in a git checkout at all (an installed wheel): unknown, and the content hash.
    plain = tmp_path / "site" / "retrieve"
    plain.mkdir(parents=True)
    (plain / "__init__.py").write_text("x = 3\n")
    monkeypatch.setattr(bench, "LIB", plain)
    assert bench.subtree_dirty() is None and bench.code_version() == "files:" + want


def test_clocks_record_shape():
    c = bench.clocks()
    assert set(c) == {"sm_mhz", "mem_mhz", "sm_max_mhz", "power_limit_w"}
    assert all(v is None or isinstance(v, float) for v in c.values())


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
    zero = {"us": 0.0, "calls": 0}
    assert bench.profile_once(lambda: torch.ones(2) + 1) == {
        "kernels": [],
        "kernels_us": 0.0,
        "kernels_calls": 0,
        "kernel_scopes": dict.fromkeys(("scorer", "topk", "epilogue", "other"), zero),
    }


def test_kernel_summary_sums_every_kernel_but_the_sentinels():
    """H-KSUM: ``kernels`` keeps the top ``top`` by self time; ``kernels_us`` / ``kernels_calls``
    sum all of them, the tail past ``top`` included, and never the two sentinels."""
    ev = lambda key, us, n: SimpleNamespace(key=key, self_device_time_total=us, count=n)  # noqa: E731
    events = [ev(f"k{i}", float(i + 1), i % 3 + 1) for i in range(12)]
    events.insert(5, ev(bench.SENTINEL, 1e6, 2))
    out = bench.kernel_summary(events, top=8)
    assert [k["kernel"] for k in out["kernels"]] == [f"k{i}" for i in range(11, 3, -1)]
    assert out["kernels"][0] == {"kernel": "k11", "us": 12.0, "calls": 3}
    assert out["kernels_us"] == sum(range(1, 13)) == 78.0
    assert out["kernels_calls"] == sum(i % 3 + 1 for i in range(12)) == 24
    assert out["kernels_us"] > sum(k["us"] for k in out["kernels"])


def test_kernel_summary_skips_compiled_graph_ranges_in_the_sums():
    """A compiled call books ``## Call CompiledFxGraph <hash> ##`` as a device event spanning its
    kernels: it stays in the top list but never in ``kernels_us`` / ``kernels_calls``."""
    ev = lambda key, us, n: SimpleNamespace(key=key, self_device_time_total=us, count=n)  # noqa: E731
    events = [
        ev("## Call CompiledFxGraph fabc123 ##", 288.0, 1),
        ev("triton_red_fused_mm_where_0", 152.0, 1),
        ev("gatherTopK", 113.0, 2),
        ev(bench.SENTINEL, 1e6, 2),
    ]
    out = bench.kernel_summary(events, top=8)
    assert out["kernels"][0]["kernel"].startswith("## Call CompiledFxGraph")
    assert out["kernels_us"] == 265.0 and out["kernels_calls"] == 3


def test_kernel_scopes_split_the_full_sum_like_for_like():
    """H-SCOPE: real kernel names from the H-KSUM and V-PROF3 tables. Ours fuses the bloom test into
    its scorer; Meta's scorer is process_cluster + payload kernels + the separate bloom_search
    kernels. The four scopes add up to kernels_us / kernels_calls; ranges and sentinels count
    nowhere."""
    ev = lambda key, us, n=1: SimpleNamespace(key=key, self_device_time_total=us, count=n)  # noqa: E731
    fk = "st::ops::fused_kmean_ann::(anonymous namespace)::"
    bs = "st::ops::bloom_search::(anonymous namespace)::"
    at = "void at::native::"
    scorer = [
        ev("_codesigned_probe_score_kernel", 100.0),
        ev(f"void {fk}process_cluster<signed char, c10::Half, 128, true, int, false>", 25.0),
        ev(f"void {fk}process_cluster_remaining<signed char, c10::Half, 128, true>", 4.6),
        ev(f"{fk}generate_remaining_payload_kernel", 8.5),
        ev(f"{fk}generate_warp_payload_kernel", 3.5),
        ev(f"{fk}generate_cluster_warp_size", 2.3),
        ev(f"void {bs}process_documents<long, 16, true>", 21.5),
        ev(f"void {bs}process_documents_on_assigned_cluster_columns_kernel<16, true>", 9.6),
        ev(f"{bs}generate_column_info_for_clusters_kernel", 2.7),
    ]
    topk = [
        ev(f"{at}mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>", 23.9, 4),
        ev("at::native::mbtopk::computeDigitCumSum(short*, unsigned int*, unsigned int)", 15.0, 4),
        ev(f"{at}sbtopk::gatherTopK<float, unsigned int, 2, false>", 16.8),
        ev(f"{at}radixSortKVInPlace<2, -1, 32, 4, float, long, unsigned int>", 8.9),
        ev(f"{at}bitonicSortKVInPlace<2, -1, 16, 16, float, long>", 7.6),
    ]
    epilogue = [
        ev(f"{at}index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<...>>", 20.8, 2),
        ev(f"{at}_scatter_gather_elementwise_kernel<128, 8, ...>", 8.4),
    ]
    other = [
        ev("probe_ids_kernel", 11.0),
        ev("void gemmSN_TN_kernel<float, 128, 16, 2, 4, 8, 9, false>", 7.2),
        ev(f"{at}unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda>", 15.3, 3),
        ev("Memcpy DtoH (Device -> Pinned)", 5.9, 3),
    ]
    noise = [ev("## Call CompiledFxGraph fabc123 ##", 500.0), ev(bench.SENTINEL, 1e6, 2)]
    out = bench.kernel_summary(scorer + topk + epilogue + other + noise)
    want = {"scorer": scorer, "topk": topk, "epilogue": epilogue, "other": other}
    for scope, evs in want.items():
        got = out["kernel_scopes"][scope]
        assert got["us"] == pytest.approx(sum(e.self_device_time_total for e in evs)), scope
        assert got["calls"] == sum(e.count for e in evs), scope
        assert all(bench.kernel_scope(e.key) == scope for e in evs)
    assert sum(v["us"] for v in out["kernel_scopes"].values()) == pytest.approx(out["kernels_us"])
    assert sum(v["calls"] for v in out["kernel_scopes"].values()) == out["kernels_calls"]


@pytest.mark.gpu
def test_profile_once_records_the_triton_bloom_kernels():
    g = torch.Generator(device="cuda").manual_seed(0)
    embs = nn.functional.normalize(torch.randn(4096, 128, generator=g, device="cuda"), dim=1)
    q = nn.functional.normalize(torch.randn(16, 128, generator=g, device="cuda"), dim=1)
    m = SilverTorch(
        k=100, n_lists=64, n_probe=8, filter_mode="bloom", m_bits=512, k_hash=5, n_iter=3,
        backend="triton",
    )  # fmt: skip
    m.register_index(embs, torch.randint(0, 50, (4096, 1, 2), generator=g, device="cuda"))
    q_attrs = torch.randint(0, 50, (16, 1), generator=g, device="cuda")
    with torch.inference_mode():
        prof = bench.profile_once(lambda: m(q, q_attrs), top=64)
    names = {e["kernel"] for e in prof["kernels"]}
    assert {"probe_ids_kernel", "_codesigned_probe_score_kernel"} <= names
    assert not any(bench.SENTINEL in n for n in names)
    # top 64 covers every kernel of this call, so the list and the sums agree
    assert prof["kernels_us"] == pytest.approx(sum(e["us"] for e in prof["kernels"]))
    assert prof["kernels_calls"] == sum(e["calls"] for e in prof["kernels"])


@pytest.mark.gpu
def test_profile_once_sums_a_compiled_call_without_its_graph_range():
    """A real torch.compile'd call: the profiler books its ``## Call CompiledFxGraph`` range as a
    device event; the top list may carry it, the sums never do."""
    x = torch.randn(4096, 256, device="cuda")
    w = torch.randn(256, 256, device="cuda")
    f = torch.compile(lambda: torch.relu(x @ w).sum(dim=1).topk(16), mode="reduce-overhead")
    with torch.inference_mode():
        for _ in range(3):  # cudagraph trees record on a later call
            f()
        prof = bench.profile_once(f, top=64)
    names = [e["kernel"] for e in prof["kernels"]]
    assert any(n.startswith(bench.RANGE_PREFIX) for n in names)
    kernels = [e for e in prof["kernels"] if not e["kernel"].startswith(bench.RANGE_PREFIX)]
    assert kernels and prof["kernels_us"] == pytest.approx(sum(e["us"] for e in kernels))
    assert prof["kernels_calls"] == sum(e["calls"] for e in kernels)


def test_atomic_write_replaces_or_leaves_nothing(tmp_path):
    bench.atomic_write = atomic_write
    p = tmp_path / "blob.pt"
    bench.atomic_write(p, lambda fh: torch.save({"v": 1}, fh))
    assert torch.load(p, weights_only=True) == {"v": 1}
    assert list(tmp_path.iterdir()) == [p]  # no .tmp left behind

    def torn(fh):
        fh.write(b"half a file")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        bench.atomic_write(p, torn)
    assert torch.load(p, weights_only=True) == {"v": 1}  # the old file survives a failed save
    assert list(tmp_path.iterdir()) == [p]
    q = tmp_path / "new" / "x.npz"
    with pytest.raises(OSError):
        bench.atomic_write(q, torn)
    assert not q.exists() and list(q.parent.iterdir()) == []  # nothing at all, not a torn file


def test_setup_seeds_and_pins_precision():
    bench.setup(7)
    a = torch.randn(4)
    bench.setup(7)
    assert torch.equal(a, torch.randn(4))
    assert torch.get_float32_matmul_precision() == "highest"


def test_inductor_cache_is_keyed_by_code_version_even_when_given():
    """H-INDCACHE: two code_versions, two cache dirs, under the default root and under a
    caller-set directory alike (the caller's dir gets the code_version appended)."""
    a = bench.inductor_cache_dir("0123abcd", None)
    b = bench.inductor_cache_dir("files:4567ef", None)
    assert a != b and a.endswith("/0123abcd") and b.endswith("/files-4567ef")
    given = [bench.inductor_cache_dir(cv, "/scratch/inductor/x") for cv in ("0123abcd", "89ab")]
    assert given == ["/scratch/inductor/x/0123abcd", "/scratch/inductor/x/89ab"]


def test_latency_group_alternates_windows_across_arms():
    """G-interleave: every arm is calibrated first, then round i runs window i of each arm in
    order — A, B, A, B, A, B — each arm with its own window size and summary."""
    calls: list[str] = []
    fns = [lambda: calls.append("A"), lambda: calls.append("B")]
    out = bench.latency_group(fns, bs=1, mode="eager", warmup=2, windows=3, n_min=4, n_max=4)
    calib = 2 + 20  # warm-up + the median estimate, per arm
    assert calls[:calib] == ["A"] * calib and calls[calib : 2 * calib] == ["B"] * calib
    assert calls[2 * calib :] == (["A"] * 4 + ["B"] * 4) * 3
    assert [len(d["window_medians_ms"]) for d, _ in out] == [3, 3]
    assert all(len(ms) == 4 and d["n"] == 4 for d, ms in out)
    one, _ = bench.latency(fns[0], bs=1, mode="eager", warmup=2, windows=3, n_min=4, n_max=4)
    assert set(one) == set(out[0][0])


@pytest.mark.parametrize(
    ("visible", "want"),
    [(None, "0"), ("", "0"), ("1", "1"), ("1,0", "1"), ("GPU-8f2c-aa01,1", "GPU-8f2c-aa01")],
)
def test_smi_device_is_the_first_visible_device(monkeypatch, visible, want):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert bench.smi_device() == want


def test_official_build_records_the_extension_and_its_flags(monkeypatch, tmp_path):
    """OF-11: the loaded extension's sha256, and the nvcc flags its build recipe wrote beside it
    (``None`` = Meta's setup.py defaults); ``None`` overall without silvertorch."""
    pkg = tmp_path / "silvertorch"
    pkg.mkdir()
    (pkg / "_C.cpython-311-x86_64-linux-gnu.so").write_bytes(b"not really an elf")
    spec = SimpleNamespace(submodule_search_locations=[str(pkg)])
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: spec if name == "silvertorch" else None
    )
    got = bench.official_build()
    assert got["so_sha256"] == hashlib.sha256(b"not really an elf").hexdigest()
    assert got["so_path"].endswith(".so") and got["nvcc_append_flags"] is None
    assert got["so_bytes"] == len(b"not really an elf")
    flags = {"nvcc_append_flags": "-O3 -Xcompiler -O3"}
    (pkg / bench.OFFICIAL_BUILD_FLAGS).write_text(json.dumps(flags))
    assert bench.official_build()["nvcc_append_flags"] == "-O3 -Xcompiler -O3"
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert bench.official_build() is None
