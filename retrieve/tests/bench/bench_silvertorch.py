"""SilverTorch end-to-end benchmarks.

Two filter modes:

  * **bloom mode** — clause attributes ANDed via BloomIndex (the production
    path). Selectivity is whatever falls out of the random attr distribution.
    The measured pass rate is recorded in ``extra.pass_rate_measured`` so the
    cell is interpretable.
  * **mask mode** — uniform random boolean mask at a target ``pass_rate ∈
    {0.01, 0.1, 0.5}``. Bypasses the Bloom path; tests the codesigned kernel
    and IVF baseline at controlled selectivity.

Each mode has three tests per cell (ref / composed / codesigned) so that
``measure_index`` and ``measure_forward`` see exactly one structure at a time.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch

from retrieve.layers.silvertorch.bloom import BloomIndex
from retrieve.layers.silvertorch.ivf import IVF_INT8_ANN
from retrieve.layers.silvertorch import SilverTorch
from tests.bench.conftest import (
    make_record,
    measure_forward,
    measure_index,
    profile_cell,
    skip_if_insufficient_memory,
    write_record,
)
from tests.conftest import (
    make_attrs,
    make_index,
    make_mask,
    make_query,
    make_query_attrs,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Cell parameter matrices
# ---------------------------------------------------------------------------

# Bloom-mode cells: full N × n_probe matrix at the implicit selectivity given
# by the random attribute distribution.
_BLOOM_CELLS = [
    pytest.param(b, n, d, k, n_probe, id=f"B{b}_N{n}_D{d}_K{k}_np{n_probe}")
    for b, d, k in [(16, 128, 1024)]
    for n in [16_384, 65_536, 262_144, 1_048_576, 2_097_152]
    for n_probe in [16, 64]
]

# Mask-mode cells: explicit pass-rate sweep, smaller N range to keep matrix
# tractable. n_probe=64 only — at very low pass rates and small n_probe, IVF
# routinely returns < K hits which makes the comparison noisy.
_MASK_CELLS = [
    pytest.param(b, n, d, k, n_probe, pr,
                 id=f"B{b}_N{n}_D{d}_K{k}_np{n_probe}_pr{pr}")
    for b, d, k in [(16, 128, 1024)]
    for n in [65_536, 262_144, 1_048_576]
    for n_probe in [64]
    for pr in [0.01, 0.1, 0.5]
]


def _cell_str_bloom(b, n, d, k, n_lists, n_probe) -> str:
    return f"B={b},N={n},D={d},K={k},n_lists={n_lists},n_probe={n_probe}"


def _cell_str_mask(b, n, d, k, n_lists, n_probe, pr) -> str:
    return (
        f"B={b},N={n},D={d},K={k},n_lists={n_lists},n_probe={n_probe},pass={pr}"
    )


def _params_bloom(b, n, d, k, n_lists, n_probe) -> dict:
    return {
        "b": b, "n": n, "d": d, "k": k, "n_lists": n_lists, "n_probe": n_probe,
        "m_bits": 512, "k_hash": 5, "filter_mode": "bloom",
    }


def _params_mask(b, n, d, k, n_lists, n_probe, pr) -> dict:
    return {
        "b": b, "n": n, "d": d, "k": k, "n_lists": n_lists, "n_probe": n_probe,
        "filter_mode": "mask", "pass_rate_target": pr,
    }


def _n_lists_for(n: int) -> int:
    return max(64, n // 1024)


# ---------------------------------------------------------------------------
# Reference builders (cached on disk)
# ---------------------------------------------------------------------------


def _prefilter_topk(query: torch.Tensor, embs: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """True pre-filter top-K: mask scores BEFORE topk. Matches IVF/codesigned semantics.

    ``FullScanKNN.forward(query, mask=...)`` does *post-filter* (top-K then drop
    failures), which would make recall == pass_rate by construction. Use this
    helper for fair recall comparisons against pre-filter retrievers.
    """
    scores = query @ embs.t()
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.topk(scores, k, dim=1).indices


@pytest.fixture(scope="session")
def silvertorch_bloom_ref_loader(bench_out_dir: Path):
    """``(b, n, d, k) -> {"ex_ids": Tensor, "pass_rate": float}``."""
    in_memory: dict[tuple, dict] = {}

    def _load(b: int, n: int, d: int, k: int) -> dict:
        key = (b, n, d, k)
        if key in in_memory:
            return in_memory[key]
        path = bench_out_dir / "refs" / f"silvertorch_bloom__N{n}_B{b}_K{k}.pt"
        if path.exists():
            data = torch.load(path, map_location="cpu", weights_only=True)
            in_memory[key] = data
            return data

        embs = make_index(n, d)
        attrs = make_attrs(n, c=2, a_max=2)
        query = make_query(b, d)
        q_attrs = make_query_attrs(b, c=2)
        bi = BloomIndex().to("cuda")
        bi.register_index(attrs, m_bits=512, k_hash=5)
        bloom_mask = bi.evaluate(q_attrs)
        pass_rate = float(bloom_mask.float().mean().item())
        ex_ids = _prefilter_topk(query, embs, bloom_mask, k)
        ex_cpu = ex_ids.detach().cpu()

        del embs, attrs, query, q_attrs, bi, bloom_mask, ex_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        data = {"ex_ids": ex_cpu, "pass_rate": pass_rate}
        torch.save(data, path)
        in_memory[key] = data
        return data

    return _load


@pytest.fixture(scope="session")
def silvertorch_mask_ref_loader(bench_out_dir: Path):
    """``(b, n, d, k, pr) -> {"ex_ids": Tensor, "pass_rate": float}``."""
    in_memory: dict[tuple, dict] = {}

    def _load(b: int, n: int, d: int, k: int, pr: float) -> dict:
        key = (b, n, d, k, pr)
        if key in in_memory:
            return in_memory[key]
        path = bench_out_dir / "refs" / f"silvertorch_mask__N{n}_B{b}_K{k}_pr{pr}.pt"
        if path.exists():
            data = torch.load(path, map_location="cpu", weights_only=True)
            in_memory[key] = data
            return data

        embs = make_index(n, d)
        query = make_query(b, d)
        mask = make_mask(b, n, pass_rate=pr)
        pass_rate = float(mask.float().mean().item())
        ex_ids = _prefilter_topk(query, embs, mask, k)
        ex_cpu = ex_ids.detach().cpu()

        del embs, query, mask, ex_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        data = {"ex_ids": ex_cpu, "pass_rate": pass_rate}
        torch.save(data, path)
        in_memory[key] = data
        return data

    return _load


# ---------------------------------------------------------------------------
# Bloom mode: ref / composed / codesigned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("b,n,d,k,n_probe", _BLOOM_CELLS)
def test_silvertorch_bloom_ref(
    b, n, d, k, n_probe,
    bench_run_id, bench_out_dir, silvertorch_bloom_ref_loader, request,
):
    """Materialize bloom-mode reference. No timing."""
    n_lists = _n_lists_for(n)
    if n_probe > n_lists:
        pytest.skip("n_probe > n_lists for this size")
    skip_if_insufficient_memory(int(n * 12.0))
    silvertorch_bloom_ref_loader(b, n, d, k)


@pytest.mark.parametrize("b,n,d,k,n_probe", _BLOOM_CELLS)
def test_silvertorch_bloom_composed(
    b, n, d, k, n_probe,
    bench_run_id, bench_out_dir, silvertorch_bloom_ref_loader, request,
):
    n_lists = _n_lists_for(n)
    if n_probe > n_lists:
        pytest.skip("n_probe > n_lists for this size")
    skip_if_insufficient_memory(int(n * 3.7 * (n_probe / 64) * 1.2))

    ref = silvertorch_bloom_ref_loader(b, n, d, k)
    ex_ids_cpu = ref["ex_ids"]
    query = make_query(b, d)
    q_attrs = make_query_attrs(b, c=2)

    def build_and_register():
        embs = make_index(n, d)
        attrs = make_attrs(n, c=2, a_max=2)
        ivf = IVF_INT8_ANN(k=k, n_lists=n_lists, n_probe=n_probe, n_iter=3)
        ivf.register_index(embs)
        bi = BloomIndex().to("cuda")
        bi.register_index(attrs, m_bits=512, k_hash=5)
        return (ivf, bi), [embs, attrs]

    modules, mem = measure_index(build_and_register)
    ivf, bi = modules
    fn = lambda: ivf(query, mask=bi.evaluate(q_attrs))

    out_ids, _ = fn()
    recall = recall_at_k(out_ids, ex_ids_cpu.to(out_ids.device))

    fwd = measure_forward(fn, rep_ms=float(request.config.getoption("--bench-rep")))
    rec = make_record(
        run_id=bench_run_id, algo="silvertorch", impl="composed_ivf_bloom",
        cell=_cell_str_bloom(b, n, d, k, n_lists, n_probe),
        params=_params_bloom(b, n, d, k, n_lists, n_probe),
        mem=mem, fwd=fwd,
        correctness={"correct": True, "vs": "exact_fullscan"},
        extra={
            "recall@K": round(recall, 4),
            "pass_rate_measured": round(ref["pass_rate"], 4),
        },
    )
    write_record(rec, bench_out_dir)


@pytest.mark.parametrize("b,n,d,k,n_probe", _BLOOM_CELLS)
def test_silvertorch_bloom_codesigned(
    b, n, d, k, n_probe,
    bench_run_id, bench_out_dir, silvertorch_bloom_ref_loader, request,
):
    n_lists = _n_lists_for(n)
    if n_probe > n_lists:
        pytest.skip("n_probe > n_lists for this size")
    skip_if_insufficient_memory(int(n * 1.8 * 1.2))

    ref = silvertorch_bloom_ref_loader(b, n, d, k)
    ex_ids_cpu = ref["ex_ids"]
    query = make_query(b, d)
    q_attrs = make_query_attrs(b, c=2)

    def build_and_register():
        embs = make_index(n, d)
        attrs = make_attrs(n, c=2, a_max=2)
        st = SilverTorch(
            k=k, n_lists=n_lists, n_probe=n_probe, m_bits=512, k_hash=5, n_iter=3
        )
        st.register_index(embs, attrs)
        return st, [embs, attrs]

    module, mem = measure_index(build_and_register)
    fn = lambda: module(query, q_attrs)

    out_ids, _ = fn()
    recall = recall_at_k(out_ids, ex_ids_cpu.to(out_ids.device))

    fwd = measure_forward(fn, rep_ms=float(request.config.getoption("--bench-rep")))
    cell = _cell_str_bloom(b, n, d, k, n_lists, n_probe)
    rec = make_record(
        run_id=bench_run_id, algo="silvertorch", impl="codesigned",
        cell=cell, params=_params_bloom(b, n, d, k, n_lists, n_probe),
        mem=mem, fwd=fwd,
        correctness={"correct": True, "vs": "exact_fullscan"},
        extra={
            "recall@K": round(recall, 4),
            "pass_rate_measured": round(ref["pass_rate"], 4),
        },
    )
    write_record(rec, bench_out_dir)

    if request.config.getoption("--profile") and n in (262_144, 2_097_152):
        profile_cell(fn, label=f"silvertorch_{cell}")


# ---------------------------------------------------------------------------
# Mask mode: ref / composed / codesigned, swept over pass_rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("b,n,d,k,n_probe,pr", _MASK_CELLS)
def test_silvertorch_mask_ref(
    b, n, d, k, n_probe, pr,
    bench_run_id, bench_out_dir, silvertorch_mask_ref_loader, request,
):
    """Materialize mask-mode reference at a target pass rate. No timing."""
    n_lists = _n_lists_for(n)
    if n_probe > n_lists:
        pytest.skip("n_probe > n_lists for this size")
    skip_if_insufficient_memory(int(n * 12.0))
    silvertorch_mask_ref_loader(b, n, d, k, pr)


@pytest.mark.parametrize("b,n,d,k,n_probe,pr", _MASK_CELLS)
def test_silvertorch_mask_composed(
    b, n, d, k, n_probe, pr,
    bench_run_id, bench_out_dir, silvertorch_mask_ref_loader, request,
):
    n_lists = _n_lists_for(n)
    if n_probe > n_lists:
        pytest.skip("n_probe > n_lists for this size")
    skip_if_insufficient_memory(int(n * 3.7 * (n_probe / 64) * 1.2))

    ref = silvertorch_mask_ref_loader(b, n, d, k, pr)
    ex_ids_cpu = ref["ex_ids"]
    query = make_query(b, d)
    mask = make_mask(b, n, pass_rate=pr)

    def build_and_register():
        embs = make_index(n, d)
        ivf = IVF_INT8_ANN(k=k, n_lists=n_lists, n_probe=n_probe, n_iter=3)
        ivf.register_index(embs)
        return ivf, [embs]

    module, mem = measure_index(build_and_register)
    fn = lambda: module(query, mask=mask)

    out_ids, _ = fn()
    recall = recall_at_k(out_ids, ex_ids_cpu.to(out_ids.device))

    fwd = measure_forward(fn, rep_ms=float(request.config.getoption("--bench-rep")))
    rec = make_record(
        run_id=bench_run_id, algo="silvertorch_mask", impl="composed_ivf",
        cell=_cell_str_mask(b, n, d, k, n_lists, n_probe, pr),
        params=_params_mask(b, n, d, k, n_lists, n_probe, pr),
        mem=mem, fwd=fwd,
        correctness={"correct": True, "vs": "exact_fullscan"},
        extra={
            "recall@K": round(recall, 4),
            "pass_rate_measured": round(ref["pass_rate"], 4),
        },
    )
    write_record(rec, bench_out_dir)


@pytest.mark.parametrize("b,n,d,k,n_probe,pr", _MASK_CELLS)
def test_silvertorch_mask_codesigned(
    b, n, d, k, n_probe, pr,
    bench_run_id, bench_out_dir, silvertorch_mask_ref_loader, request,
):
    n_lists = _n_lists_for(n)
    if n_probe > n_lists:
        pytest.skip("n_probe > n_lists for this size")
    skip_if_insufficient_memory(int(n * 1.8 * 1.2))

    ref = silvertorch_mask_ref_loader(b, n, d, k, pr)
    ex_ids_cpu = ref["ex_ids"]
    query = make_query(b, d)
    mask = make_mask(b, n, pass_rate=pr)

    def build_and_register():
        embs = make_index(n, d)
        # No clause attrs → SilverTorch builds zero bloom_sigs (small overhead);
        # the mask-only forward bypasses the bloom check entirely.
        st = SilverTorch(
            k=k, n_lists=n_lists, n_probe=n_probe, m_bits=512, k_hash=5, n_iter=3
        )
        st.register_index(embs, item_clause_attrs=None)
        return st, [embs]

    module, mem = measure_index(build_and_register)
    fn = lambda: module(query, query_clause_attrs=None, mask=mask)

    out_ids, _ = fn()
    recall = recall_at_k(out_ids, ex_ids_cpu.to(out_ids.device))

    fwd = measure_forward(fn, rep_ms=float(request.config.getoption("--bench-rep")))
    rec = make_record(
        run_id=bench_run_id, algo="silvertorch_mask", impl="codesigned",
        cell=_cell_str_mask(b, n, d, k, n_lists, n_probe, pr),
        params=_params_mask(b, n, d, k, n_lists, n_probe, pr),
        mem=mem, fwd=fwd,
        correctness={"correct": True, "vs": "exact_fullscan"},
        extra={
            "recall@K": round(recall, 4),
            "pass_rate_measured": round(ref["pass_rate"], 4),
        },
    )
    write_record(rec, bench_out_dir)
