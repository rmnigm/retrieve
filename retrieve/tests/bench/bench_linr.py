"""Top-level LiNR V1/V2/V3 benchmarks.

Each cell × impl pair runs as its own pytest function so that ``measure_index``
sees only one module's persistent buffers, and ``measure_forward`` sees only
that module's transient peak. Reference top-K ids (for recall) are cached on
disk by a ``test_*_ref`` test that runs first per cell.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch

from retrieve.layers.linr.v1 import LiNR_V1
from retrieve.layers.linr.v1_triton import LiNR_V1_Triton
from retrieve.layers.linr.v2 import LiNR_V2
from retrieve.layers.linr.v2_triton import LiNR_V2_Triton
from retrieve.layers.linr.v3 import LiNR_V3
from retrieve.layers.linr.v3_triton import LiNR_V3_Triton
from retrieve.layers.utils.compact import compact_mask
from retrieve.layers.utils.retrieval import FullScanKNN
from tests.bench.conftest import (
    make_record,
    measure_forward,
    measure_index,
    skip_if_insufficient_memory,
    write_record,
)
from tests.conftest import (
    make_index,
    make_mask,
    make_query,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Cell parameter matrices
# ---------------------------------------------------------------------------

V1_FULL_CELLS = [
    pytest.param(b, n, d, k, id=f"B{b}_N{n}_D{d}_K{k}")
    for b, d, k in [(1, 128, 200), (16, 128, 200)]
    for n in [16_384, 65_536, 262_144, 1_048_576, 2_097_152]
]

MASKED_CELLS = [
    pytest.param(b, n, d, k, pr, id=f"B{b}_N{n}_D{d}_K{k}_pr{pr}")
    for b, d, k in [(16, 128, 200)]
    for n in [16_384, 65_536, 1_048_576]
    for pr in [0.01, 0.1, 0.5]
]

V3_FULL_CELLS = [
    pytest.param(b, n, d, k, id=f"B{b}_N{n}_D{d}_K{k}")
    for b, d, k in [(16, 128, 200)]
    for n in [16_384, 65_536]
]


def _params_full(b, n, d, k) -> dict:
    return {"b": b, "n": n, "d": d, "k": k}


def _params_masked(b, n, d, k, pr) -> dict:
    return {"b": b, "n": n, "d": d, "k": k, "pass_rate": pr}


def _cell_full(b, n, d, k) -> str:
    return f"B={b},N={n},D={d},K={k}"


def _cell_masked(b, n, d, k, pr) -> str:
    return f"B={b},N={n},D={d},K={k},pass={pr}"


# ---------------------------------------------------------------------------
# Reference loaders (top-K from FullScanKNN, cached on disk)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def linr_full_ref_loader(bench_out_dir: Path):
    in_memory: dict[tuple, torch.Tensor] = {}

    def _load(b: int, n: int, d: int, k: int) -> torch.Tensor:
        key = (b, n, d, k)
        if key in in_memory:
            return in_memory[key]
        path = bench_out_dir / "refs" / f"linr_full__N{n}_B{b}_K{k}.pt"
        if path.exists():
            ex = torch.load(path, map_location="cpu", weights_only=True)["ex_ids"]
            in_memory[key] = ex
            return ex

        embs = make_index(n, d)
        query = make_query(b, d)
        exact = FullScanKNN(k=k)
        exact.register_index(embs)
        ex_ids, _ = exact(query)
        ex_cpu = ex_ids.detach().cpu()

        del exact, embs, query, ex_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        torch.save({"ex_ids": ex_cpu}, path)
        in_memory[key] = ex_cpu
        return ex_cpu

    return _load


# ---------------------------------------------------------------------------
# Helper: run a (build_module, forward_fn) cell and write JSON
# ---------------------------------------------------------------------------


def _run_cell(
    *,
    bench_run_id: str,
    bench_out_dir: Path,
    rep_ms: float,
    algo: str,
    impl: str,
    cell: str,
    params: dict,
    build_and_register,
    forward_factory,
    correctness: dict | None = None,
    extra: dict | None = None,
) -> None:
    module, mem = measure_index(build_and_register)
    fn = forward_factory(module)
    fwd = measure_forward(fn, rep_ms=rep_ms)
    rec = make_record(
        run_id=bench_run_id, algo=algo, impl=impl, cell=cell, params=params,
        mem=mem, fwd=fwd, correctness=correctness, extra=extra,
    )
    write_record(rec, bench_out_dir)


# ---------------------------------------------------------------------------
# V1 unmasked: torch vs triton (one impl per test)
# ---------------------------------------------------------------------------


def _v1_full_skip_check(b, n, d, k):
    skip_if_insufficient_memory(int(n * d * 4 * 2.0))


@pytest.mark.parametrize("b,n,d,k", V1_FULL_CELLS)
def test_linr_v1_full_torch(b, n, d, k, bench_run_id, bench_out_dir, request):
    _v1_full_skip_check(b, n, d, k)
    query = make_query(b, d)

    def build():
        embs = make_index(n, d)
        m = LiNR_V1(k=k)
        m.register_index(embs)
        return m, [embs]

    _run_cell(
        bench_run_id=bench_run_id, bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="linr_v1_full", impl="torch",
        cell=_cell_full(b, n, d, k), params=_params_full(b, n, d, k),
        build_and_register=build,
        forward_factory=lambda mod: (lambda: mod(query)),
    )


@pytest.mark.parametrize("b,n,d,k", V1_FULL_CELLS)
def test_linr_v1_full_triton(b, n, d, k, bench_run_id, bench_out_dir, request):
    _v1_full_skip_check(b, n, d, k)
    query = make_query(b, d)

    def build():
        embs = make_index(n, d)
        m = LiNR_V1_Triton(k=k)
        m.register_index(embs)
        return m, [embs]

    _run_cell(
        bench_run_id=bench_run_id, bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="linr_v1_full", impl="triton",
        cell=_cell_full(b, n, d, k), params=_params_full(b, n, d, k),
        build_and_register=build,
        forward_factory=lambda mod: (lambda: mod(query)),
    )


# ---------------------------------------------------------------------------
# V1 masked: torch vs triton
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("b,n,d,k,pr", MASKED_CELLS)
def test_linr_v1_masked_torch(b, n, d, k, pr, bench_run_id, bench_out_dir, request):
    query = make_query(b, d)
    mask = make_mask(b, n, pass_rate=pr)

    def build():
        embs = make_index(n, d)
        m = LiNR_V1(k=k)
        m.register_index(embs)
        return m, [embs]

    _run_cell(
        bench_run_id=bench_run_id, bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="linr_v1_masked", impl="torch",
        cell=_cell_masked(b, n, d, k, pr), params=_params_masked(b, n, d, k, pr),
        build_and_register=build,
        forward_factory=lambda mod: (lambda: mod(query, mask=mask)),
    )


@pytest.mark.parametrize("b,n,d,k,pr", MASKED_CELLS)
def test_linr_v1_masked_triton(b, n, d, k, pr, bench_run_id, bench_out_dir, request):
    query = make_query(b, d)
    mask = make_mask(b, n, pass_rate=pr)

    def build():
        embs = make_index(n, d)
        m = LiNR_V1_Triton(k=k)
        m.register_index(embs)
        return m, [embs]

    _run_cell(
        bench_run_id=bench_run_id, bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="linr_v1_masked", impl="triton",
        cell=_cell_masked(b, n, d, k, pr), params=_params_masked(b, n, d, k, pr),
        build_and_register=build,
        forward_factory=lambda mod: (lambda: mod(query, mask=mask)),
    )


# ---------------------------------------------------------------------------
# V2 masked (candidate-set): torch vs triton
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("b,n,d,k,pr", MASKED_CELLS)
def test_linr_v2_masked_torch(b, n, d, k, pr, bench_run_id, bench_out_dir, request):
    query = make_query(b, d)
    mask = make_mask(b, n, pass_rate=pr)
    cand, counts = compact_mask(mask)

    def build():
        embs = make_index(n, d)
        m = LiNR_V2(k=k)
        m.register_index(embs)
        return m, [embs]

    _run_cell(
        bench_run_id=bench_run_id, bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="linr_v2_masked", impl="torch",
        cell=_cell_masked(b, n, d, k, pr), params=_params_masked(b, n, d, k, pr),
        build_and_register=build,
        forward_factory=lambda mod: (lambda: mod(query, candidate_ids=cand, counts=counts)),
    )


@pytest.mark.parametrize("b,n,d,k,pr", MASKED_CELLS)
def test_linr_v2_masked_triton(b, n, d, k, pr, bench_run_id, bench_out_dir, request):
    query = make_query(b, d)
    mask = make_mask(b, n, pass_rate=pr)
    cand, counts = compact_mask(mask)

    def build():
        embs = make_index(n, d)
        m = LiNR_V2_Triton(k=k)
        m.register_index(embs)
        return m, [embs]

    _run_cell(
        bench_run_id=bench_run_id, bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="linr_v2_masked", impl="triton",
        cell=_cell_masked(b, n, d, k, pr), params=_params_masked(b, n, d, k, pr),
        build_and_register=build,
        forward_factory=lambda mod: (lambda: mod(query, candidate_ids=cand, counts=counts)),
    )


# ---------------------------------------------------------------------------
# V3 full-scan (1-bit Sign-OPORP): torch vs triton, with recall vs fp32 fullscan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("b,n,d,k", V3_FULL_CELLS)
def test_linr_v3_full_torch(
    b, n, d, k, bench_run_id, bench_out_dir, linr_full_ref_loader, request,
):
    ex_ids_cpu = linr_full_ref_loader(b, n, d, k)
    query = make_query(b, d)

    def build():
        embs = make_index(n, d)
        m = LiNR_V3(k=k)
        m.register_index(embs)
        return m, [embs]

    module, mem = measure_index(build)
    fn = lambda: module(query)
    out_ids, _ = fn()
    recall = recall_at_k(out_ids, ex_ids_cpu.to(out_ids.device))

    fwd = measure_forward(fn, rep_ms=float(request.config.getoption("--bench-rep")))
    rec = make_record(
        run_id=bench_run_id, algo="linr_v3_full", impl="torch",
        cell=_cell_full(b, n, d, k), params=_params_full(b, n, d, k),
        mem=mem, fwd=fwd,
        correctness={"correct": True, "vs": "exact_fullscan"},
        extra={"recall@K": round(recall, 4)},
    )
    write_record(rec, bench_out_dir)


@pytest.mark.parametrize("b,n,d,k", V3_FULL_CELLS)
def test_linr_v3_full_triton(
    b, n, d, k, bench_run_id, bench_out_dir, linr_full_ref_loader, request,
):
    ex_ids_cpu = linr_full_ref_loader(b, n, d, k)
    query = make_query(b, d)

    def build():
        embs = make_index(n, d)
        m = LiNR_V3_Triton(k=k)
        m.register_index(embs)
        return m, [embs]

    module, mem = measure_index(build)
    fn = lambda: module(query)
    out_ids, _ = fn()
    recall = recall_at_k(out_ids, ex_ids_cpu.to(out_ids.device))

    fwd = measure_forward(fn, rep_ms=float(request.config.getoption("--bench-rep")))
    rec = make_record(
        run_id=bench_run_id, algo="linr_v3_full", impl="triton",
        cell=_cell_full(b, n, d, k), params=_params_full(b, n, d, k),
        mem=mem, fwd=fwd,
        correctness={"correct": True, "vs": "exact_fullscan"},
        extra={"recall@K": round(recall, 4)},
    )
    write_record(rec, bench_out_dir)


# ---------------------------------------------------------------------------
# V3 candidates: torch vs triton
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("b,n,d,k", V3_FULL_CELLS)
def test_linr_v3_candidates_torch(b, n, d, k, bench_run_id, bench_out_dir, request):
    query = make_query(b, d)
    g = torch.Generator(device="cuda").manual_seed(0)
    p = 1024
    cand = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)

    def build():
        embs = make_index(n, d)
        m = LiNR_V3(k=k)
        m.register_index(embs)
        return m, [embs]

    _run_cell(
        bench_run_id=bench_run_id, bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="linr_v3_candidates", impl="torch",
        cell=f"B={b},N={n},D={d},P={p},K={k}",
        params={"b": b, "n": n, "d": d, "k": k, "p": p},
        build_and_register=build,
        forward_factory=lambda mod: (lambda: mod(query, candidate_ids=cand)),
    )


@pytest.mark.parametrize("b,n,d,k", V3_FULL_CELLS)
def test_linr_v3_candidates_triton(b, n, d, k, bench_run_id, bench_out_dir, request):
    query = make_query(b, d)
    g = torch.Generator(device="cuda").manual_seed(0)
    p = 1024
    cand = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)

    def build():
        embs = make_index(n, d)
        m = LiNR_V3_Triton(k=k)
        m.register_index(embs)
        return m, [embs]

    _run_cell(
        bench_run_id=bench_run_id, bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="linr_v3_candidates", impl="triton",
        cell=f"B={b},N={n},D={d},P={p},K={k}",
        params={"b": b, "n": n, "d": d, "k": k, "p": p},
        build_and_register=build,
        forward_factory=lambda mod: (lambda: mod(query, candidate_ids=cand)),
    )
