"""Per-kernel microbenchmarks: Triton vs torch baseline.

Each (kernel, impl) is its own pytest function. ``measure_index`` here treats
the persistent kernel inputs (embs, codes, scales, bloom sigs, etc.) as the
"index" — there's no nn.Module to register. ``measure_forward`` captures the
transient peak of the kernel call itself.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.triton.linr.fused_masked_knn_topk import fused_masked_knn_topk
from retrieve.kernels.triton.linr.fused_matmul_topk import fused_matmul_topk
from retrieve.kernels.triton.linr.oporp_1bit_match_topk import oporp_1bit_match_topk
from retrieve.kernels.triton.silvertorch.bloom_match import bloom_match
from retrieve.kernels.triton.silvertorch.codesigned_probe_score import (
    codesigned_probe_score,
)
from retrieve.kernels.triton.silvertorch.int8_ann_fused import int8_ann_fused
from retrieve.layers.silvertorch.bloom import BloomIndex, _build_signatures, _generate_seeds
from retrieve.layers.utils.compact import compact_mask
from retrieve.layers.utils.quantize import (
    project_oporp_1bit_query,
    quantize_int8,
    quantize_oporp_1bit,
)
from tests.bench.conftest import (
    make_record,
    measure_forward,
    measure_index,
    write_record,
)
from tests.conftest import make_attrs, make_index, make_mask, make_query, make_query_attrs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_kernel_cell(
    *,
    bench_run_id: str,
    bench_out_dir,
    rep_ms: float,
    algo: str,
    impl: str,
    cell: str,
    params: dict,
    build_inputs,
    forward_factory,
) -> None:
    """Like _run_cell in bench_linr but the "module" is just the input tensors."""
    inputs, mem = measure_index(build_inputs)
    fn = forward_factory(inputs)
    fwd = measure_forward(fn, rep_ms=rep_ms)
    rec = make_record(
        run_id=bench_run_id,
        algo=algo,
        impl=impl,
        cell=cell,
        params=params,
        mem=mem,
        fwd=fwd,
    )
    write_record(rec, bench_out_dir)


# ---------------------------------------------------------------------------
# fused_matmul_topk
# ---------------------------------------------------------------------------

_MATMUL_CELLS = [
    pytest.param(b, n, d, k, id=f"B{b}_N{n}_D{d}_K{k}")
    for b, n, d, k in [(1, 4096, 128, 16), (16, 65_536, 128, 200)]
]


def _matmul_inputs(b, n, d):
    def _build():
        embs = make_index(n, d)
        query = make_query(b, d)
        # Both are persistent kernel inputs; nothing to free.
        return (embs, query), []

    return _build


@pytest.mark.parametrize("b,n,d,k", _MATMUL_CELLS)
def test_kern_fused_matmul_topk_torch(b, n, d, k, bench_run_id, bench_out_dir, request):
    _run_kernel_cell(
        bench_run_id=bench_run_id,
        bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="fused_matmul_topk",
        impl="torch",
        cell=f"B={b},N={n},D={d},K={k}",
        params={"b": b, "n": n, "d": d, "k": k},
        build_inputs=_matmul_inputs(b, n, d),
        forward_factory=lambda inputs: lambda: torch.topk(inputs[1] @ inputs[0].t(), k, dim=1),
    )


@pytest.mark.parametrize("b,n,d,k", _MATMUL_CELLS)
def test_kern_fused_matmul_topk_triton(b, n, d, k, bench_run_id, bench_out_dir, request):
    _run_kernel_cell(
        bench_run_id=bench_run_id,
        bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="fused_matmul_topk",
        impl="triton",
        cell=f"B={b},N={n},D={d},K={k}",
        params={"b": b, "n": n, "d": d, "k": k},
        build_inputs=_matmul_inputs(b, n, d),
        forward_factory=lambda inputs: lambda: fused_matmul_topk(inputs[1], inputs[0], k),
    )


# ---------------------------------------------------------------------------
# fused_masked_knn_topk
# ---------------------------------------------------------------------------

_MASKED_CELLS = [
    pytest.param(b, n, d, k, pr, id=f"B{b}_N{n}_D{d}_K{k}_pr{pr}")
    for b, n, d, k in [(16, 65_536, 128, 200)]
    for pr in [0.05, 0.5]
]


def _masked_inputs(b, n, d, pr):
    def _build():
        embs = make_index(n, d)
        query = make_query(b, d)
        mask = make_mask(b, n, pass_rate=pr)
        pos, counts = compact_mask(mask)
        return (embs, query, mask, pos, counts), []

    return _build


@pytest.mark.parametrize("b,n,d,k,pr", _MASKED_CELLS)
def test_kern_fused_masked_knn_topk_torch(b, n, d, k, pr, bench_run_id, bench_out_dir, request):
    def fwd_factory(inputs):
        embs, query, mask, _, _ = inputs

        def fn():
            scores = query @ embs.t()
            scores = scores.masked_fill(~mask, float("-inf"))
            return torch.topk(scores, k, dim=1)

        return fn

    _run_kernel_cell(
        bench_run_id=bench_run_id,
        bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="fused_masked_knn_topk",
        impl="torch",
        cell=f"B={b},N={n},D={d},K={k},pass={pr}",
        params={"b": b, "n": n, "d": d, "k": k, "pass_rate": pr},
        build_inputs=_masked_inputs(b, n, d, pr),
        forward_factory=fwd_factory,
    )


@pytest.mark.parametrize("b,n,d,k,pr", _MASKED_CELLS)
def test_kern_fused_masked_knn_topk_triton(b, n, d, k, pr, bench_run_id, bench_out_dir, request):
    def fwd_factory(inputs):
        embs, query, _, pos, counts = inputs
        return lambda: fused_masked_knn_topk(query, embs, pos, counts, k)

    _run_kernel_cell(
        bench_run_id=bench_run_id,
        bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="fused_masked_knn_topk",
        impl="triton",
        cell=f"B={b},N={n},D={d},K={k},pass={pr}",
        params={"b": b, "n": n, "d": d, "k": k, "pass_rate": pr},
        build_inputs=_masked_inputs(b, n, d, pr),
        forward_factory=fwd_factory,
    )


# ---------------------------------------------------------------------------
# bloom_match
# ---------------------------------------------------------------------------

_BLOOM_CELLS = [
    pytest.param(b, n, mb, id=f"B{b}_N{n}_m{mb}") for b, n in [(64, 65_536)] for mb in [512, 1024]
]


def _bloom_inputs(b, n, mb):
    def _build():
        attrs = make_attrs(n, c=2, a_max=3, n_vocab=200, pad_rate=0.1)
        bi = BloomIndex().to("cuda")
        bi.register_index(attrs, m_bits=mb, k_hash=5)
        q = make_query_attrs(b, c=2, n_vocab=200, inactive_rate=0.0)
        qb_sigs = _build_signatures(
            q.long().unsqueeze(-1),
            bi.hash_seeds,
            bi.m_bits,
            bi.k_hash,
            bi.word_count,
        )
        # bi keeps its own bloom_sigs; attrs/q can be freed.
        return (bi, qb_sigs), [attrs, q]

    return _build


@pytest.mark.parametrize("b,n,mb", _BLOOM_CELLS)
def test_kern_bloom_match_torch(b, n, mb, bench_run_id, bench_out_dir, request):
    def fwd_factory(inputs):
        bi, qb_sigs = inputs

        def fn():
            match = (qb_sigs.unsqueeze(1) & bi.bloom_sigs.unsqueeze(0)) == qb_sigs.unsqueeze(1)
            return match.all(dim=-1)

        return fn

    _run_kernel_cell(
        bench_run_id=bench_run_id,
        bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="bloom_match",
        impl="torch",
        cell=f"B={b},N={n},m_bits={mb}",
        params={"b": b, "n": n, "m_bits": mb},
        build_inputs=_bloom_inputs(b, n, mb),
        forward_factory=fwd_factory,
    )


@pytest.mark.parametrize("b,n,mb", _BLOOM_CELLS)
def test_kern_bloom_match_triton(b, n, mb, bench_run_id, bench_out_dir, request):
    def fwd_factory(inputs):
        bi, qb_sigs = inputs
        return lambda: bloom_match(qb_sigs, bi.bloom_sigs)

    _run_kernel_cell(
        bench_run_id=bench_run_id,
        bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="bloom_match",
        impl="triton",
        cell=f"B={b},N={n},m_bits={mb}",
        params={"b": b, "n": n, "m_bits": mb},
        build_inputs=_bloom_inputs(b, n, mb),
        forward_factory=fwd_factory,
    )


# ---------------------------------------------------------------------------
# int8_ann_fused
# ---------------------------------------------------------------------------

_INT8_CELLS = [pytest.param(16, 65_536, 128, 1024, 200, id="B16_N65536_D128_P1024_K200")]


def _int8_inputs(b, n, d, p):
    def _build():
        embs = make_index(n, d)
        codes, scales = quantize_int8(embs)
        query = make_query(b, d)
        g = torch.Generator(device="cuda").manual_seed(0)
        pos = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)
        counts = torch.full((b,), p, dtype=torch.long, device="cuda")
        return (codes, scales, query, pos, counts), [embs]

    return _build


@pytest.mark.parametrize("b,n,d,p,k", _INT8_CELLS)
def test_kern_int8_ann_fused_torch(b, n, d, p, k, bench_run_id, bench_out_dir, request):
    def fwd_factory(inputs):
        codes, scales, query, pos, _ = inputs

        def fn():
            cand_codes = codes[pos].float()
            cand_scales = scales[pos]
            scores = torch.einsum("bd,bpd->bp", query, cand_codes) * cand_scales
            return torch.topk(scores, k, dim=1)

        return fn

    _run_kernel_cell(
        bench_run_id=bench_run_id,
        bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="int8_ann_fused",
        impl="torch",
        cell=f"B={b},N={n},D={d},P={p},K={k}",
        params={"b": b, "n": n, "d": d, "p": p, "k": k},
        build_inputs=_int8_inputs(b, n, d, p),
        forward_factory=fwd_factory,
    )


@pytest.mark.parametrize("b,n,d,p,k", _INT8_CELLS)
def test_kern_int8_ann_fused_triton(b, n, d, p, k, bench_run_id, bench_out_dir, request):
    def fwd_factory(inputs):
        codes, scales, query, pos, counts = inputs
        return lambda: int8_ann_fused(query, codes, scales, pos, counts, k)

    _run_kernel_cell(
        bench_run_id=bench_run_id,
        bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="int8_ann_fused",
        impl="triton",
        cell=f"B={b},N={n},D={d},P={p},K={k}",
        params={"b": b, "n": n, "d": d, "p": p, "k": k},
        build_inputs=_int8_inputs(b, n, d, p),
        forward_factory=fwd_factory,
    )


# ---------------------------------------------------------------------------
# oporp_1bit_match_topk
# ---------------------------------------------------------------------------

_OPORP_CELLS = [
    pytest.param(b, n, d, k, id=f"B{b}_N{n}_D{d}_K{k}")
    for b, n, d, k in [(1, 65_536, 128, 200), (16, 65_536, 128, 200)]
]


def _oporp_inputs(b, n, d):
    def _build():
        embs = make_index(n, d)
        query = make_query(b, d)
        item_bits, signs, perm = quantize_oporp_1bit(embs, seed=0)
        query_bits = project_oporp_1bit_query(query, signs, perm)
        return (item_bits, query_bits), [embs, query, signs, perm]

    return _build


@pytest.mark.parametrize("b,n,d,k", _OPORP_CELLS)
def test_kern_oporp_1bit_torch(b, n, d, k, bench_run_id, bench_out_dir, request):
    def fwd_factory(inputs):
        item_bits, query_bits = inputs
        d_total = 64 * item_bits.shape[1]

        def fn():
            xor = query_bits.unsqueeze(1) ^ item_bits.unsqueeze(0)
            from retrieve.layers.utils.quantize import popcount_int64

            hamming = popcount_int64(xor).sum(dim=-1)
            scores = (d_total - 2 * hamming).to(torch.float32)
            return torch.topk(scores, k, dim=1)

        return fn

    _run_kernel_cell(
        bench_run_id=bench_run_id,
        bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="oporp_1bit_match_topk",
        impl="torch",
        cell=f"B={b},N={n},D={d},K={k}",
        params={"b": b, "n": n, "d": d, "k": k},
        build_inputs=_oporp_inputs(b, n, d),
        forward_factory=fwd_factory,
    )


@pytest.mark.parametrize("b,n,d,k", _OPORP_CELLS)
def test_kern_oporp_1bit_triton(b, n, d, k, bench_run_id, bench_out_dir, request):
    def fwd_factory(inputs):
        item_bits, query_bits = inputs
        return lambda: oporp_1bit_match_topk(query_bits, item_bits, k)

    _run_kernel_cell(
        bench_run_id=bench_run_id,
        bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="oporp_1bit_match_topk",
        impl="triton",
        cell=f"B={b},N={n},D={d},K={k}",
        params={"b": b, "n": n, "d": d, "k": k},
        build_inputs=_oporp_inputs(b, n, d),
        forward_factory=fwd_factory,
    )


# ---------------------------------------------------------------------------
# codesigned_probe_score
# ---------------------------------------------------------------------------

_CODESIGNED_CELLS = [pytest.param(16, 65_536, 128, 1024, 200, id="B16_N65536_D128_P1024_K200")]


def _codesigned_inputs(b, n, d, p):
    def _build():
        embs = make_index(n, d)
        codes, scales = quantize_int8(embs)
        query = make_query(b, d)
        attrs = make_attrs(n, c=2, a_max=2)
        q_attrs = make_query_attrs(b, c=2)
        seeds = _generate_seeds(k_hash=5, device=embs.device)
        sigs = _build_signatures(attrs.long(), seeds, m_bits=512, k_hash=5, word_count=8)
        qb = _build_signatures(
            q_attrs.long().unsqueeze(-1), seeds, m_bits=512, k_hash=5, word_count=8
        )
        g = torch.Generator(device="cuda").manual_seed(11)
        flat = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)
        pad = torch.rand(b, p, generator=g, device="cuda") < 0.05
        flat[pad] = -1
        return (codes, scales, query, qb, sigs, flat), [embs, attrs, q_attrs, seeds]

    return _build


@pytest.mark.parametrize("b,n,d,p,k", _CODESIGNED_CELLS)
def test_kern_codesigned_probe_torch(b, n, d, p, k, bench_run_id, bench_out_dir, request):
    def fwd_factory(inputs):
        codes, scales, query, qb, sigs, flat = inputs

        def fn():
            valid = flat >= 0
            safe = flat.clamp(min=0)
            probed_sigs = sigs[safe]
            match = (qb.unsqueeze(1) & probed_sigs) == qb.unsqueeze(1)
            keep = valid & match.all(dim=-1)
            cand_codes = codes[safe].float()
            cand_scales = scales[safe]
            scores = torch.einsum("bd,bpd->bp", query, cand_codes) * cand_scales
            scores = scores.masked_fill(~keep, float("-inf"))
            actual_k = min(k, scores.shape[1])
            topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
            topk_ids = flat.gather(1, topk_local)
            return topk_ids, topk_scores

        return fn

    _run_kernel_cell(
        bench_run_id=bench_run_id,
        bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="codesigned_probe_score",
        impl="torch",
        cell=f"B={b},N={n},D={d},P={p},K={k}",
        params={"b": b, "n": n, "d": d, "p": p, "k": k},
        build_inputs=_codesigned_inputs(b, n, d, p),
        forward_factory=fwd_factory,
    )


@pytest.mark.parametrize("b,n,d,p,k", _CODESIGNED_CELLS)
def test_kern_codesigned_probe_triton(b, n, d, p, k, bench_run_id, bench_out_dir, request):
    def fwd_factory(inputs):
        codes, scales, query, qb, sigs, flat = inputs
        return lambda: codesigned_probe_score(
            query, flat, codes, scales, k, query_bits=qb, bloom_sigs=sigs
        )

    _run_kernel_cell(
        bench_run_id=bench_run_id,
        bench_out_dir=bench_out_dir,
        rep_ms=float(request.config.getoption("--bench-rep")),
        algo="codesigned_probe_score",
        impl="triton",
        cell=f"B={b},N={n},D={d},P={p},K={k}",
        params={"b": b, "n": n, "d": d, "p": p, "k": k},
        build_inputs=_codesigned_inputs(b, n, d, p),
        forward_factory=fwd_factory,
    )
