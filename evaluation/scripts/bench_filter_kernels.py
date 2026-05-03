"""Standalone microbench for the filter-tree Triton kernels.

Times ``bloom_compact`` vs ``compact_mask(BloomFilter.evaluate_mask(.))`` and
``clause_mask`` vs the pure-torch broadcast that ``ClauseIndex.evaluate_mask``
uses on CPU. Not wired into the Yambda end-to-end harness — this is a
kernel-level perf-and-memory check.

Run on a free GPU:

    python evaluation/scripts/bench_filter_kernels.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

from retrieve.kernels.triton.filters.bloom_compact import bloom_compact
from retrieve.kernels.triton.filters.clause_mask import clause_mask
from retrieve.kernels.triton.silvertorch.bloom_match import bloom_match
from retrieve.layers.filters import BloomFilter
from retrieve.layers.utils.compact import compact_mask

WARMUP = 5
ITERS = 50


@dataclass
class Row:
    label: str
    median_ms: float
    peak_mib: float


def _time(fn) -> tuple[float, float]:
    """Returns (median_ms, peak_alloc_delta_mib) over ITERS iterations after WARMUP."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    for _ in range(ITERS):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    peak = torch.cuda.max_memory_allocated() - base
    times.sort()
    median = times[len(times) // 2]
    return median, peak / (1024 * 1024)


def _make_attrs(
    n: int,
    c: int,
    a_max: int,
    *,
    n_vocab: int,
    pad_rate: float,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randint(0, n_vocab, (n, c, a_max), device="cuda", dtype=torch.int64, generator=g)
    pad = torch.rand((n, c, a_max), device="cuda", generator=g) < pad_rate
    return torch.where(pad, torch.full_like(raw, -1), raw)


def _make_query(
    b: int,
    c: int,
    *,
    n_vocab: int,
    inactive_rate: float,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randint(0, n_vocab, (b, c), device="cuda", dtype=torch.int64, generator=g)
    inactive = torch.rand((b, c), device="cuda", generator=g) < inactive_rate
    return torch.where(inactive, torch.full_like(raw, -1), raw)


def bench_bloom(n: int, b: int, m_bits: int, k_hash: int) -> list[Row]:
    attrs = _make_attrs(n, c=2, a_max=3, n_vocab=200, pad_rate=0.1, seed=n)
    bf = BloomFilter(m_bits=m_bits, k_hash=k_hash).to("cuda")
    bf.register_index(attrs)
    q = _make_query(b, c=2, n_vocab=200, inactive_rate=0.1, seed=m_bits + k_hash)

    qb = bf._build_query_sigs(q)

    fused_med, fused_peak = _time(lambda: bloom_compact(qb, bf.bloom_sigs))
    fallback_med, fallback_peak = _time(
        lambda: compact_mask(bloom_match(qb, bf.bloom_sigs))
    )
    label = f"bloom n={n} b={b} m={m_bits} k={k_hash}"
    return [
        Row(f"{label} :: bloom_compact", fused_med, fused_peak),
        Row(f"{label} :: compact_mask(bloom_match)", fallback_med, fallback_peak),
    ]


def _ref_mask(attrs: torch.Tensor, is_reverse: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    qe = q.unsqueeze(1).unsqueeze(-1)
    ic = attrs.unsqueeze(0)
    match = qe == ic
    clause_pass = match.any(dim=-1)
    rev = is_reverse.unsqueeze(0).unsqueeze(0)
    clause_pass = torch.where(rev, ~clause_pass, clause_pass)
    inactive = (q == -1).unsqueeze(1)
    clause_pass = clause_pass | inactive
    return clause_pass.all(dim=-1)


def bench_clause(n: int, b: int, c: int, a_max: int) -> list[Row]:
    attrs = _make_attrs(n, c=c, a_max=a_max, n_vocab=40, pad_rate=0.2, seed=n + c)
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = _make_query(b, c=c, n_vocab=40, inactive_rate=0.1, seed=a_max)

    fused_med, fused_peak = _time(lambda: clause_mask(attrs, is_reverse, q))
    fallback_med, fallback_peak = _time(lambda: _ref_mask(attrs, is_reverse, q))
    label = f"clause n={n} b={b} c={c} a={a_max}"
    return [
        Row(f"{label} :: clause_mask", fused_med, fused_peak),
        Row(f"{label} :: torch broadcast", fallback_med, fallback_peak),
    ]


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available — skipping bench.", file=sys.stderr)
        return 1

    rows: list[Row] = []
    for n in (4_096, 65_536):
        for m_bits in (256, 1024):
            rows.extend(bench_bloom(n=n, b=64, m_bits=m_bits, k_hash=4))
    for n in (4_096, 65_536):
        for a_max in (2, 4):
            rows.extend(bench_clause(n=n, b=64, c=3, a_max=a_max))

    width = max(len(r.label) for r in rows)
    print(f"{'label'.ljust(width)}  {'median_ms':>10}  {'peak_MiB':>10}")
    print("-" * (width + 24))
    for r in rows:
        print(f"{r.label.ljust(width)}  {r.median_ms:>10.3f}  {r.peak_mib:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
