"""Temporary bench for fused_masked_knn_topk before/after optimization.

Run twice — once on the optimized branch (HEAD), once on baseline:

  $ uv run python bench_fused_knn.py optimized
  $ git stash push -- retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py
  $ uv run python bench_fused_knn.py baseline
  $ git stash pop
  $ uv run python bench_fused_knn.py compare
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import triton

from retrieve.kernels.triton.linr.fused_masked_knn_topk import (
    _bucket_p,
    fused_masked_knn_topk,
)

RESULTS_DIR = Path(__file__).parent / "_bench_results"


def make_inputs(b: int, p_real: int, n: int, d: int, count_frac: float, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    embs = torch.randn(n, d, device="cuda", generator=g)
    embs = embs / embs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    query = torch.randn(b, d, device="cuda", generator=g)
    query = query / query.norm(dim=1, keepdim=True).clamp_min(1e-8)

    positive_indices = torch.full((b, p_real), -1, dtype=torch.int64, device="cuda")
    counts = torch.zeros(b, dtype=torch.int64, device="cuda")
    for bi in range(b):
        cnt = max(1, int(p_real * count_frac))
        ids = torch.randperm(n, device="cuda", generator=g)[:cnt]
        positive_indices[bi, :cnt] = ids
        counts[bi] = cnt
    return query, embs, positive_indices, counts


# (label, b, n, d, p_real, count_frac, k)
CONFIGS = [
    ("loose-bloom (P_real<<bucket)", 16, 50_000,  64,   300, 1.0,  10),
    ("loose-bloom-largeB",           64, 50_000,  64,   300, 1.0,  10),
    ("medium (P_real ~= 0.9*bucket)",16, 50_000,  64,  1800, 0.8,  10),
    ("at-bucket boundary",           16, 50_000,  64,  2048, 0.5,  10),
    ("d=128 medium",                  8, 30_000, 128,  1500, 0.9,  50),
    ("large P_real",                  4,200_000,  64, 12000, 0.7, 100),
    ("huge P_real",                   2,500_000,  64,100000, 0.5, 200),
]


def bench(label: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    print(f"=== {label} ===")
    print(
        f"{'config':<34} {'p_bucket':>9} {'k':>5} "
        f"{'p50_us':>10} {'p20_us':>10} {'p80_us':>10} "
        f"{'peak_MB':>10}"
    )
    print("-" * 100)
    for name, b, n, d, p_real, frac, k in CONFIGS:
        query, embs, pos, counts = make_inputs(b, p_real, n, d, frac)
        # warm + autotune (first call triggers Triton autotune sweep)
        for _ in range(3):
            fused_masked_knn_topk(query, embs, pos, counts, k)
        torch.cuda.synchronize()

        # 1. Timing — do_bench uses a ~256MB L2-flush buffer, so don't read
        # peak memory inside this block.
        def fn():
            fused_masked_knn_topk(query, embs, pos, counts, k)

        ms = triton.testing.do_bench(fn, warmup=50, rep=200, quantiles=[0.2, 0.5, 0.8])

        # 2. Per-call transient memory — fresh peak counter, single call.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_mem = torch.cuda.memory_allocated()
        fused_masked_knn_topk(query, embs, pos, counts, k)
        torch.cuda.synchronize()
        peak_mb = (torch.cuda.max_memory_allocated() - baseline_mem) / (1024 * 1024)
        p_bucket = _bucket_p(p_real)
        p20_us, p50_us, p80_us = ms[0] * 1e3, ms[1] * 1e3, ms[2] * 1e3
        print(
            f"{name:<34} {p_bucket:>9} {k:>5} "
            f"{p50_us:>10.2f} {p20_us:>10.2f} {p80_us:>10.2f} "
            f"{peak_mb:>10.2f}"
        )
        rows.append(
            {
                "name": name, "b": b, "n": n, "d": d,
                "p_real": p_real, "p_bucket": p_bucket, "k": k,
                "p50_us": p50_us, "p20_us": p20_us, "p80_us": p80_us,
                "peak_MB": peak_mb,
            }
        )
    out = RESULTS_DIR / f"{label}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


def compare() -> None:
    base = json.loads((RESULTS_DIR / "baseline.json").read_text())
    opt = json.loads((RESULTS_DIR / "optimized.json").read_text())
    print(
        f"{'config':<34} {'p50 base':>10} {'p50 opt':>10} {'speedup':>9} "
        f"{'mem base':>10} {'mem opt':>10} {'mem ratio':>10}"
    )
    print("-" * 100)
    for b, o in zip(base, opt):
        assert b["name"] == o["name"]
        speedup = b["p50_us"] / o["p50_us"]
        mem_ratio = o["peak_MB"] / b["peak_MB"] if b["peak_MB"] > 0 else float("nan")
        print(
            f"{b['name']:<34} {b['p50_us']:>10.2f} {o['p50_us']:>10.2f} "
            f"{speedup:>8.2f}x {b['peak_MB']:>10.2f} {o['peak_MB']:>10.2f} "
            f"{mem_ratio:>9.2f}x"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("baseline", "optimized", "compare"):
        print("usage: bench_fused_knn.py {baseline|optimized|compare}")
        sys.exit(2)
    if sys.argv[1] == "compare":
        compare()
    else:
        bench(sys.argv[1])
