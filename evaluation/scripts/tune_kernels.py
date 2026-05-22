"""Offline tuner for the Triton kernels in ``retrieve.kernels.triton``.

Run once on the target arch; paste the printed ``DEFAULT_CONFIG = ...``
line into the corresponding kernel file. The library ships a single
default per kernel — there is no REGISTRY of pre-baked configs.
Callers who want a non-default tile pass ``config=`` to the wrapper
(or to the ``_<name>_impl`` private function for ``@custom_op`` kernels).

Usage:

    uv run tune-kernels --kernel fused_masked_knn_topk
    uv run tune-kernels --kernel oporp_1bit_match_topk
    uv run tune-kernels --kernel codesigned_probe_score
    uv run tune-kernels --kernel clause_mask
    uv run tune-kernels --kernel clause_compact
    uv run tune-kernels --kernel bloom_compact

Optionally dump full per-shape sweep results to JSON via ``--json-out``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import click
import torch
import triton.testing as ttesting

from retrieve.kernels.triton.filters.bloom_compact import (
    BloomCompactConfig,
    _bloom_compact_impl,
)
from retrieve.kernels.triton.filters.clause_compact import (
    ClauseCompactConfig,
    _clause_compact_impl,
)
from retrieve.kernels.triton.filters.clause_mask import (
    ClauseMaskConfig,
    _clause_mask_impl,
)
from retrieve.kernels.triton.linr.fused_masked_knn_topk import (
    _P_BUCKETS,
    FusedMaskedKnnTopkConfig,
    _fused_masked_knn_topk_impl,
)
from retrieve.kernels.triton.linr.oporp_1bit_match_topk import (
    _N_BUCKETS,
    Oporp1BitMatchTopkConfig,
    oporp_1bit_match_topk,
)
from retrieve.kernels.triton.silvertorch.codesigned_probe_score import (
    CodesignedProbeScoreConfig,
    codesigned_probe_score,
)


# Mirror the autotune grids that lived in the kernel files before this
# stage. Same product as the deleted ``_autotune_configs()`` helpers.
_FMKT_GRID = [(bn, nw) for bn in (32, 64, 128, 256) for nw in (4, 8)]
_OPORP_GRID = [(bn, nw) for bn in (64, 128, 256, 512) for nw in (4, 8)]
_CPS_GRID = [(bp, nw) for bp in (32, 64, 128, 256) for nw in (4, 8)]
# Filter-index kernels share the same loop shape (grid over [B, cdiv(N,
# BLOCK_N)]); sweep a wider BLOCK_N range since the inner body varies
# (clause loop vs bloom subset-test) and the optimum can land far from 256.
_CLAUSE_MASK_GRID = [(bn, nw) for bn in (128, 256, 512, 1024) for nw in (2, 4, 8)]
_CLAUSE_COMPACT_GRID = [(bn, nw) for bn in (128, 256, 512, 1024) for nw in (2, 4, 8)]
_BLOOM_COMPACT_GRID = [(bn, nw) for bn in (128, 256, 512, 1024) for nw in (2, 4, 8)]

# Regime sweeps for the filter kernels. Mirror the real-eval shapes
# from ``evaluation/data/{goodreads-work-id,arxiv-retrieval}/item_attrs_narrow.pt``
# (N≈800K / 3M, C=4 / 5, A_MAX=4) and ``evaluation/retrieval/config.py``
# ``batch_sizes=[1, 8, 16]`` (we test both B extremes). Bloom uses W=16
# from the default ``m_bits=1024`` in
# ``evaluation/retrieval/algos/filter.py:build_filter``. The 15M arxiv-
# synth catalog is dropped — same kernel pattern as 3M arXiv at much
# higher tuner cost; spot-check separately if it ever diverges.
_CLAUSE_REGIMES = (
    # (N, B, C, A_MAX)
    (797_085,    1, 4, 4),  # goodreads, single-query
    (797_085,   16, 4, 4),  # goodreads, batched
    (2_988_997,  1, 5, 4),  # arxiv-retrieval, single-query
    (2_988_997, 16, 5, 4),  # arxiv-retrieval, batched
    (15_000_001, 16, 5, 4),  # arxiv-synth-15m, batched (DRAM-streaming extreme)
)
_BLOOM_REGIMES = (
    # (N, B, W) — W=16 from m_bits=1024 default
    (797_085,    1, 16),
    (797_085,   16, 16),
    (2_988_997,  1, 16),
    (2_988_997, 16, 16),
    (15_000_001, 16, 16),  # arxiv-synth-15m, batched (DRAM-streaming extreme)
)


def _arch_str(device: torch.device) -> str:
    cap = torch.cuda.get_device_capability(device)
    return f"sm_{cap[0]}{cap[1]}"


def _bench(fn) -> float:
    """Median ms for ``fn`` under do_bench. Long enough to settle the
    JIT cache; short enough to keep the full sweep under a few minutes.
    """
    median, _, _ = ttesting.do_bench(fn, quantiles=[0.5, 0.2, 0.8], rep=500, warmup=100)
    return float(median)


# ---------------------------------------------------------------------
# fused_masked_knn_topk


def _tune_fmkt(dev: torch.device) -> dict:
    """For each bucket, pick the (block_n, num_warps) with the lowest
    median latency. Aggregate the per-bucket winners into a single
    DEFAULT_CONFIG via plurality vote (ties broken by lower num_warps).
    """
    d = 128
    b = 16

    per_bucket: dict[int, dict] = {}
    for p in _P_BUCKETS:
        # Need N >= P; pick generous N so the gather is realistic.
        n = max(p * 2, 1 << 16)
        torch.manual_seed(0)
        query = torch.randn(b, d, device=dev)
        embs = torch.randn(n, d, device=dev)
        # Fill positive_indices with random valid ids; counts at full p.
        pos = torch.randint(0, n, (b, p), dtype=torch.long, device=dev)
        counts = torch.full((b,), p, dtype=torch.long, device=dev)
        k = min(64, p)

        results: list[dict] = []
        best: tuple[int, int] | None = None
        best_ms = float("inf")
        for block_n, num_warps in _FMKT_GRID:
            cfg = FusedMaskedKnnTopkConfig(block_n=block_n, num_warps=num_warps)
            # Warm the JIT cache for this (P_bucket, BLOCK_N, num_warps)
            # before measuring; do_bench's own warmup is generous, but a
            # cold compile would still leak into the first measured rep.
            for _ in range(3):
                _fused_masked_knn_topk_impl(query, embs, pos, counts, k, config=cfg)
            torch.cuda.synchronize()
            ms = _bench(
                lambda c=cfg: _fused_masked_knn_topk_impl(query, embs, pos, counts, k, config=c)
            )
            results.append({"block_n": block_n, "num_warps": num_warps, "ms": ms})
            if ms < best_ms:
                best_ms = ms
                best = (block_n, num_warps)
            click.echo(
                f"  [P={p:>7}] block_n={block_n:<4} num_warps={num_warps}  "
                f"-> {ms:.3f} ms",
                err=True,
            )
        assert best is not None
        per_bucket[p] = {"winner": best, "winner_ms": best_ms, "all": results}
        click.echo(
            f"  [P={p:>7}] winner: block_n={best[0]} num_warps={best[1]} "
            f"({best_ms:.3f} ms)",
            err=True,
        )

    # Aggregate to a single default: plurality among per-bucket winners.
    votes = Counter(per_bucket[p]["winner"] for p in _P_BUCKETS)
    # Ties: prefer lower num_warps (cheaper register pressure).
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1]))
    default = top[0][0]
    return {"per_bucket": per_bucket, "default": default}


def _print_fmkt(arch: str, result: dict) -> None:
    bn, nw = result["default"]
    click.echo("")
    click.echo(
        "# Paste into retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py"
    )
    click.echo(f"# Tuned on {arch}; per-bucket details in JSON output if --json-out was used.")
    click.echo(
        f"DEFAULT_CONFIG = FusedMaskedKnnTopkConfig(block_n={bn}, num_warps={nw})"
    )


# ---------------------------------------------------------------------
# oporp_1bit_match_topk


def _tune_oporp(dev: torch.device) -> dict:
    """Sweep across (n_bucket × has_indices) regimes, pick winning tile
    per regime, aggregate to a single DEFAULT_CONFIG.
    """
    w = 2  # D = 128 bits packed
    b = 16

    per_regime: dict[str, dict] = {}
    for n_bucket in _N_BUCKETS:
        for has_indices in (False, True):
            key = f"n={n_bucket},has_indices={has_indices}"
            torch.manual_seed(0)
            corpus_n = n_bucket if not has_indices else max(n_bucket, 1 << 16)
            query_bits = torch.randint(
                torch.iinfo(torch.int64).min,
                torch.iinfo(torch.int64).max,
                (b, w),
                dtype=torch.int64,
                device=dev,
            )
            item_bits = torch.randint(
                torch.iinfo(torch.int64).min,
                torch.iinfo(torch.int64).max,
                (corpus_n, w),
                dtype=torch.int64,
                device=dev,
            )
            if has_indices:
                p = n_bucket // 4 if n_bucket >= 4 else 1
                pos = torch.randint(0, corpus_n, (b, p), dtype=torch.long, device=dev)
                counts = torch.full((b,), p, dtype=torch.long, device=dev)
                k = min(64, p)
            else:
                pos = counts = None
                k = 64

            results: list[dict] = []
            best: tuple[int, int] | None = None
            best_ms = float("inf")
            for block_n, num_warps in _OPORP_GRID:
                cfg = Oporp1BitMatchTopkConfig(block_n=block_n, num_warps=num_warps)
                for _ in range(3):
                    oporp_1bit_match_topk(
                        query_bits,
                        item_bits,
                        k,
                        positive_indices=pos,
                        counts=counts,
                        config=cfg,
                    )
                torch.cuda.synchronize()
                ms = _bench(
                    lambda c=cfg: oporp_1bit_match_topk(
                        query_bits,
                        item_bits,
                        k,
                        positive_indices=pos,
                        counts=counts,
                        config=c,
                    )
                )
                results.append({"block_n": block_n, "num_warps": num_warps, "ms": ms})
                if ms < best_ms:
                    best_ms = ms
                    best = (block_n, num_warps)
                click.echo(
                    f"  [{key}] block_n={block_n:<4} num_warps={num_warps}  "
                    f"-> {ms:.3f} ms",
                    err=True,
                )
            assert best is not None
            per_regime[key] = {"winner": best, "winner_ms": best_ms, "all": results}
            click.echo(
                f"  [{key}] winner: block_n={best[0]} num_warps={best[1]} "
                f"({best_ms:.3f} ms)",
                err=True,
            )

    votes = Counter(per_regime[k]["winner"] for k in per_regime)
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1]))
    default = top[0][0]
    return {"per_regime": per_regime, "default": default}


def _print_oporp(arch: str, result: dict) -> None:
    bn, nw = result["default"]
    click.echo("")
    click.echo(
        "# Paste into retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py"
    )
    click.echo(f"# Tuned on {arch}; per-regime details in JSON output if --json-out was used.")
    click.echo(
        f"DEFAULT_CONFIG = Oporp1BitMatchTopkConfig(block_n={bn}, num_warps={nw})"
    )


# ---------------------------------------------------------------------
# codesigned_probe_score (silvertorch)


def _tune_cps(dev: torch.device) -> dict:
    """SilverTorch's phase-2+3 fused kernel. ``P = n_probe * max_cluster_size``
    is fixed per registered index, so no bucketing — sweep a few
    representative P values that span typical IVF sizings (small, medium,
    large catalog × small/large probes), plus has_qb on/off.
    """
    d = 128
    b = 16
    w = 4  # 256-bit bloom signatures

    per_regime: dict[str, dict] = {}
    p_grid = (1024, 8192, 65536)
    for p in p_grid:
        n = max(p * 4, 1 << 16)
        for has_qb in (False, True):
            key = f"P={p},has_qb={has_qb}"
            torch.manual_seed(0)
            query = torch.randn(b, d, device=dev)
            flat_items = torch.randint(0, n, (b, p), dtype=torch.long, device=dev)
            item_codes = torch.randint(-128, 128, (n, d), dtype=torch.int8, device=dev)
            global_scale = 0.01
            if has_qb:
                qb = torch.randint(
                    torch.iinfo(torch.int64).min,
                    torch.iinfo(torch.int64).max,
                    (b, w),
                    dtype=torch.int64,
                    device=dev,
                )
                sigs = torch.randint(
                    torch.iinfo(torch.int64).min,
                    torch.iinfo(torch.int64).max,
                    (n, w),
                    dtype=torch.int64,
                    device=dev,
                )
            else:
                qb = sigs = None
            k = min(64, p)

            results: list[dict] = []
            best: tuple[int, int] | None = None
            best_ms = float("inf")
            for block_p, num_warps in _CPS_GRID:
                cfg = CodesignedProbeScoreConfig(block_p=block_p, num_warps=num_warps)
                for _ in range(3):
                    codesigned_probe_score(
                        query,
                        flat_items,
                        item_codes,
                        global_scale,
                        k,
                        query_bits=qb,
                        bloom_sigs=sigs,
                        config=cfg,
                    )
                torch.cuda.synchronize()
                ms = _bench(
                    lambda c=cfg: codesigned_probe_score(
                        query,
                        flat_items,
                        item_codes,
                        global_scale,
                        k,
                        query_bits=qb,
                        bloom_sigs=sigs,
                        config=c,
                    )
                )
                results.append({"block_p": block_p, "num_warps": num_warps, "ms": ms})
                if ms < best_ms:
                    best_ms = ms
                    best = (block_p, num_warps)
                click.echo(
                    f"  [{key}] block_p={block_p:<4} num_warps={num_warps}  "
                    f"-> {ms:.3f} ms",
                    err=True,
                )
            assert best is not None
            per_regime[key] = {"winner": best, "winner_ms": best_ms, "all": results}
            click.echo(
                f"  [{key}] winner: block_p={best[0]} num_warps={best[1]} "
                f"({best_ms:.3f} ms)",
                err=True,
            )

    votes = Counter(per_regime[k]["winner"] for k in per_regime)
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1]))
    default = top[0][0]
    return {"per_regime": per_regime, "default": default}


def _print_cps(arch: str, result: dict) -> None:
    bp, nw = result["default"]
    click.echo("")
    click.echo(
        "# Paste into "
        "retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py"
    )
    click.echo(f"# Tuned on {arch}; per-regime details in JSON output if --json-out was used.")
    click.echo(
        f"DEFAULT_CONFIG = CodesignedProbeScoreConfig(block_p={bp}, num_warps={nw})"
    )


# ---------------------------------------------------------------------
# clause_mask


def _make_clause_inputs(
    dev: torch.device, n: int, b: int, c: int, a_max: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build realistic (item_attrs, is_reverse, query_attrs) on ``dev``.

    Item attribute vocab is small enough that random queries get a
    non-trivial mix of pass/fail per row — mimics real eval shapes.
    """
    torch.manual_seed(0)
    n_vocab = 40
    item_attrs = torch.randint(0, n_vocab, (n, c, a_max), dtype=torch.int64, device=dev)
    is_reverse = torch.zeros(c, dtype=torch.bool, device=dev)
    query_attrs = torch.randint(0, n_vocab, (b, c), dtype=torch.int64, device=dev)
    return item_attrs, is_reverse, query_attrs


def _tune_clause_mask(dev: torch.device) -> dict:
    """Sweep (N, B, C, A_MAX) regimes; aggregate per-regime winners to a
    single DEFAULT_CONFIG via plurality vote (ties → lower num_warps)."""
    per_regime: dict[str, dict] = {}
    for n, b, c, a_max in _CLAUSE_REGIMES:
        key = f"N={n},B={b},C={c},A_MAX={a_max}"
        item_attrs, is_reverse, query_attrs = _make_clause_inputs(dev, n, b, c, a_max)

        results: list[dict] = []
        best: tuple[int, int] | None = None
        best_ms = float("inf")
        for block_n, num_warps in _CLAUSE_MASK_GRID:
            cfg = ClauseMaskConfig(block_n=block_n, num_warps=num_warps)
            for _ in range(3):
                _clause_mask_impl(item_attrs, is_reverse, query_attrs, config=cfg)
            torch.cuda.synchronize()
            ms = _bench(
                lambda c=cfg: _clause_mask_impl(
                    item_attrs, is_reverse, query_attrs, config=c
                )
            )
            results.append({"block_n": block_n, "num_warps": num_warps, "ms": ms})
            if ms < best_ms:
                best_ms = ms
                best = (block_n, num_warps)
            click.echo(
                f"  [{key}] block_n={block_n:<4} num_warps={num_warps}  "
                f"-> {ms:.3f} ms",
                err=True,
            )
        assert best is not None
        per_regime[key] = {"winner": best, "winner_ms": best_ms, "all": results}
        click.echo(
            f"  [{key}] winner: block_n={best[0]} num_warps={best[1]} "
            f"({best_ms:.3f} ms)",
            err=True,
        )

    votes = Counter(per_regime[k]["winner"] for k in per_regime)
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1]))
    default = top[0][0]
    return {"per_regime": per_regime, "default": default}


def _print_clause_mask(arch: str, result: dict) -> None:
    bn, nw = result["default"]
    click.echo("")
    click.echo(
        "# Paste into retrieve/src/retrieve/kernels/triton/filters/clause_mask.py"
    )
    click.echo(f"# Tuned on {arch}; per-regime details in JSON output if --json-out was used.")
    click.echo(f"DEFAULT_CONFIG = ClauseMaskConfig(block_n={bn}, num_warps={nw})")


# ---------------------------------------------------------------------
# clause_compact


def _tune_clause_compact(dev: torch.device) -> dict:
    """Same regime grid as clause_mask; calls ``_clause_compact_impl`` which
    allocates fresh out_indices (-1 filled) and counts (zeros) per call —
    the atomic_add accumulation worry in the kernel doc does NOT apply to
    the offline tuner."""
    per_regime: dict[str, dict] = {}
    for n, b, c, a_max in _CLAUSE_REGIMES:
        key = f"N={n},B={b},C={c},A_MAX={a_max}"
        item_attrs, is_reverse, query_attrs = _make_clause_inputs(dev, n, b, c, a_max)

        results: list[dict] = []
        best: tuple[int, int] | None = None
        best_ms = float("inf")
        for block_n, num_warps in _CLAUSE_COMPACT_GRID:
            cfg = ClauseCompactConfig(block_n=block_n, num_warps=num_warps)
            for _ in range(3):
                _clause_compact_impl(item_attrs, is_reverse, query_attrs, config=cfg)
            torch.cuda.synchronize()
            ms = _bench(
                lambda c=cfg: _clause_compact_impl(
                    item_attrs, is_reverse, query_attrs, config=c
                )
            )
            results.append({"block_n": block_n, "num_warps": num_warps, "ms": ms})
            if ms < best_ms:
                best_ms = ms
                best = (block_n, num_warps)
            click.echo(
                f"  [{key}] block_n={block_n:<4} num_warps={num_warps}  "
                f"-> {ms:.3f} ms",
                err=True,
            )
        assert best is not None
        per_regime[key] = {"winner": best, "winner_ms": best_ms, "all": results}
        click.echo(
            f"  [{key}] winner: block_n={best[0]} num_warps={best[1]} "
            f"({best_ms:.3f} ms)",
            err=True,
        )

    votes = Counter(per_regime[k]["winner"] for k in per_regime)
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1]))
    default = top[0][0]
    return {"per_regime": per_regime, "default": default}


def _print_clause_compact(arch: str, result: dict) -> None:
    bn, nw = result["default"]
    click.echo("")
    click.echo(
        "# Paste into retrieve/src/retrieve/kernels/triton/filters/clause_compact.py"
    )
    click.echo(f"# Tuned on {arch}; per-regime details in JSON output if --json-out was used.")
    click.echo(f"DEFAULT_CONFIG = ClauseCompactConfig(block_n={bn}, num_warps={nw})")


# ---------------------------------------------------------------------
# bloom_compact


def _tune_bloom_compact(dev: torch.device) -> dict:
    """Sweep (N, B, W) bloom regimes. Same fresh-buffer-per-call story as
    ``_tune_clause_compact`` — atomic_add is safe across reps."""
    per_regime: dict[str, dict] = {}
    for n, b, w in _BLOOM_REGIMES:
        key = f"N={n},B={b},W={w}"
        torch.manual_seed(0)
        qb = torch.randint(
            torch.iinfo(torch.int64).min,
            torch.iinfo(torch.int64).max,
            (b, w),
            dtype=torch.int64,
            device=dev,
        )
        sigs = torch.randint(
            torch.iinfo(torch.int64).min,
            torch.iinfo(torch.int64).max,
            (n, w),
            dtype=torch.int64,
            device=dev,
        )

        results: list[dict] = []
        best: tuple[int, int] | None = None
        best_ms = float("inf")
        for block_n, num_warps in _BLOOM_COMPACT_GRID:
            cfg = BloomCompactConfig(block_n=block_n, num_warps=num_warps)
            for _ in range(3):
                _bloom_compact_impl(qb, sigs, config=cfg)
            torch.cuda.synchronize()
            ms = _bench(lambda c=cfg: _bloom_compact_impl(qb, sigs, config=c))
            results.append({"block_n": block_n, "num_warps": num_warps, "ms": ms})
            if ms < best_ms:
                best_ms = ms
                best = (block_n, num_warps)
            click.echo(
                f"  [{key}] block_n={block_n:<4} num_warps={num_warps}  "
                f"-> {ms:.3f} ms",
                err=True,
            )
        assert best is not None
        per_regime[key] = {"winner": best, "winner_ms": best_ms, "all": results}
        click.echo(
            f"  [{key}] winner: block_n={best[0]} num_warps={best[1]} "
            f"({best_ms:.3f} ms)",
            err=True,
        )

    votes = Counter(per_regime[k]["winner"] for k in per_regime)
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1]))
    default = top[0][0]
    return {"per_regime": per_regime, "default": default}


def _print_bloom_compact(arch: str, result: dict) -> None:
    bn, nw = result["default"]
    click.echo("")
    click.echo(
        "# Paste into retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py"
    )
    click.echo(f"# Tuned on {arch}; per-regime details in JSON output if --json-out was used.")
    click.echo(f"DEFAULT_CONFIG = BloomCompactConfig(block_n={bn}, num_warps={nw})")


# ---------------------------------------------------------------------
# CLI


@click.command()
@click.option(
    "--kernel",
    type=click.Choice(
        [
            "fused_masked_knn_topk",
            "oporp_1bit_match_topk",
            "codesigned_probe_score",
            "clause_mask",
            "clause_compact",
            "bloom_compact",
        ]
    ),
    required=True,
)
@click.option("--device", default="cuda:0")
@click.option(
    "--json-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for full per-shape sweep results.",
)
def main(kernel: str, device: str, json_out: Path | None) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required to tune Triton kernels.")
    dev = torch.device(device)
    arch = _arch_str(dev)
    click.echo(f"# Tuning {kernel} on {arch} ({dev})", err=True)

    if kernel == "fused_masked_knn_topk":
        result = _tune_fmkt(dev)
        _print_fmkt(arch, result)
    elif kernel == "oporp_1bit_match_topk":
        result = _tune_oporp(dev)
        _print_oporp(arch, result)
    elif kernel == "codesigned_probe_score":
        result = _tune_cps(dev)
        _print_cps(arch, result)
    elif kernel == "clause_mask":
        result = _tune_clause_mask(dev)
        _print_clause_mask(arch, result)
    elif kernel == "clause_compact":
        result = _tune_clause_compact(dev)
        _print_clause_compact(arch, result)
    else:  # bloom_compact
        result = _tune_bloom_compact(dev)
        _print_bloom_compact(arch, result)

    if json_out is not None:
        json_out.write_text(json.dumps({"arch": arch, "kernel": kernel, **result}, indent=2))
        click.echo(f"# Full sweep written to {json_out}", err=True)


if __name__ == "__main__":
    main()
