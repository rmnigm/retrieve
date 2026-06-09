"""Offline tuner for the Triton kernels in ``retrieve.kernels``: run once on the target arch and
paste the printed ``DEFAULT_CONFIG = ...`` line into the kernel file (or pass it via the
wrapper's ``config=``). Each kernel is a subcommand (see ``--help``); ``--regime`` overrides the
built-in eval shapes for the filter kernels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import click
import torch
import triton.testing as ttesting

from retrieve.kernels.filters.bloom_compact import (
    BloomCompactConfig,
    _bloom_compact_impl,
)
from retrieve.kernels.filters.clause_compact import (
    ClauseCompactConfig,
    _clause_compact_impl,
)
from retrieve.kernels.filters.clause_mask import (
    ClauseMaskConfig,
    _clause_mask_impl,
)
from retrieve.kernels.linr.fused_masked_knn_topk import (
    _P_BUCKETS,
    FusedMaskedKnnTopkConfig,
    _fused_masked_knn_topk_impl,
)
from retrieve.kernels.linr.oporp_1bit_match_topk import (
    _N_BUCKETS,
    Oporp1BitMatchTopkConfig,
    _oporp_1bit_match_topk_impl,
)
from retrieve.kernels.silvertorch.codesigned_probe_score import (
    CodesignedProbeScoreConfig,
    codesigned_probe_score,
)

_FMKT_GRID = [(bn, nw) for bn in (32, 64, 128, 256) for nw in (4, 8)]
_OPORP_GRID = [(bn, nw) for bn in (64, 128, 256, 512) for nw in (4, 8)]
_CPS_GRID = [(bp, nw) for bp in (32, 64, 128, 256) for nw in (4, 8)]
# Filter-index kernels sweep a wider BLOCK_N range since the inner body varies and the optimum can
# land far from 256.
_CLAUSE_MASK_GRID = [(bn, nw) for bn in (128, 256, 512, 1024) for nw in (2, 4, 8)]
_CLAUSE_COMPACT_GRID = [(bn, nw) for bn in (128, 256, 512, 1024) for nw in (2, 4, 8)]
_BLOOM_COMPACT_GRID = [(bn, nw) for bn in (128, 256, 512, 1024) for nw in (2, 4, 8)]

# Default filter regimes mirror real-eval shapes (goodreads N≈800K C=4, arxiv N≈3M C=5, A_MAX=4;
# bloom W=16 from m_bits=1024). End-users override via --regime.
_DEFAULT_CLAUSE_REGIMES = (
    # (N, B, C, A_MAX)
    (797_085, 1, 4, 4),  # goodreads, single-query
    (797_085, 16, 4, 4),  # goodreads, batched
    (2_988_997, 1, 5, 4),  # arxiv-retrieval, single-query
    (2_988_997, 16, 5, 4),  # arxiv-retrieval, batched
    (15_000_001, 16, 5, 4),  # arxiv-synth-15m, batched (DRAM-streaming extreme)
)
_DEFAULT_BLOOM_REGIMES = (
    # (N, B, W) — W=16 from m_bits=1024 default
    (797_085, 1, 16),
    (797_085, 16, 16),
    (2_988_997, 1, 16),
    (2_988_997, 16, 16),
    (15_000_001, 16, 16),  # arxiv-synth-15m, batched (DRAM-streaming extreme)
)

# Default CPS P-grid spanning typical IVF sizings; hard-coded since the kernel isn't bucketed on P.
_DEFAULT_CPS_P_GRID = (1024, 8192, 65536)


def _arch_str(device: torch.device) -> str:
    cap = torch.cuda.get_device_capability(device)
    return f"sm_{cap[0]}{cap[1]}"


def _bench(fn) -> float:
    """Median ms for ``fn`` under do_bench — long enough to settle the JIT cache, short enough to
    keep the sweep quick."""
    median, _, _ = ttesting.do_bench(fn, quantiles=[0.5, 0.2, 0.8], rep=500, warmup=100)
    return float(median)


def _parse_regime(spec: str, arity: int) -> tuple[int, ...]:
    """Parse a ``--regime N,B,C,A_MAX`` (or ``N,B,W``) flag value."""
    parts = spec.split(",")
    if len(parts) != arity:
        raise click.BadParameter(
            f"expected {arity} comma-separated ints, got {len(parts)!r}: {spec!r}"
        )
    try:
        return tuple(int(p) for p in parts)
    except ValueError as e:
        raise click.BadParameter(f"non-integer field in {spec!r}: {e}") from e


def _tune_fmkt(dev: torch.device, d: int, b: int) -> dict:
    """Per bucket pick the lowest-latency (block_n, num_warps), then aggregate winners into one
    DEFAULT_CONFIG by plurality vote (ties → lower num_warps)."""
    per_bucket: dict[int, dict] = {}
    for p in _P_BUCKETS:
        # Need N >= P; pick generous N so the gather is realistic.
        n = max(p * 2, 1 << 16)
        torch.manual_seed(0)
        query = torch.randn(b, d, device=dev)
        embs = torch.randn(n, d, device=dev)
        pos = torch.randint(0, n, (b, p), dtype=torch.long, device=dev)
        counts = torch.full((b,), p, dtype=torch.long, device=dev)
        k = min(64, p)

        results: list[dict] = []
        best: tuple[int, int] | None = None
        best_ms = float("inf")
        for block_n, num_warps in _FMKT_GRID:
            cfg = FusedMaskedKnnTopkConfig(block_n=block_n, num_warps=num_warps)
            # Warm the JIT cache before measuring; do_bench's warmup wouldn't cover a cold compile
            # of this config.
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
                f"  [P={p:>7}] block_n={block_n:<4} num_warps={num_warps}  -> {ms:.3f} ms",
                err=True,
            )
        assert best is not None
        per_bucket[p] = {"winner": best, "winner_ms": best_ms, "all": results}
        click.echo(
            f"  [P={p:>7}] winner: block_n={best[0]} num_warps={best[1]} ({best_ms:.3f} ms)",
            err=True,
        )

    votes = Counter(per_bucket[p]["winner"] for p in _P_BUCKETS)
    # Ties → lower num_warps (cheaper register pressure).
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1]))
    default = top[0][0]
    return {"per_bucket": per_bucket, "default": default}


def _print_fmkt(arch: str, result: dict) -> None:
    bn, nw = result["default"]
    click.echo("")
    click.echo("# Paste into retrieve/src/retrieve/kernels/linr/fused_masked_knn_topk.py")
    click.echo(f"# Tuned on {arch}; per-bucket details in JSON output if --json-out was used.")
    click.echo(f"DEFAULT_CONFIG = FusedMaskedKnnTopkConfig(block_n={bn}, num_warps={nw})")


def _tune_oporp(dev: torch.device, w: int, b: int) -> dict:
    """Sweep (n_bucket × has_indices) regimes, pick the winning tile per regime, aggregate to one
    DEFAULT_CONFIG."""
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
                    _oporp_1bit_match_topk_impl(
                        query_bits,
                        item_bits,
                        k,
                        positive_indices=pos,
                        counts=counts,
                        config=cfg,
                    )
                torch.cuda.synchronize()
                ms = _bench(
                    lambda c=cfg: _oporp_1bit_match_topk_impl(
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
                    f"  [{key}] block_n={block_n:<4} num_warps={num_warps}  -> {ms:.3f} ms",
                    err=True,
                )
            assert best is not None
            per_regime[key] = {"winner": best, "winner_ms": best_ms, "all": results}
            click.echo(
                f"  [{key}] winner: block_n={best[0]} num_warps={best[1]} ({best_ms:.3f} ms)",
                err=True,
            )

    votes = Counter(per_regime[k]["winner"] for k in per_regime)
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1]))
    default = top[0][0]
    return {"per_regime": per_regime, "default": default}


def _print_oporp(arch: str, result: dict) -> None:
    bn, nw = result["default"]
    click.echo("")
    click.echo("# Paste into retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py")
    click.echo(f"# Tuned on {arch}; per-regime details in JSON output if --json-out was used.")
    click.echo(f"DEFAULT_CONFIG = Oporp1BitMatchTopkConfig(block_n={bn}, num_warps={nw})")


def _tune_cps(dev: torch.device, d: int, b: int, w: int) -> dict:
    """SilverTorch phase-2+3 kernel: P is fixed per index (no bucketing), so sweep representative P
    values (× has_qb on/off) and aggregate to one DEFAULT_CONFIG."""
    per_regime: dict[str, dict] = {}
    for p in _DEFAULT_CPS_P_GRID:
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
                    f"  [{key}] block_p={block_p:<4} num_warps={num_warps}  -> {ms:.3f} ms",
                    err=True,
                )
            assert best is not None
            per_regime[key] = {"winner": best, "winner_ms": best_ms, "all": results}
            click.echo(
                f"  [{key}] winner: block_p={best[0]} num_warps={best[1]} ({best_ms:.3f} ms)",
                err=True,
            )

    votes = Counter(per_regime[k]["winner"] for k in per_regime)
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1]))
    default = top[0][0]
    return {"per_regime": per_regime, "default": default}


def _print_cps(arch: str, result: dict) -> None:
    bp, nw = result["default"]
    click.echo("")
    click.echo("# Paste into retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py")
    click.echo(f"# Tuned on {arch}; per-regime details in JSON output if --json-out was used.")
    click.echo(f"DEFAULT_CONFIG = CodesignedProbeScoreConfig(block_p={bp}, num_warps={nw})")


def _make_clause_inputs(
    dev: torch.device, n: int, b: int, c: int, a_max: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build realistic (item_attrs, is_reverse, query_attrs) on ``dev`` — small vocab so random
    queries get a non-trivial pass/fail mix."""
    torch.manual_seed(0)
    n_vocab = 40
    item_attrs = torch.randint(0, n_vocab, (n, c, a_max), dtype=torch.int64, device=dev)
    is_reverse = torch.zeros(c, dtype=torch.bool, device=dev)
    query_attrs = torch.randint(0, n_vocab, (b, c), dtype=torch.int64, device=dev)
    return item_attrs, is_reverse, query_attrs


def _tune_clause_mask(dev: torch.device, regimes: tuple[tuple[int, int, int, int], ...]) -> dict:
    """Sweep (N, B, C, A_MAX) regimes; aggregate per-regime winners to one DEFAULT_CONFIG
    (plurality, ties → lower num_warps)."""
    per_regime: dict[str, dict] = {}
    for n, b, c, a_max in regimes:
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
                lambda c=cfg: _clause_mask_impl(item_attrs, is_reverse, query_attrs, config=c)
            )
            results.append({"block_n": block_n, "num_warps": num_warps, "ms": ms})
            if ms < best_ms:
                best_ms = ms
                best = (block_n, num_warps)
            click.echo(
                f"  [{key}] block_n={block_n:<4} num_warps={num_warps}  -> {ms:.3f} ms",
                err=True,
            )
        assert best is not None
        per_regime[key] = {"winner": best, "winner_ms": best_ms, "all": results}
        click.echo(
            f"  [{key}] winner: block_n={best[0]} num_warps={best[1]} ({best_ms:.3f} ms)",
            err=True,
        )

    votes = Counter(per_regime[k]["winner"] for k in per_regime)
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1]))
    default = top[0][0]
    return {"per_regime": per_regime, "default": default}


def _print_clause_mask(arch: str, result: dict) -> None:
    bn, nw = result["default"]
    click.echo("")
    click.echo("# Paste into retrieve/src/retrieve/kernels/filters/clause_mask.py")
    click.echo(f"# Tuned on {arch}; per-regime details in JSON output if --json-out was used.")
    click.echo(f"DEFAULT_CONFIG = ClauseMaskConfig(block_n={bn}, num_warps={nw})")


def _tune_clause_compact(dev: torch.device, regimes: tuple[tuple[int, int, int, int], ...]) -> dict:
    """Same regime grid as clause_mask; ``_clause_compact_impl`` allocates fresh buffers per call,
    so the kernel's atomic_add accumulation warning doesn't apply to the tuner."""
    per_regime: dict[str, dict] = {}
    for n, b, c, a_max in regimes:
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
                lambda c=cfg: _clause_compact_impl(item_attrs, is_reverse, query_attrs, config=c)
            )
            results.append({"block_n": block_n, "num_warps": num_warps, "ms": ms})
            if ms < best_ms:
                best_ms = ms
                best = (block_n, num_warps)
            click.echo(
                f"  [{key}] block_n={block_n:<4} num_warps={num_warps}  -> {ms:.3f} ms",
                err=True,
            )
        assert best is not None
        per_regime[key] = {"winner": best, "winner_ms": best_ms, "all": results}
        click.echo(
            f"  [{key}] winner: block_n={best[0]} num_warps={best[1]} ({best_ms:.3f} ms)",
            err=True,
        )

    votes = Counter(per_regime[k]["winner"] for k in per_regime)
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1]))
    default = top[0][0]
    return {"per_regime": per_regime, "default": default}


def _print_clause_compact(arch: str, result: dict) -> None:
    bn, nw = result["default"]
    click.echo("")
    click.echo("# Paste into retrieve/src/retrieve/kernels/filters/clause_compact.py")
    click.echo(f"# Tuned on {arch}; per-regime details in JSON output if --json-out was used.")
    click.echo(f"DEFAULT_CONFIG = ClauseCompactConfig(block_n={bn}, num_warps={nw})")


def _tune_bloom_compact(dev: torch.device, regimes: tuple[tuple[int, int, int], ...]) -> dict:
    """Sweep (N, B, W) bloom regimes; fresh buffers per call, so atomic_add is safe across reps
    (like ``_tune_clause_compact``)."""
    per_regime: dict[str, dict] = {}
    for n, b, w in regimes:
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
                f"  [{key}] block_n={block_n:<4} num_warps={num_warps}  -> {ms:.3f} ms",
                err=True,
            )
        assert best is not None
        per_regime[key] = {"winner": best, "winner_ms": best_ms, "all": results}
        click.echo(
            f"  [{key}] winner: block_n={best[0]} num_warps={best[1]} ({best_ms:.3f} ms)",
            err=True,
        )

    votes = Counter(per_regime[k]["winner"] for k in per_regime)
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1]))
    default = top[0][0]
    return {"per_regime": per_regime, "default": default}


def _print_bloom_compact(arch: str, result: dict) -> None:
    bn, nw = result["default"]
    click.echo("")
    click.echo("# Paste into retrieve/src/retrieve/kernels/filters/bloom_compact.py")
    click.echo(f"# Tuned on {arch}; per-regime details in JSON output if --json-out was used.")
    click.echo(f"DEFAULT_CONFIG = BloomCompactConfig(block_n={bn}, num_warps={nw})")


def _resolve_device(device: str) -> torch.device:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required to tune Triton kernels.")
    return torch.device(device)


def _dump_json(json_out: Path | None, arch: str, kernel: str, result: dict) -> None:
    if json_out is None:
        return
    json_out.write_text(json.dumps({"arch": arch, "kernel": kernel, **result}, indent=2))
    click.echo(f"# Full sweep written to {json_out}", err=True)


_device_opt = click.option("--device", default="cuda:0", show_default=True)
_json_out_opt = click.option(
    "--json-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for full per-shape sweep results.",
)


@click.group()
def main() -> None:
    """Offline tuner for retrieve's Triton kernels: pick a kernel subcommand (flags vary) and paste
    the printed ``DEFAULT_CONFIG = …`` line into the kernel source."""


@main.command("fused-masked-knn-topk")
@click.option("--d", default=128, show_default=True, help="Embedding dimension.")
@click.option("--b", default=16, show_default=True, help="Batch size.")
@_device_opt
@_json_out_opt
def _cmd_fmkt(d: int, b: int, device: str, json_out: Path | None) -> None:
    dev = _resolve_device(device)
    arch = _arch_str(dev)
    click.echo(f"# Tuning fused_masked_knn_topk on {arch} ({dev})", err=True)
    result = _tune_fmkt(dev, d=d, b=b)
    _print_fmkt(arch, result)
    _dump_json(json_out, arch, "fused_masked_knn_topk", result)


@main.command("oporp-1bit-match-topk")
@click.option("--w", default=2, show_default=True, help="int64 words per bit-vector (D=64*W).")
@click.option("--b", default=16, show_default=True, help="Batch size.")
@_device_opt
@_json_out_opt
def _cmd_oporp(w: int, b: int, device: str, json_out: Path | None) -> None:
    dev = _resolve_device(device)
    arch = _arch_str(dev)
    click.echo(f"# Tuning oporp_1bit_match_topk on {arch} ({dev})", err=True)
    result = _tune_oporp(dev, w=w, b=b)
    _print_oporp(arch, result)
    _dump_json(json_out, arch, "oporp_1bit_match_topk", result)


@main.command("codesigned-probe-score")
@click.option("--d", default=128, show_default=True, help="Embedding dimension.")
@click.option("--b", default=16, show_default=True, help="Batch size.")
@click.option("--w", default=4, show_default=True, help="int64 words per bloom signature.")
@_device_opt
@_json_out_opt
def _cmd_cps(d: int, b: int, w: int, device: str, json_out: Path | None) -> None:
    dev = _resolve_device(device)
    arch = _arch_str(dev)
    click.echo(f"# Tuning codesigned_probe_score on {arch} ({dev})", err=True)
    result = _tune_cps(dev, d=d, b=b, w=w)
    _print_cps(arch, result)
    _dump_json(json_out, arch, "codesigned_probe_score", result)


_regime_clause_opt = click.option(
    "--regime",
    "regime_specs",
    multiple=True,
    metavar="N,B,C,A_MAX",
    help="Repeatable; tune for these shapes instead of the built-in eval regimes.",
)
_regime_bloom_opt = click.option(
    "--regime",
    "regime_specs",
    multiple=True,
    metavar="N,B,W",
    help="Repeatable; tune for these shapes instead of the built-in eval regimes.",
)


def _clause_regimes(specs: tuple[str, ...]) -> tuple[tuple[int, int, int, int], ...]:
    if not specs:
        return _DEFAULT_CLAUSE_REGIMES
    return tuple(_parse_regime(s, 4) for s in specs)  # type: ignore[return-value]


def _bloom_regimes(specs: tuple[str, ...]) -> tuple[tuple[int, int, int], ...]:
    if not specs:
        return _DEFAULT_BLOOM_REGIMES
    return tuple(_parse_regime(s, 3) for s in specs)  # type: ignore[return-value]


@main.command("clause-mask")
@_regime_clause_opt
@_device_opt
@_json_out_opt
def _cmd_clause_mask(regime_specs: tuple[str, ...], device: str, json_out: Path | None) -> None:
    dev = _resolve_device(device)
    arch = _arch_str(dev)
    click.echo(f"# Tuning clause_mask on {arch} ({dev})", err=True)
    result = _tune_clause_mask(dev, regimes=_clause_regimes(regime_specs))
    _print_clause_mask(arch, result)
    _dump_json(json_out, arch, "clause_mask", result)


@main.command("clause-compact")
@_regime_clause_opt
@_device_opt
@_json_out_opt
def _cmd_clause_compact(regime_specs: tuple[str, ...], device: str, json_out: Path | None) -> None:
    dev = _resolve_device(device)
    arch = _arch_str(dev)
    click.echo(f"# Tuning clause_compact on {arch} ({dev})", err=True)
    result = _tune_clause_compact(dev, regimes=_clause_regimes(regime_specs))
    _print_clause_compact(arch, result)
    _dump_json(json_out, arch, "clause_compact", result)


@main.command("bloom-compact")
@_regime_bloom_opt
@_device_opt
@_json_out_opt
def _cmd_bloom_compact(regime_specs: tuple[str, ...], device: str, json_out: Path | None) -> None:
    dev = _resolve_device(device)
    arch = _arch_str(dev)
    click.echo(f"# Tuning bloom_compact on {arch} ({dev})", err=True)
    result = _tune_bloom_compact(dev, regimes=_bloom_regimes(regime_specs))
    _print_bloom_compact(arch, result)
    _dump_json(json_out, arch, "bloom_compact", result)


if __name__ == "__main__":
    main()
