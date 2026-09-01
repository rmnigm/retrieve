"""Offline tuner for the Triton kernels in ``retrieve.kernels``: run once on the target arch and
paste the printed ``DEFAULT_CONFIG = ...`` line into the kernel file (or pass it via the
wrapper's ``config=``). Each kernel is one ``KernelTuneSpec`` in ``KERNELS``, from which its click
subcommand is generated (see ``--help``); ``--regime`` overrides the built-in eval shapes for the
regime-swept kernels."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import click
import torch
import triton.testing as ttesting

from retrieve.kernels.filters.bloom_compact import BloomCompactConfig, _bloom_compact_impl
from retrieve.kernels.filters.clause_compact import ClauseCompactConfig, _clause_compact_impl
from retrieve.kernels.filters.clause_mask import ClauseMaskConfig, _clause_mask_impl
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
    _codesigned_probe_score_impl,
)
from retrieve.kernels.silvertorch.codesigned_probe_score_cuda import (
    CodesignedProbeScoreCudaConfig,
    _codesigned_probe_score_cuda_impl,
    _codesigned_probe_score_exact_cuda_impl,
    words_per_cluster,
)
from retrieve.kernels.silvertorch.codesigned_probe_score_exact import (
    CodesignedProbeScoreExactConfig,
    _codesigned_probe_score_exact_impl,
)

_FMKT_GRID = tuple((bn, nw) for bn in (32, 64, 128, 256) for nw in (4, 8))
_OPORP_GRID = tuple((bn, nw) for bn in (64, 128, 256, 512) for nw in (4, 8))
_CPS_GRID = tuple((bp, nw) for bp in (32, 64, 128, 256) for nw in (4, 8))
# Starts at 128, not 32 like _CPS_GRID: not a constraint (every _CPS_GRID point
# already satisfies block_p % num_warps == 0), but amortization. A CUDA block does a
# fixed amount of per-block setup — query words into registers, the (cluster, slot)
# divide, the tile bookkeeping — and at block_p=32 with num_warps=8 a warp scores 4
# items before exiting, so that setup dominates. The Triton kernel has no equivalent
# per-block register preload, which is why its grid can start lower.
# Third axis is the CUDA scorer's UNROLL (items in flight per segment), a template
# parameter of the .cu kernel — hence the fixed {1, 2, 4} rather than a range.
_CPS_CUDA_GRID = tuple(
    (bp, nw, un) for bp in (128, 256, 512, 1024) for nw in (4, 8) for un in (1, 2, 4)
)
# Filter-index kernels sweep a wider BLOCK_N range since the inner body varies and the optimum can
# land far from 256.
_FILTER_GRID = tuple((bn, nw) for bn in (128, 256, 512, 1024) for nw in (2, 4, 8))

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


def _rand_bits(shape: tuple[int, ...], dev: torch.device) -> torch.Tensor:
    ii = torch.iinfo(torch.int64)
    return torch.randint(ii.min, ii.max, shape, dtype=torch.int64, device=dev)


def _fmkt_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    p, d, b = regime
    n = max(p * 2, 1 << 16)  # need N >= P; generous N keeps the gather realistic
    torch.manual_seed(0)
    return dict(
        query=torch.randn(b, d, device=dev),
        item_embs=torch.randn(n, d, device=dev),
        positive_indices=torch.randint(0, n, (b, p), dtype=torch.long, device=dev),
        counts=torch.full((b,), p, dtype=torch.long, device=dev),
        k=min(64, p),
    )


def _oporp_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    n_bucket, has_indices, w, b = regime
    torch.manual_seed(0)
    corpus_n = max(n_bucket, 1 << 16) if has_indices else n_bucket
    if has_indices:
        p = n_bucket // 4 if n_bucket >= 4 else 1
        pos = torch.randint(0, corpus_n, (b, p), dtype=torch.long, device=dev)
        counts = torch.full((b,), p, dtype=torch.long, device=dev)
        k = min(64, p)
    else:
        pos = counts = None
        k = 64
    return dict(
        query_bits=_rand_bits((b, w), dev),
        item_bits=_rand_bits((corpus_n, w), dev),
        k=k,
        positive_indices=pos,
        counts=counts,
    )


def _cps_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    p, has_qb, d, b, w = regime
    n = max(p * 4, 1 << 16)
    torch.manual_seed(0)
    return dict(
        query=torch.randn(b, d, device=dev),
        flat_probed_items=torch.randint(0, n, (b, p), dtype=torch.long, device=dev),
        item_codes=torch.randint(-128, 128, (n, d), dtype=torch.int8, device=dev),
        global_scale=0.01,
        k=min(64, p),
        query_bits=_rand_bits((b, w), dev) if has_qb else None,
        bloom_sigs=_rand_bits((n, w), dev) if has_qb else None,
    )


def _cps_cuda_probe_family(
    dev: torch.device, p: int, b: int, n_target: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Synthetic padded IVF layout shared by both cuda CPS specs — the shape the phase-2
    mask kernels decode, which the Triton specs' flat random ids cannot stand in for.

    Clusters hold ``max_size = min(P, 1024)`` slots and each id at most once (randperm);
    ``n_probe = P / max_size`` clusters are probed per row so ``flat`` is exactly ``[B, P]``.
    Every slot holds a real id — there is **no** ``-1`` cluster padding here, unlike a real
    index — so the scorer's ``id < 0`` early-out never fires and the sweep measures the
    worst case for it. Deliberate: a padding rate is deployment-specific, and tuning
    against the fully-populated case cannot pick a config that only wins on empty slots.
    ``n_target`` is the corpus size to aim for; the returned ``N`` rounds it down to a whole
    number of clusters (and up to ``P``, since every probed slot must hold a real id).
    Returns ``(padded, probe_ids, flat, N, max_size)``; seed the caller's RNG first."""
    max_size = min(p, 1024)
    if p % max_size:
        raise click.ClickException(f"P must be a multiple of {max_size} for the cuda regimes")
    n_probe = p // max_size
    n_lists = max(n_target, p) // max_size
    n = n_lists * max_size
    padded = torch.randperm(n, device=dev).reshape(n_lists, max_size)
    probe_ids = torch.randint(0, n_lists, (b, n_probe), dtype=torch.int64, device=dev)
    return padded, probe_ids, padded[probe_ids].reshape(b, -1), n, max_size


def _cps_cuda_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    """CUDA-backend inputs for the same ``(P, HAS_QB, D, B, W)`` regimes as ``_cps_inputs``:
    a synthetic padded IVF layout, probed cluster ids, a sparse query signature, and a
    transposed index tuned to a ~20% pass rate.

    The HAS_QB=1 inputs deliberately differ from ``_cps_inputs``; see
    docs/system/kernels.md on comparing tuner output across the two backends."""
    p, has_qb, d, b, w = regime
    torch.manual_seed(0)
    _, probe_ids, flat, n, max_size = _cps_cuda_probe_family(dev, p, b, max(p * 4, 1 << 16))
    n_lists = n // max_size
    inputs: dict[str, Any] = dict(
        query=torch.randn(b, d, device=dev),
        flat_probed_items=flat,
        item_codes=torch.randint(-128, 128, (n, d), dtype=torch.int8, device=dev),
        global_scale=0.01,
        k=min(64, p),
        query_bits=None,
        bloom_sigs_t=None,
        probe_ids=None,
    )
    if has_qb:
        m_bits = w * 64
        n_set = min(25, m_bits)  # ~ k_hash=5 × 5 active values
        qb = torch.zeros(b, w, dtype=torch.int64, device=dev)
        pos = torch.randint(0, m_bits, (b, n_set), device=dev)
        rows = torch.arange(b, device=dev)
        for i in range(n_set):
            qb[rows, pos[:, i] // 64] |= torch.tensor(1, dtype=torch.int64, device=dev) << (
                pos[:, i] % 64
            )
        # OR of 4 uniform words → per-bit density 1 - 2^-4 = 0.9375 → pass rate
        # ≈ 0.9375^25 ≈ 0.2 at 25 set query bits.
        cols = n_lists * words_per_cluster(max_size)
        sigs_t = _rand_bits((m_bits, cols), dev)
        for _ in range(3):
            sigs_t |= _rand_bits((m_bits, cols), dev)
        inputs.update(query_bits=qb, bloom_sigs_t=sigs_t, probe_ids=probe_ids)
    return inputs


def _clause_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    """Build realistic (item_attrs, is_reverse, query_attrs) on ``dev`` — small vocab so random
    queries get a non-trivial pass/fail mix."""
    n, b, c, a_max = regime
    torch.manual_seed(0)
    n_vocab = 40
    return dict(
        item_clause_attrs=torch.randint(0, n_vocab, (n, c, a_max), dtype=torch.int64, device=dev),
        clause_is_reverse=torch.zeros(c, dtype=torch.bool, device=dev),
        query_clause_attrs=torch.randint(0, n_vocab, (b, c), dtype=torch.int64, device=dev),
    )


def _cpse_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    n, b, c, a_max = regime
    inputs = _clause_inputs(dev, regime)
    # D mirrors the codesigned-probe-score default (--d 128). P = n_probe × max_cluster_size is
    # deployment-specific; the middle of the CPS P-grid (capped by N) stands in for it here.
    d = 128
    p = min(n, 8192)
    inputs.update(
        query=torch.randn(b, d, device=dev),
        flat_probed_items=torch.randint(0, n, (b, p), dtype=torch.long, device=dev),
        item_codes=torch.randint(-128, 128, (n, d), dtype=torch.int8, device=dev),
        global_scale=0.01,
        k=min(64, p),
    )
    return inputs


def _cpse_cuda_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    """CUDA-backend twin of ``_cpse_inputs``: the same ``(N, B, C, A_MAX)`` clause regimes
    and the same D=128 / P stand-ins, but over the padded IVF probe family the phase-2
    clause-mask kernel decodes. ``max_size`` is the only extra input — the mask kernel reads
    ids from ``flat_probed_items``, so no cluster-major attribute copy exists.

    Unlike the bloom pair, these rows *are* comparable with ``codesigned-probe-score-exact``:
    both backends see the same attrs distribution and the same predicate."""
    n, b, c, a_max = regime
    torch.manual_seed(0)
    # D mirrors the codesigned-probe-score default (--d 128); P as in _cpse_inputs.
    d = 128
    p = min(n, 8192)
    _, _, flat, n_items, max_size = _cps_cuda_probe_family(dev, p, b, n)
    inputs = _clause_inputs(dev, (n_items, b, c, a_max))
    inputs.update(
        query=torch.randn(b, d, device=dev),
        flat_probed_items=flat,
        item_codes=torch.randint(-128, 128, (n_items, d), dtype=torch.int8, device=dev),
        global_scale=0.01,
        k=min(64, p),
        max_size=max_size,
    )
    return inputs


def _bloom_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    n, b, w = regime
    torch.manual_seed(0)
    return dict(qb=_rand_bits((b, w), dev), sigs=_rand_bits((n, w), dev))


@dataclass(frozen=True)
class KernelTuneSpec:
    """Everything the generic ``_sweep``/``_print``/CLI machinery needs to tune one kernel.

    A regime is one benchmark shape (a tuple of ints named by ``regime_labels``). Regime-swept
    kernels (``regime_arity`` set) expose a repeatable ``--regime`` flag over ``default_regimes``;
    dimension-swept kernels expose the ``dims`` flags instead and derive their regimes via
    ``expand_regimes`` (crossing the flags with the module's bucket/P axes)."""

    name: str  # click subcommand, e.g. "clause-mask"
    config_cls: type  # ClauseMaskConfig, ... — first dataclass field is the block size
    # Candidate configs as positional field values: (block, num_warps) for most kernels,
    # (block_p, num_warps, unroll) for the CUDA scorer. Entries are splatted into
    # config_cls, so their order must match the dataclass field order.
    grid: tuple[tuple[int, ...], ...]
    regime_labels: tuple[str, ...]  # names for the regime fields, e.g. ("N", "B", "C", "A_MAX")
    make_inputs: Callable[[torch.device, tuple[int, ...]], dict[str, Any]]  # regime → kwargs
    run: Callable[[dict[str, Any], Any], Any]  # (inputs, config) → one _impl call
    paste_path: str  # file the DEFAULT_CONFIG line goes into
    smoke_regime: tuple[int, ...]  # one tiny regime for the CI smoke test
    # Regime-swept kernels (--regime):
    default_regimes: tuple[tuple[int, ...], ...] = ()
    regime_arity: int | None = None  # 4 = N,B,C,A_MAX; 3 = N,B,W; None = dimension flags instead
    regime_fmt: str = ""  # --regime metavar for --help
    # Dimension-swept kernels (--d/--b/--w):
    dims: tuple[tuple[str, int, str], ...] = ()  # (flag, default, help)
    expand_regimes: Callable[..., tuple[tuple[int, ...], ...]] | None = None


def _config_fields(spec: KernelTuneSpec, entry: tuple[int, ...]) -> tuple[str, ...]:
    """Dataclass field names for a grid entry, positionally — the grid may be shorter than
    the config (the Triton kernels never sweep ``num_stages``)."""
    return tuple(f.name for f in fields(spec.config_cls))[: len(entry)]


def _fmt_entry(spec: KernelTuneSpec, entry: tuple[int, ...]) -> str:
    """One grid entry as ``block_p=256  num_warps=4  unroll=1``, padded so the sweep log
    stays column-aligned across block sizes."""
    return " ".join(
        f"{name}={v:<4}" for name, v in zip(_config_fields(spec, entry), entry)
    ).rstrip()


def _sweep(spec: KernelTuneSpec, dev: torch.device, regimes: tuple[tuple[int, ...], ...]) -> dict:
    """Per regime pick the lowest-latency grid entry from ``spec.grid``, then aggregate to one
    default by plurality vote (ties → lower ``num_warps``, then lower remaining fields).

    Tuning is offline and out-of-kernel on purpose; see
    docs/system/kernels.md § Autotune separation."""
    per_regime: dict[str, dict] = {}
    for regime in regimes:
        key = ",".join(f"{label}={v}" for label, v in zip(spec.regime_labels, regime))
        inputs = spec.make_inputs(dev, regime)

        results: list[dict] = []
        best: tuple[int, ...] | None = None
        best_ms = float("inf")
        for entry in spec.grid:
            cfg = spec.config_cls(*entry)
            # Warm the JIT cache before measuring; do_bench's warmup wouldn't cover a cold compile
            # of this config.
            for _ in range(3):
                spec.run(inputs, cfg)
            torch.cuda.synchronize()
            ms = _bench(lambda c=cfg: spec.run(inputs, c))
            results.append({**dict(zip(_config_fields(spec, entry), entry)), "ms": ms})
            if ms < best_ms:
                best_ms = ms
                best = entry
            click.echo(f"  [{key}] {_fmt_entry(spec, entry)}  -> {ms:.3f} ms", err=True)
        assert best is not None
        per_regime[key] = {"winner": best, "winner_ms": best_ms, "all": results}
        click.echo(
            f"  [{key}] winner: {_fmt_entry(spec, best)} ({best_ms:.3f} ms)",
            err=True,
        )

    votes = Counter(per_regime[k]["winner"] for k in per_regime)
    # Ties → lower num_warps (cheaper register pressure), then lower everything after it.
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][1:]))
    return {"per_regime": per_regime, "default": top[0][0]}


def _print(spec: KernelTuneSpec, arch: str, result: dict) -> None:
    entry = result["default"]
    args = ", ".join(f"{name}={v}" for name, v in zip(_config_fields(spec, entry), entry))
    cls_name = spec.config_cls.__name__
    click.echo("")
    click.echo(f"# Paste into {spec.paste_path}")
    click.echo(f"# Tuned on {arch}; per-regime details in JSON output if --json-out was used.")
    click.echo(f"DEFAULT_CONFIG = {cls_name}({args})")


KERNELS: tuple[KernelTuneSpec, ...] = (
    KernelTuneSpec(
        name="fused-masked-knn-topk",
        config_cls=FusedMaskedKnnTopkConfig,
        grid=_FMKT_GRID,
        regime_labels=("P", "D", "B"),
        make_inputs=_fmkt_inputs,
        run=lambda inputs, config: _fused_masked_knn_topk_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/kernels/linr/fused_masked_knn_topk.py",
        smoke_regime=(256, 64, 2),
        dims=(("d", 128, "Embedding dimension."), ("b", 16, "Batch size.")),
        expand_regimes=lambda d, b: tuple((p, d, b) for p in _P_BUCKETS),
    ),
    KernelTuneSpec(
        name="oporp-1bit-match-topk",
        config_cls=Oporp1BitMatchTopkConfig,
        grid=_OPORP_GRID,
        regime_labels=("N", "HAS_IDX", "W", "B"),
        make_inputs=_oporp_inputs,
        run=lambda inputs, config: _oporp_1bit_match_topk_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py",
        smoke_regime=(4096, 1, 2, 2),
        dims=(("w", 2, "int64 words per bit-vector (D=64*W)."), ("b", 16, "Batch size.")),
        expand_regimes=lambda w, b: tuple((n, hi, w, b) for n in _N_BUCKETS for hi in (0, 1)),
    ),
    KernelTuneSpec(
        name="codesigned-probe-score",
        config_cls=CodesignedProbeScoreConfig,
        grid=_CPS_GRID,
        regime_labels=("P", "HAS_QB", "D", "B", "W"),
        make_inputs=_cps_inputs,
        run=lambda inputs, config: _codesigned_probe_score_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py",
        smoke_regime=(1024, 1, 64, 2, 4),
        dims=(
            ("d", 128, "Embedding dimension."),
            ("b", 16, "Batch size."),
            ("w", 4, "int64 words per bloom signature."),
        ),
        expand_regimes=lambda d, b, w: tuple(
            (p, hq, d, b, w) for p in _DEFAULT_CPS_P_GRID for hq in (0, 1)
        ),
    ),
    # Shares regime axes with codesigned-probe-score so the two --json-out files join on
    # identical keys — but the HAS_QB=1 rows are NOT comparable (different input
    # distributions by design). See docs/system/kernels.md before comparing them.
    KernelTuneSpec(
        name="codesigned-probe-score-cuda",
        config_cls=CodesignedProbeScoreCudaConfig,
        grid=_CPS_CUDA_GRID,
        regime_labels=("P", "HAS_QB", "D", "B", "W"),
        make_inputs=_cps_cuda_inputs,
        run=lambda inputs, config: _codesigned_probe_score_cuda_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_cuda.py",
        smoke_regime=(1024, 1, 64, 2, 4),
        dims=(
            ("d", 128, "Embedding dimension."),
            ("b", 16, "Batch size."),
            ("w", 4, "int64 words per bloom signature."),
        ),
        expand_regimes=lambda d, b, w: tuple(
            (p, hq, d, b, w) for p in _DEFAULT_CPS_P_GRID for hq in (0, 1)
        ),
    ),
    # (N, B, C, A_MAX) defaults mirror the shipped clause regimes; the exact kernel's optimum also
    # tracks P = n_probe × max_cluster_size (approximated inside _cpse_inputs), so re-tune with
    # --regime per deployment rather than trusting these defaults.
    KernelTuneSpec(
        name="codesigned-probe-score-exact",
        config_cls=CodesignedProbeScoreExactConfig,
        grid=_CPS_GRID,
        regime_labels=("N", "B", "C", "A_MAX"),
        make_inputs=_cpse_inputs,
        run=lambda inputs, config: _codesigned_probe_score_exact_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py",
        smoke_regime=(4096, 2, 2, 2),
        default_regimes=_DEFAULT_CLAUSE_REGIMES,
        regime_arity=4,
        regime_fmt="N,B,C,A_MAX",
    ),
    # Same regime axes as codesigned-probe-score-exact, and unlike the bloom pair these
    # rows ARE comparable head-to-head (same attrs, same predicate). Shares
    # CodesignedProbeScoreCudaConfig — and therefore the single DEFAULT_CONFIG line in
    # codesigned_probe_score_cuda.py — with codesigned-probe-score-cuda, so reconcile the
    # two sweeps before pasting.
    KernelTuneSpec(
        name="codesigned-probe-score-exact-cuda",
        config_cls=CodesignedProbeScoreCudaConfig,
        grid=_CPS_CUDA_GRID,
        regime_labels=("N", "B", "C", "A_MAX"),
        make_inputs=_cpse_cuda_inputs,
        run=lambda inputs, config: _codesigned_probe_score_exact_cuda_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_cuda.py",
        smoke_regime=(4096, 2, 2, 2),
        default_regimes=_DEFAULT_CLAUSE_REGIMES,
        regime_arity=4,
        regime_fmt="N,B,C,A_MAX",
    ),
    KernelTuneSpec(
        name="clause-mask",
        config_cls=ClauseMaskConfig,
        grid=_FILTER_GRID,
        regime_labels=("N", "B", "C", "A_MAX"),
        make_inputs=_clause_inputs,
        run=lambda inputs, config: _clause_mask_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/kernels/filters/clause_mask.py",
        smoke_regime=(4096, 2, 2, 2),
        default_regimes=_DEFAULT_CLAUSE_REGIMES,
        regime_arity=4,
        regime_fmt="N,B,C,A_MAX",
    ),
    KernelTuneSpec(
        name="clause-compact",
        config_cls=ClauseCompactConfig,
        grid=_FILTER_GRID,
        regime_labels=("N", "B", "C", "A_MAX"),
        make_inputs=_clause_inputs,
        run=lambda inputs, config: _clause_compact_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/kernels/filters/clause_compact.py",
        smoke_regime=(4096, 2, 2, 2),
        default_regimes=_DEFAULT_CLAUSE_REGIMES,
        regime_arity=4,
        regime_fmt="N,B,C,A_MAX",
    ),
    KernelTuneSpec(
        name="bloom-compact",
        config_cls=BloomCompactConfig,
        grid=_FILTER_GRID,
        regime_labels=("N", "B", "W"),
        make_inputs=_bloom_inputs,
        run=lambda inputs, config: _bloom_compact_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/kernels/filters/bloom_compact.py",
        smoke_regime=(4096, 2, 4),
        default_regimes=_DEFAULT_BLOOM_REGIMES,
        regime_arity=3,
        regime_fmt="N,B,W",
    ),
)


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


def _register_subcommand(group: click.Group, spec: KernelTuneSpec) -> None:
    """Generate ``spec``'s click subcommand: a repeatable ``--regime`` when the spec sweeps
    declared regimes, the spec's dimension flags (``--d``/``--b``/``--w``) otherwise, plus the
    common ``--device``/``--json-out``."""
    kernel = spec.name.replace("-", "_")

    def callback(
        device: str, json_out: Path | None, regime_specs: tuple[str, ...] = (), **dims: int
    ) -> None:
        dev = _resolve_device(device)
        arch = _arch_str(dev)
        click.echo(f"# Tuning {kernel} on {arch} ({dev})", err=True)
        if spec.regime_arity is not None:
            parsed = tuple(_parse_regime(s, spec.regime_arity) for s in regime_specs)
            regimes = parsed or spec.default_regimes
        else:
            assert spec.expand_regimes is not None
            regimes = spec.expand_regimes(**dims)
        result = _sweep(spec, dev, regimes)
        _print(spec, arch, result)
        _dump_json(json_out, arch, kernel, result)

    cmd = _json_out_opt(callback)
    cmd = _device_opt(cmd)
    if spec.regime_arity is not None:
        cmd = click.option(
            "--regime",
            "regime_specs",
            multiple=True,
            metavar=spec.regime_fmt,
            help="Repeatable; tune for these shapes instead of the built-in eval regimes.",
        )(cmd)
    else:
        for flag, default, help_text in reversed(spec.dims):
            cmd = click.option(f"--{flag}", default=default, show_default=True, help=help_text)(cmd)
    group.command(spec.name)(cmd)


for _spec in KERNELS:
    _register_subcommand(main, _spec)


if __name__ == "__main__":
    main()
