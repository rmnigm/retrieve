"""Nested loop driver for the retrieval benchmark.

```
run_sweep                    # pin globals, warm GPU, dispatch
└─ run_filter_kind           # build filter modules
   └─ run_one_sweep          # synth qa, build/load oracle, iterate algos
      └─ evaluate_cell       # build algo, quality, prewarm, per-bs perf rows
```

Run-wide inputs travel in a ``SweepContext``, per-filter_kind state in a
``FilterAssets``. See docs/system/evaluation.md for the driver flow.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, cast, get_args

import torch
from loguru import logger

from retrieval.algos import (
    RetrievalAlgo,
    build_algorithm,
    build_filter,
    is_valid_combo,
    supports,
)
from retrieval.config import EvalConfig, FilterCfg, FilterKind, FilterSweepCfg
from retrieval.context import EMPTY_ASSETS, FilterAssets, SweepContext
from retrieval.loaders import build_sweep_qa, load_filter_assets
from retrieval.measure import (
    PerfStats,
    cuda_allocated_mib,
    pin_precision_globals,
    warm_gpu_once,
)
from retrieval.oracle import load_or_build_oracle
from retrieval.passes import QualityStats, perf_pass_cached, quality_pass_cached
from retrieve.interfaces import Backend, FilterModule

# Backends whose only non-torch kernels are SilverTorch's fused probe scoring (the CUDA C++
# backend and its CuTe DSL port). They have no filter kernels of their own, so standalone
# FilterModules — and the oracle — are built with triton for these cells.
_TRITON_FILTER_BACKENDS: tuple[Backend, ...] = ("cuda", "cute")

# ----- top-level driver -------------------------------------------------------


def run_sweep(ctx: SweepContext) -> list[dict]:
    """Loop over (filter_kind, sweep, algo, k, batch_size) and emit rows."""
    pin_precision_globals()
    warm_gpu_once(ctx.device)

    if ctx.cfg.filters is not None:
        ctx.gt_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for filter_kind, fcfg in _select_filter_iter(ctx.cfg, ctx.filter_kinds):
        rows.extend(run_filter_kind(filter_kind, fcfg, ctx))
    return rows


def _select_filter_iter(
    cfg: EvalConfig, filter_kinds: tuple[str, ...]
) -> Iterator[tuple[FilterKind, FilterCfg]]:
    """Yield ``(filter_kind, FilterCfg)`` for each enabled filter kind.

    Yambda inserts a single synthetic ``("none", …)`` cell. ``--filter-kind``
    on the CLI restricts which kinds are yielded; empty tuple = all. YAML
    keys are plain strings; this is the one place they are narrowed to
    ``FilterKind`` — an unknown kind raises here rather than silently
    failing every ``supports`` check downstream.
    """
    if cfg.filters is None:
        yield "none", FilterCfg(sweeps=[FilterSweepCfg(name="full_scan")])
        return
    for filter_kind, fcfg in cfg.filters.items():
        if filter_kind not in get_args(FilterKind):
            raise ValueError(f"unknown filter_kind in config: {filter_kind!r}")
        if filter_kinds and filter_kind not in filter_kinds:
            continue
        yield cast(FilterKind, filter_kind), fcfg


# ----- per filter_kind --------------------------------------------------------


def run_filter_kind(filter_kind: FilterKind, fcfg: FilterCfg, ctx: SweepContext) -> list[dict]:
    """Build filter+oracle modules, iterate sweeps."""
    assets = _build_filter_modules(filter_kind, fcfg, ctx)

    rows: list[dict] = []
    for sweep in fcfg.sweeps:
        if ctx.sweep_filter and sweep.name != ctx.sweep_filter:
            continue
        rows.extend(run_one_sweep(sweep, filter_kind, ctx, assets))
        # Release dynamo compile cache + CUDA-graph private pools between
        # sweeps. Without this, graphs from earlier sweeps stay pinned and
        # large-N configs OOM (observed on arxiv/d256). Costs ~30-60s
        # recompile at the start of the next sweep.
        torch._dynamo.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if ctx.cfg.filters is not None:
        # Drop filter_mods/oracle_filter refs so the pool can reclaim them.
        del assets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _build_filter_modules(
    filter_kind: FilterKind, fcfg: FilterCfg, ctx: SweepContext
) -> FilterAssets:
    """Build per-backend index-side filters and the oracle-side exact filter.

    The oracle filter is always **exact**: bloom's false positives must never leak
    into ground truth.

    One ``FilterModule`` per backend so the filter kernel backend matches
    the algo's in each cell. The oracle-side exact filter is always
    triton (or torch on CPU); on ``filter_kind="clause"`` it reuses the
    triton variant of the index-side filter when available. On
    ``"bloom"`` we always build a separate ``ExactAttributeFilter``
    over the same attrs — bloom's false positives must NOT leak into the
    ground truth.
    """
    if ctx.cfg.filters is None:
        return EMPTY_ASSETS

    item_attrs_narrow, clause_is_reverse = load_filter_assets(
        filter_kind, fcfg, Path(ctx.cfg.data_dir), ctx.device
    )

    filter_mods: dict[Backend, FilterModule | None] = {}
    for backend in ctx.backends:
        # No CUDA C++ / CuTe DSL filter kernels exist (only SilverTorch's fused probe
        # scoring), so cuda and cute cells get the triton filter rather than silently
        # falling to torch.
        filter_backend: Backend = "triton" if backend in _TRITON_FILTER_BACKENDS else backend
        filter_mods[backend] = build_filter(
            filter_kind,
            item_attrs_narrow=item_attrs_narrow,
            clause_is_reverse=clause_is_reverse,
            bloom_m_bits=fcfg.m_bits,
            bloom_k_hash=fcfg.k_hash,
            device=ctx.device,
            backend=filter_backend,
        )

    oracle_backend: Backend = (
        "triton"
        if any(b == "triton" or b in _TRITON_FILTER_BACKENDS for b in ctx.backends)
        else ctx.backends[0]
    )
    if filter_kind == "clause":
        oracle_filter: FilterModule | None = filter_mods[
            "triton" if "triton" in ctx.backends else ctx.backends[0]
        ]
    elif filter_kind == "bloom":
        oracle_filter = build_filter(
            "clause",
            item_attrs_narrow=item_attrs_narrow,
            clause_is_reverse=clause_is_reverse,
            device=ctx.device,
            backend=oracle_backend,
        )
    else:
        oracle_filter = None
    if filter_mods:
        any_mod = next(iter(filter_mods.values()))
        if any_mod is not None:
            logger.info(
                "  filter modules built: {} × {} (oracle: {})",
                type(any_mod).__name__,
                len(filter_mods),
                type(oracle_filter).__name__ if oracle_filter is not None else "none",
            )
    n_clauses = int(item_attrs_narrow.shape[1]) if item_attrs_narrow is not None else 0
    return FilterAssets(
        filter_mods=filter_mods,
        oracle_filter=oracle_filter,
        item_attrs_narrow=item_attrs_narrow,
        clause_is_reverse=clause_is_reverse,
        n_clauses=n_clauses,
    )


# ----- per sweep --------------------------------------------------------------


def run_one_sweep(
    sweep: FilterSweepCfg,
    filter_kind: FilterKind,
    ctx: SweepContext,
    assets: FilterAssets,
) -> list[dict]:
    """Synthesise per-sweep qa, load/build oracle, then iterate algo x backend x params x k.

    The oracle is built once per sweep and shared across every cell below it.
    """
    logger.info("=== filter_kind={} sweep={} ===", filter_kind, sweep.name)

    qa_n_sweep, skip_mask = build_sweep_qa(
        sweep, filter_kind, ctx.qa_narrow_all, assets.n_clauses
    )

    n_users = ctx.queries.shape[0]
    n_kept = int((~skip_mask).sum().item()) if skip_mask is not None else n_users
    logger.info("  kept users: {} / {}", n_kept, n_users)

    # Skipped when --skip-quality is on: oracle is only used to score recall,
    # never for perf timing or skip-mask synthesis.
    oracle_topk: torch.Tensor | None = None
    if ctx.cfg.filters is not None and filter_kind != "none" and not ctx.skip_quality:
        oracle_topk = load_or_build_oracle(
            ctx.gt_dir,
            sweep.name,
            ctx.k_gt,
            item_embs=ctx.item_embs,
            queries=ctx.queries,
            qa_narrow_sweep=qa_n_sweep,
            skip_mask=skip_mask,
            oracle_filter=assets.oracle_filter,
            device=ctx.device,
        )

    sweep_assets = assets.for_sweep(
        qa_n_sweep=qa_n_sweep,
        skip_mask=skip_mask,
        oracle_topk=oracle_topk,
        n_kept=n_kept,
    )

    rows: list[dict] = []
    for algo in ctx.algorithms:
        if not supports(algo, filter_kind):
            logger.info("skipping {}: unsupported filter_kind {}", algo, filter_kind)
            continue
        for backend in ctx.backends:
            for params in ctx.cfg.algo_params.get(algo, [{}]):
                params = dict(params)  # defensive copy; combos are reused across k/bs loops
                if not is_valid_combo(algo, params):
                    logger.warning("skipping invalid combo {}: {}", algo, params)
                    continue
                for k in ctx.cfg.ks:
                    rows.extend(
                        evaluate_cell(
                            algo,
                            params,
                            k,
                            backend,
                            sweep,
                            filter_kind,
                            ctx,
                            sweep_assets,
                        )
                    )
    return rows


# ----- per cell ---------------------------------------------------------------


def evaluate_cell(
    algo: str,
    params: dict[str, Any],
    k: int,
    backend: Backend,
    sweep: FilterSweepCfg,
    filter_kind: FilterKind,
    ctx: SweepContext,
    assets: FilterAssets,
) -> list[dict]:
    """Build the algo, score quality, prewarm autotune, time each batch size."""
    _reset_cuda_state_for_cell()
    mem_before = cuda_allocated_mib()

    algo_obj = _build_algo(algo, params, k, backend, filter_kind, ctx, assets)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    index_mem = cuda_allocated_mib() - mem_before

    desc = f"{filter_kind}/{sweep.name}/{algo} k={k}"
    quality = _run_quality(algo_obj, k, desc, ctx, assets)

    _autotune_prewarm(algo_obj, ctx, assets)

    rows: list[dict] = []
    for bs in ctx.cfg.batch_sizes:
        stats = perf_pass_cached(
            algo_obj,
            ctx.queries,
            batch_size=bs,
            device=ctx.device,
            seed=ctx.cfg.seed,
            qa_narrow=assets.qa_n_sweep,
            skip_mask=assets.skip_mask,
        )
        rows.append(
            _make_perf_row(
                filter_kind=filter_kind,
                sweep_name=sweep.name,
                algo=algo,
                params=params,
                k=k,
                bs=bs,
                suite=ctx.suite,
                n_kept=assets.n_kept,
                seed=ctx.cfg.seed,
                stats=stats,
                index_mem=index_mem,
                quality=quality,
                backend=backend,
            )
        )
        _log_perf_line(filter_kind, sweep.name, algo, backend, params, k, bs, stats, quality)

    _release_algo(algo_obj)
    return rows


# ----- per-cell helpers -------------------------------------------------------


def _reset_cuda_state_for_cell() -> None:
    """Sync, drop pooled memory, reset peak — so per-cell index_mem is clean."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _build_algo(
    algo: str,
    params: dict[str, Any],
    k: int,
    backend: Backend,
    filter_kind: FilterKind,
    ctx: SweepContext,
    assets: FilterAssets,
) -> RetrievalAlgo:
    """Construct one cell's algo from the run context.

    Eligibility is decided by ``supports`` in ``run_one_sweep``, so any exception
    here is a genuine construction error and propagates rather than dropping the cell.
    """
    return build_algorithm(
        algo,
        ctx.item_embs,
        k=k,
        filter_kind=filter_kind,
        filter_mod=assets.filter_mods.get(backend),
        item_attrs_narrow=assets.item_attrs_narrow,
        clause_is_reverse=assets.clause_is_reverse,
        params=params,
        backend=backend,
    )


def _run_quality(
    algo_obj: RetrievalAlgo,
    k: int,
    desc: str,
    ctx: SweepContext,
    assets: FilterAssets,
) -> QualityStats:
    """Dispatch quality pass: NaN, held-out targets, or oracle top-K."""
    if ctx.skip_quality:
        nan = float("nan")
        return QualityStats(recall=nan, ndcg=nan, precision=nan, mrr=nan)

    if assets.oracle_topk is None:
        # No oracle: run_one_sweep builds one exactly when
        # cfg.filters is set and filter_kind != "none"; otherwise (yambda /
        # the "none" cell) quality is scored against the held-out targets.
        return quality_pass_cached(
            algo_obj,
            ctx.queries,
            ctx.targets,
            ctx.n_targets,
            k=k,
            device=ctx.device,
            desc=desc,
            qa_narrow=assets.qa_n_sweep,
            skip_mask=assets.skip_mask,
        )

    ot_k = assets.oracle_topk[:, :k].contiguous()
    # Per-row count of valid (non-padding) oracle targets: tight filters can
    # pass < k items, so the oracle pads the tail with -1. Using a flat
    # denominator of k would under-count perfect runs (e.g. 17 real items /
    # 100 → 0.17) and fold zero-target rows into a 0-recall mean.
    nt_k = (ot_k != -1).sum(dim=1).clamp(max=k)
    zero_target = nt_k == 0
    combined_skip = (
        zero_target if assets.skip_mask is None else (assets.skip_mask | zero_target)
    )
    return quality_pass_cached(
        algo_obj,
        ctx.queries,
        ot_k,
        nt_k,
        k=k,
        device=ctx.device,
        desc=desc,
        qa_narrow=assets.qa_n_sweep,
        skip_mask=combined_skip,
    )


def _autotune_prewarm(
    algo_obj: RetrievalAlgo, ctx: SweepContext, assets: FilterAssets
) -> None:
    """Call forward once per batch size before timing, so compilation and the
    per-shape cudagraph capture happen outside the timed window.

    ``reduce-overhead`` captures a cudagraph per batch-size shape; that cold cost
    has been observed collapsing a cell's median to a single ~1.5 s sample.
    """
    if not torch.cuda.is_available():
        return
    with torch.inference_mode():
        for bs in ctx.cfg.batch_sizes:
            if bs > ctx.queries.shape[0]:
                continue
            q = ctx.queries[:bs].to(ctx.device, non_blocking=True)
            kw: dict = {}
            if assets.qa_n_sweep is not None:
                kw["qa_narrow"] = assets.qa_n_sweep[:bs].to(ctx.device, non_blocking=True)
            algo_obj(q, **kw)
    torch.cuda.synchronize()


@lru_cache(maxsize=1)
def _provenance() -> dict[str, str]:
    """Hardware/software provenance stamped into every row's ``extra``.

    Computed once per process (all rows of a run share it) so cross-machine
    result pooling doesn't have to consult run logs. ``commit`` degrades to
    "unknown" outside a git checkout / without git on PATH."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    return {"gpu": gpu, "torch": torch.__version__, "commit": commit}


def _make_perf_row(
    *,
    filter_kind: FilterKind,
    sweep_name: str,
    algo: str,
    params: dict[str, Any],
    k: int,
    bs: int,
    suite: str,
    n_kept: int,
    seed: int,
    stats: PerfStats,
    index_mem: float,
    quality: QualityStats,
    backend: Backend,
) -> dict:
    """Build one perf-row dict.

    Column names are load-bearing for downstream joins: add fields, never rename
    them. ``backend`` appears both in the ``cell`` join key and as its own field."""
    return {
        "suite": suite,
        "cell": f"{filter_kind}_{sweep_name}_{backend}_bs{bs}_k{k}",
        "filter_kind": filter_kind,
        "sweep": sweep_name,
        "impl": algo,
        "backend": backend,
        "device": "cuda",
        "seed": seed,
        "batch_size": bs,
        "k": k,
        "n_users_kept": n_kept,
        **stats.as_row_fields(),
        "index_mem_mib": index_mem,
        f"recall@{k}": quality.recall,
        f"ndcg@{k}": quality.ndcg,
        f"precision@{k}": quality.precision,
        f"mrr@{k}": quality.mrr,
        "extra": {
            "params": {str(pk): str(pv) for pk, pv in params.items()},
            **_provenance(),
        },
    }


def _log_perf_line(
    filter_kind: FilterKind,
    sweep_name: str,
    algo: str,
    backend: Backend,
    params: dict[str, Any],
    k: int,
    bs: int,
    stats: PerfStats,
    quality: QualityStats,
) -> None:
    """Single-line per-cell summary."""
    params_str = (
        " " + ",".join(f"{pk}={pv}" for pk, pv in params.items()) if params else ""
    )
    logger.info(
        "{}/{}/{}[{}]{} k={} bs={} median={:.3f}ms p20={:.3f} p80={:.3f} "
        "peak={:.1f}MiB recall={:.4f} ndcg={:.4f}",
        filter_kind,
        sweep_name,
        algo,
        backend,
        params_str,
        k,
        bs,
        stats.median_ms,
        stats.p20_ms,
        stats.p80_ms,
        stats.peak_mib,
        quality.recall,
        quality.ndcg,
    )


def _release_algo(algo_obj: RetrievalAlgo) -> None:
    """Drop algo's modules and reclaim GPU pool. Called at end of every cell."""
    algo_obj.algo_modules.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = [
    "evaluate_cell",
    "run_filter_kind",
    "run_one_sweep",
    "run_sweep",
]
