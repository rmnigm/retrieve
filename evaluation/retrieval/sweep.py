"""Nested loop driver for the retrieval benchmark.

Decomposes the cross-product
``(filter_kind, sweep, algo, params, k, batch_size)`` into one function per
loop level, so each piece reads top-to-bottom and can be reasoned about in
isolation.

```
run_sweep                    # pin globals, warm GPU, dispatch
└─ run_filter_kind           # subsample users, build filter modules
   └─ run_one_sweep          # synth qa, build/load oracle, iterate algos
      └─ evaluate_cell       # build algo, quality, prewarm, per-bs perf rows
```

Yambda (``cfg.filters is None``) iterates a single synthetic
``("none", FilterSweepCfg(name="full_scan"))`` cell so the loop body stays
uniform. Filtered datasets (goodreads / arxiv) iterate the configured
filter sweeps; both ``clause`` and ``bloom`` filter_kinds run over the same
narrow attribute tensor — only the algo on top differs.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from retrieval.algos import build_algorithm, build_filter
from retrieval.bench_tools import (
    cuda_allocated_mib,
    perf_pass_cached,
    pin_precision_globals,
    quality_pass_cached,
    warm_gpu_once,
)
from retrieval.config import EvalConfig, FilterCfg, FilterSweepCfg
from retrieval.loaders import build_sweep_qa, load_filter_assets
from retrieval.oracle import load_or_build_oracle
from retrieve.interfaces import FilterModule

# ----- top-level driver -------------------------------------------------------


def run_sweep(
    cfg: EvalConfig,
    item_embs: torch.Tensor,
    queries: torch.Tensor,
    targets: torch.Tensor,
    n_targets: torch.Tensor,
    qa_narrow_all: torch.Tensor | None,
    *,
    data_path: Path,
    device: torch.device,
    filter_kinds: tuple[str, ...] = (),
    sweep_filter: str | None = None,
    skip_quality: bool = False,
) -> list[dict]:
    """Loop over (filter_kind, sweep, algo, k, batch_size) and emit rows."""
    # Pin TF32/matmul-precision so two runs on the same box don't silently
    # diverge depending on what code earlier in the process touched these
    # globals. Pre-warm the GPU once so cuBLAS-init / kernel-load / pinned-mem
    # one-time costs are paid outside any cell's measured window — without
    # this, whichever cell runs first absorbs them and reports inflated time.
    pin_precision_globals()
    warm_gpu_once(device)

    K_GT = max(cfg.ks)
    suite = "yambda" if cfg.filters is None else "filter"
    gt_dir = data_path / cfg.gt_subdir
    if cfg.filters is not None:
        gt_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for filter_kind, fcfg in _select_filter_iter(cfg, filter_kinds):
        rows.extend(
            run_filter_kind(
                filter_kind,
                fcfg,
                cfg=cfg,
                item_embs=item_embs,
                queries=queries,
                targets=targets,
                n_targets=n_targets,
                qa_narrow_all=qa_narrow_all,
                data_path=data_path,
                device=device,
                sweep_filter=sweep_filter,
                skip_quality=skip_quality,
                gt_dir=gt_dir,
                K_GT=K_GT,
                suite=suite,
            )
        )
    return rows


def _select_filter_iter(
    cfg: EvalConfig, filter_kinds: tuple[str, ...]
) -> Iterator[tuple[str, FilterCfg]]:
    """Yield ``(filter_kind, FilterCfg)`` for each enabled filter kind.

    Yambda inserts a single synthetic ``("none", …)`` cell. ``--filter-kind``
    on the CLI restricts which kinds are yielded; empty tuple = all.
    """
    if cfg.filters is None:
        yield "none", FilterCfg(sweeps=[FilterSweepCfg(name="full_scan")])
        return
    for filter_kind, fcfg in cfg.filters.items():
        if filter_kinds and filter_kind not in filter_kinds:
            continue
        yield filter_kind, fcfg


# ----- per filter_kind --------------------------------------------------------


def run_filter_kind(
    filter_kind: str,
    fcfg: FilterCfg,
    *,
    cfg: EvalConfig,
    item_embs: torch.Tensor,
    queries: torch.Tensor,
    targets: torch.Tensor,
    n_targets: torch.Tensor,
    qa_narrow_all: torch.Tensor | None,
    data_path: Path,
    device: torch.device,
    sweep_filter: str | None,
    skip_quality: bool,
    gt_dir: Path,
    K_GT: int,
    suite: str,
) -> list[dict]:
    """Subsample users, build filter+oracle modules, iterate sweeps."""
    queries_f, targets_f, n_targets_f, qa_narrow_f = _apply_users_limit(
        cfg, queries, targets, n_targets, qa_narrow_all
    )
    filter_mod, oracle_filter, item_attrs_narrow, n_clauses = _build_filter_modules(
        filter_kind, fcfg, cfg, data_path, device
    )

    rows: list[dict] = []
    for sweep in fcfg.sweeps:
        if sweep_filter and sweep.name != sweep_filter:
            continue
        rows.extend(
            run_one_sweep(
                sweep,
                filter_kind,
                cfg=cfg,
                item_embs=item_embs,
                queries_f=queries_f,
                targets_f=targets_f,
                n_targets_f=n_targets_f,
                qa_narrow_f=qa_narrow_f,
                filter_mod=filter_mod,
                oracle_filter=oracle_filter,
                item_attrs_narrow=item_attrs_narrow,
                n_clauses=n_clauses,
                gt_dir=gt_dir,
                K_GT=K_GT,
                suite=suite,
                device=device,
                skip_quality=skip_quality,
            )
        )

    if cfg.filters is not None:
        del filter_mod, oracle_filter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _apply_users_limit(
    cfg: EvalConfig,
    queries: torch.Tensor,
    targets: torch.Tensor,
    n_targets: torch.Tensor,
    qa_narrow_all: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Apply ``cfg.users_limit`` uniformly to quality and filter tensors.

    Goodreads has 313k test users; the bs=1 quality stream is the wall-clock
    bottleneck so capping speeds runs up substantially.
    """
    if cfg.users_limit is None or cfg.users_limit >= queries.shape[0]:
        return queries, targets, n_targets, qa_narrow_all
    n_keep = int(cfg.users_limit)
    logger.info(
        "  users_limit={}: subsampling {}→{} users",
        n_keep,
        queries.shape[0],
        n_keep,
    )
    return (
        queries[:n_keep].contiguous(),
        targets[:n_keep].contiguous(),
        n_targets[:n_keep].contiguous(),
        qa_narrow_all[:n_keep] if qa_narrow_all is not None else None,
    )


def _build_filter_modules(
    filter_kind: str,
    fcfg: FilterCfg,
    cfg: EvalConfig,
    data_path: Path,
    device: torch.device,
) -> tuple[FilterModule | None, FilterModule | None, torch.Tensor | None, int]:
    """Build the index-side filter and the oracle-side exact filter.

    On ``filter_kind="clause"`` the wired filter is already an exact filter,
    so the oracle reuses it. On ``"bloom"`` we build a separate
    ``ExactAttributeFilter`` over the same attrs — bloom's false positives
    must NOT leak into the ground truth.
    """
    if cfg.filters is None:
        return None, None, None, 0

    item_attrs_narrow, clause_is_reverse = load_filter_assets(
        filter_kind, fcfg, data_path, device
    )
    filter_mod = build_filter(
        filter_kind,
        item_attrs_narrow=item_attrs_narrow,
        clause_is_reverse=clause_is_reverse,
        bloom_m_bits=fcfg.m_bits,
        bloom_k_hash=fcfg.k_hash,
        device=device,
    )
    if filter_kind == "clause":
        oracle_filter: FilterModule | None = filter_mod
    elif filter_kind == "bloom":
        oracle_filter = build_filter(
            "clause",
            item_attrs_narrow=item_attrs_narrow,
            clause_is_reverse=clause_is_reverse,
            device=device,
        )
    else:
        oracle_filter = None
    if filter_mod is not None:
        logger.info(
            "  filter module built: {} (oracle: {})",
            type(filter_mod).__name__,
            type(oracle_filter).__name__ if oracle_filter is not None else "none",
        )
    n_clauses = int(item_attrs_narrow.shape[1]) if item_attrs_narrow is not None else 0
    return filter_mod, oracle_filter, item_attrs_narrow, n_clauses


# ----- per sweep --------------------------------------------------------------


def run_one_sweep(
    sweep: FilterSweepCfg,
    filter_kind: str,
    *,
    cfg: EvalConfig,
    item_embs: torch.Tensor,
    queries_f: torch.Tensor,
    targets_f: torch.Tensor,
    n_targets_f: torch.Tensor,
    qa_narrow_f: torch.Tensor | None,
    filter_mod: FilterModule | None,
    oracle_filter: FilterModule | None,
    item_attrs_narrow: torch.Tensor | None,
    n_clauses: int,
    gt_dir: Path,
    K_GT: int,
    suite: str,
    device: torch.device,
    skip_quality: bool,
) -> list[dict]:
    """Synthesise per-sweep qa, load/build oracle, iterate (algo, params, k)."""
    logger.info("=== filter_kind={} sweep={} ===", filter_kind, sweep.name)

    if cfg.filters is None or filter_kind == "none":
        qa_n_sweep = skip_mask = None
    else:
        qa_n_sweep, skip_mask = build_sweep_qa(
            sweep, filter_kind, qa_narrow_f, n_clauses
        )

    n_users = queries_f.shape[0]
    n_kept = int((~skip_mask).sum().item()) if skip_mask is not None else n_users
    logger.info("  kept users: {} / {}", n_kept, n_users)

    # Skipped when --skip-quality is on: oracle is only used to score recall,
    # never for perf timing or skip-mask synthesis.
    oracle_topk: torch.Tensor | None = None
    if cfg.filters is not None and filter_kind != "none" and not skip_quality:
        oracle_topk = load_or_build_oracle(
            gt_dir,
            sweep.name,
            n_users,
            K_GT,
            item_embs=item_embs,
            queries=queries_f,
            qa_narrow_sweep=qa_n_sweep,
            skip_mask=skip_mask,
            oracle_filter=oracle_filter,
            device=device,
        )

    rows: list[dict] = []
    for algo in cfg.algorithms:
        for params in expand_param_combos(cfg.algo_params.get(algo, [{}])):
            if not is_valid_combo(algo, params):
                logger.warning("skipping invalid combo {}: {}", algo, params)
                continue
            for k in cfg.ks:
                rows.extend(
                    evaluate_cell(
                        algo,
                        params,
                        k,
                        sweep,
                        filter_kind,
                        cfg=cfg,
                        item_embs=item_embs,
                        queries_f=queries_f,
                        targets_f=targets_f,
                        n_targets_f=n_targets_f,
                        qa_n_sweep=qa_n_sweep,
                        skip_mask=skip_mask,
                        oracle_topk=oracle_topk,
                        filter_mod=filter_mod,
                        item_attrs_narrow=item_attrs_narrow,
                        n_kept=n_kept,
                        suite=suite,
                        device=device,
                        skip_quality=skip_quality,
                    )
                )
    return rows


# ----- per cell ---------------------------------------------------------------


def evaluate_cell(
    algo: str,
    params: dict[str, Any],
    k: int,
    sweep: FilterSweepCfg,
    filter_kind: str,
    *,
    cfg: EvalConfig,
    item_embs: torch.Tensor,
    queries_f: torch.Tensor,
    targets_f: torch.Tensor,
    n_targets_f: torch.Tensor,
    qa_n_sweep: torch.Tensor | None,
    skip_mask: torch.Tensor | None,
    oracle_topk: torch.Tensor | None,
    filter_mod: FilterModule | None,
    item_attrs_narrow: torch.Tensor | None,
    n_kept: int,
    suite: str,
    device: torch.device,
    skip_quality: bool,
) -> list[dict]:
    """Build the algo, score quality, prewarm autotune, time each batch size."""
    _reset_cuda_state_for_cell()
    mem_before = cuda_allocated_mib()

    algo_obj = _try_build_algo(
        algo,
        item_embs,
        k=k,
        filter_kind=filter_kind,
        filter_mod=filter_mod,
        item_attrs_narrow=item_attrs_narrow,
        params=params,
    )
    if algo_obj is None:
        return []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    index_mem = 0.0 if algo_obj.is_cpu else cuda_allocated_mib() - mem_before

    recall, ndcg = _run_quality(
        algo_obj,
        k,
        sweep,
        filter_kind,
        algo,
        cfg=cfg,
        queries_f=queries_f,
        targets_f=targets_f,
        n_targets_f=n_targets_f,
        qa_n_sweep=qa_n_sweep,
        skip_mask=skip_mask,
        oracle_topk=oracle_topk,
        device=device,
        skip_quality=skip_quality,
    )

    _autotune_prewarm(algo_obj, queries_f, qa_n_sweep, cfg.batch_sizes, device)

    rows: list[dict] = []
    for bs in cfg.batch_sizes:
        med, p20, p80, peak, scratch = perf_pass_cached(
            algo_obj.forward,
            queries_f,
            batch_size=bs,
            device=device,
            is_cpu=algo_obj.is_cpu,
            seed=cfg.seed,
            qa_narrow=qa_n_sweep,
            skip_mask=skip_mask,
        )
        rows.append(
            _make_perf_row(
                filter_kind=filter_kind,
                sweep_name=sweep.name,
                algo=algo,
                params=params,
                k=k,
                bs=bs,
                suite=suite,
                is_cpu=algo_obj.is_cpu,
                n_kept=n_kept,
                seed=cfg.seed,
                med=med,
                p20=p20,
                p80=p80,
                peak=peak,
                index_mem=index_mem,
                scratch=scratch,
                recall=recall,
                ndcg=ndcg,
            )
        )
        _log_perf_line(
            filter_kind, sweep.name, algo, params, k, bs, med, p20, p80, peak, recall, ndcg
        )

    _release_algo(algo_obj)
    return rows


# ----- per-cell helpers -------------------------------------------------------


def _reset_cuda_state_for_cell() -> None:
    """Sync, drop pooled memory, reset peak — so per-cell index_mem is clean."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _try_build_algo(
    algo: str,
    item_embs: torch.Tensor,
    *,
    k: int,
    filter_kind: str,
    filter_mod: FilterModule | None,
    item_attrs_narrow: torch.Tensor | None,
    params: dict[str, Any],
) -> Any | None:
    """Wrap ``build_algorithm`` with the ``ValueError → skip cell`` contract.

    Algos like ``linr_v2`` raise on ``filter_kind="none"``; the caller treats
    that as "this combo isn't applicable" and moves on.
    """
    try:
        return build_algorithm(
            algo,
            item_embs,
            k=k,
            filter_kind=filter_kind,
            filter_mod=filter_mod,
            item_attrs_narrow=item_attrs_narrow,
            params=params,
        )
    except ValueError as e:
        logger.debug("  skipping {}: {}", algo, e)
        return None


def _run_quality(
    algo_obj: Any,
    k: int,
    sweep: FilterSweepCfg,
    filter_kind: str,
    algo: str,
    *,
    cfg: EvalConfig,
    queries_f: torch.Tensor,
    targets_f: torch.Tensor,
    n_targets_f: torch.Tensor,
    qa_n_sweep: torch.Tensor | None,
    skip_mask: torch.Tensor | None,
    oracle_topk: torch.Tensor | None,
    device: torch.device,
    skip_quality: bool,
) -> tuple[float, float]:
    """Dispatch quality pass: NaN, held-out targets, or oracle top-K."""
    if skip_quality:
        return float("nan"), float("nan")

    desc = f"{filter_kind}/{sweep.name}/{algo} k={k}"
    if cfg.filters is None or filter_kind == "none":
        return quality_pass_cached(
            algo_obj.forward,
            queries_f,
            targets_f,
            n_targets_f,
            k=k,
            device=device,
            desc=desc,
            qa_narrow=qa_n_sweep,
            skip_mask=skip_mask,
        )

    assert oracle_topk is not None
    ot_k = oracle_topk[:, :k].contiguous()
    # Per-row count of valid (non-padding) oracle targets: tight filters can
    # pass < k items, so the oracle pads the tail with -1. Using a flat
    # denominator of k would under-count perfect runs (e.g. 17 real items /
    # 100 → 0.17) and fold zero-target rows into a 0-recall mean.
    nt_k = (ot_k != -1).sum(dim=1).clamp(max=k)
    zero_target = nt_k == 0
    combined_skip = zero_target if skip_mask is None else (skip_mask | zero_target)
    return quality_pass_cached(
        algo_obj.forward,
        queries_f,
        ot_k,
        nt_k,
        k=k,
        device=device,
        desc=desc,
        qa_narrow=qa_n_sweep,
        skip_mask=combined_skip,
    )


def _autotune_prewarm(
    algo_obj: Any,
    queries_f: torch.Tensor,
    qa_n_sweep: torch.Tensor | None,
    batch_sizes: list[int],
    device: torch.device,
) -> None:
    """Call forward once per batch size before any bs is timed.

    Per-bs warmup inside ``perf_pass_cached`` is supposed to populate Triton's
    autotune cache, but a slow last-mile autotune config has been observed
    leaking into the timing window (median collapsing to a single ~1.5 s
    sample on the first cell). Belt-and-suspenders defense.
    """
    if algo_obj.is_cpu or not torch.cuda.is_available():
        return
    with torch.inference_mode():
        for bs in batch_sizes:
            if bs > queries_f.shape[0]:
                continue
            q = queries_f[:bs].to(device, non_blocking=True)
            kw: dict = {}
            if qa_n_sweep is not None:
                kw["qa_narrow"] = qa_n_sweep[:bs].to(device, non_blocking=True)
            algo_obj.forward(q, **kw)
    torch.cuda.synchronize()


def _make_perf_row(
    *,
    filter_kind: str,
    sweep_name: str,
    algo: str,
    params: dict[str, Any],
    k: int,
    bs: int,
    suite: str,
    is_cpu: bool,
    n_kept: int,
    seed: int,
    med: float,
    p20: float,
    p80: float,
    peak: float,
    index_mem: float,
    scratch: float,
    recall: float,
    ndcg: float,
) -> dict:
    """Build one perf-row dict. Keys/order MUST match the pre-refactor schema."""
    return {
        "suite": suite,
        "cell": f"{filter_kind}_{sweep_name}_bs{bs}_k{k}",
        "filter_kind": filter_kind,
        "sweep": sweep_name,
        "impl": algo,
        "device": "cpu" if is_cpu else "cuda",
        "seed": seed,
        "batch_size": bs,
        "k": k,
        "n_users_kept": n_kept,
        "median_ms": med,
        "p20_ms": p20,
        "p80_ms": p80,
        "peak_mem_mib": peak,
        "index_mem_mib": index_mem,
        "fwd_scratch_mib": scratch,
        f"recall@{k}": recall,
        f"ndcg@{k}": ndcg,
        "extra": {"params": {str(pk): str(pv) for pk, pv in params.items()}},
    }


def _log_perf_line(
    filter_kind: str,
    sweep_name: str,
    algo: str,
    params: dict[str, Any],
    k: int,
    bs: int,
    med: float,
    p20: float,
    p80: float,
    peak: float,
    recall: float,
    ndcg: float,
) -> None:
    """Single-line per-cell summary. Format MUST match pre-refactor logs."""
    params_str = (
        " " + ",".join(f"{pk}={pv}" for pk, pv in params.items()) if params else ""
    )
    logger.info(
        "{}/{}/{}{} k={} bs={} median={:.3f}ms p20={:.3f} p80={:.3f} "
        "peak={:.1f}MiB recall={:.4f} ndcg={:.4f}",
        filter_kind,
        sweep_name,
        algo,
        params_str,
        k,
        bs,
        med,
        p20,
        p80,
        peak,
        recall,
        ndcg,
    )


def _release_algo(algo_obj: Any) -> None:
    """Drop algo's modules and reclaim GPU pool. Called at end of every cell."""
    algo_obj.modules.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----- param-combo utilities (used by run_one_sweep and external tests) -------


def expand_param_combos(combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``algo_params[algo]`` is a list of dicts; each dict is one explicit
    combo, taken as-is. Algos without a params entry iterate over ``[{}]``
    so the caller's loop stays uniform.
    """
    return [dict(combo) for combo in combos]


def is_valid_combo(algo: str, params: dict[str, Any]) -> bool:
    """Skip combos the underlying algo would assert on."""
    if algo == "silvertorch":
        n_lists = params.get("n_lists")
        n_probe = params.get("n_probe")
        if n_lists is not None and n_probe is not None and n_probe > n_lists:
            return False
    return True


__all__ = [
    "evaluate_cell",
    "expand_param_combos",
    "is_valid_combo",
    "run_filter_kind",
    "run_one_sweep",
    "run_sweep",
]
