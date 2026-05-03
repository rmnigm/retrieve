"""Goodreads filter-bench harness — peer of the yambda
[benchmark.py](benchmark.py).

Runs the full `(filter_kind, sweep, algo, k)` matrix from a YAML config
([conf/goodreads-d*.yaml](../conf/)). For each filtered cell, recall@K /
NDCG@K are computed against a **filtered-FullScan top-K_GT oracle**
(cached per sweep at `<data_dir>/gt/gt_topk_<sweep>.pt`) — the bench
question is "how does this ANN's filtered top-K compare to a brute-force
filtered top-K?" rather than "did we find the held-out target", since
the dataset has no filtered ground truth.

`filter_kind="none"` is the cross-check sweep — it uses the held-out
test target like yambda, so its `recall@K` for `torch_fullscan` should
match the yambda regression number to ±0.0005 (verification step #2 in
[goodreads-filter-eval.md](../../docs/plans/goodreads-filter-eval.md)).

Usage::

    cd /workspace/retrieve/evaluation
    uv run python -m retrieval.eval_goodreads_retrieval \\
      --config conf/goodreads-d128.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import polars as pl
import torch
from loguru import logger
from tqdm import tqdm

from retrieval._bench_primitives import (
    cuda_allocated_mib,
    encode_queries,
    load_model_for_eval,
    perf_pass_cached,
    quality_pass_cached,
)
from retrieval.config import FilterCfg, FilterSweepCfg, load_eval_config
from retrieval.filter_registry import (
    SilvertorchSkippedOnNarrow,
    build_filter_modules,
    build_filtered_algorithm,
    synthesize_query_attrs_narrow,
)
from retrieve.layers.filters import combine_masks


N_NARROW_CLAUSES = 5  # genre, lang, format, year, author


# ----- per-sweep query attribute synthesis -----------------------------------


def _build_sweep_qa(
    sweep: FilterSweepCfg,
    filter_kind: str,
    qa_narrow_all: torch.Tensor | None,  # [N_users, C] from eval_split.parquet
    qa_wide_1shelf: torch.Tensor | None,  # [N_users] int64
    qa_wide_2shelf: torch.Tensor | None,  # [N_users, 2] int64
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Returns ``(qa_narrow_sweep, qa_wide_sweep, skip_mask)``.

    `qa_*_sweep` are the per-user attribute tensors for this sweep (or
    None if the sweep doesn't activate that filter side); `skip_mask` is
    a [N_users] bool — True for rows to drop from this sweep (e.g. wide
    eval rows where the target has zero surviving shelves).
    """
    qa_n_sweep: torch.Tensor | None = None
    qa_w_sweep: torch.Tensor | None = None
    skip_mask: torch.Tensor | None = None

    if filter_kind in ("clause", "combined") and sweep.active_clauses:
        if qa_narrow_all is None:
            raise ValueError(f"sweep {sweep.name!r} needs qa_narrow but eval_split has none")
        qa_n_sweep = synthesize_query_attrs_narrow(
            qa_narrow_all, sweep, n_clauses=N_NARROW_CLAUSES
        )
        # Skip users with all-(-1) narrow query (target had no usable attr)
        sweep_skip = (qa_n_sweep == -1).all(dim=1)
        skip_mask = sweep_skip if skip_mask is None else (skip_mask | sweep_skip)

    if filter_kind in ("bloom", "combined") and sweep.query_attrs_field:
        field = sweep.query_attrs_field
        if field == "query_attrs_wide_1shelf":
            if qa_wide_1shelf is None:
                raise ValueError("sweep references query_attrs_wide_1shelf but none loaded")
            qa_w_sweep = qa_wide_1shelf.unsqueeze(-1)
        elif field == "query_attrs_wide_2shelf":
            if qa_wide_2shelf is None:
                raise ValueError("sweep references query_attrs_wide_2shelf but none loaded")
            qa_w_sweep = qa_wide_2shelf
        else:
            raise ValueError(f"unknown query_attrs_field: {field}")
        sweep_skip = (qa_w_sweep == -1).any(dim=-1)
        skip_mask = sweep_skip if skip_mask is None else (skip_mask | sweep_skip)

    return qa_n_sweep, qa_w_sweep, skip_mask


# ----- ground-truth oracle ---------------------------------------------------


@torch.inference_mode()
def compute_filtered_oracle(
    item_embs: torch.Tensor,  # [N+1, D] on device
    queries: torch.Tensor,  # [N_users, D] on cpu
    qa_narrow_sweep: torch.Tensor | None,  # [N_users, C] cpu
    qa_wide_sweep: torch.Tensor | None,  # [N_users, M] cpu
    skip_mask: torch.Tensor | None,
    ci,  # ClauseIndex | None (on device, pre-built)
    bf,  # BloomFilter | None (on device, pre-built)
    K_GT: int,
    *,
    batch_size: int = 64,
    device: torch.device,
) -> torch.Tensor:
    """Brute-force filtered FullScan: returns ``[N_users, K_GT]`` int64 ids.

    Skipped rows get all -1. `id 0` is masked out (padding row of `item_embs`).
    Only the active filter side(s) for the sweep are evaluated, so
    no-filter cells should not call this — they use held-out targets.
    """
    n_users = queries.shape[0]
    out = torch.full((n_users, K_GT), -1, dtype=torch.long)
    keep = ~skip_mask if skip_mask is not None else torch.ones(n_users, dtype=torch.bool)
    keep_idx = keep.nonzero(as_tuple=False).reshape(-1)
    if keep_idx.numel() == 0:
        return out

    item_embs_t = item_embs.t().contiguous()
    n_total = int(item_embs.shape[0])
    K_eff = min(K_GT, n_total)

    for s in tqdm(range(0, keep_idx.numel(), batch_size), desc="oracle", leave=False):
        batch_idx = keep_idx[s : s + batch_size]
        q = queries[batch_idx].to(device, non_blocking=True)
        qa_n = (
            qa_narrow_sweep[batch_idx].to(device, non_blocking=True)
            if qa_narrow_sweep is not None
            else None
        )
        qa_w = (
            qa_wide_sweep[batch_idx].to(device, non_blocking=True)
            if qa_wide_sweep is not None
            else None
        )
        masks: list[torch.Tensor | None] = []
        if ci is not None and qa_n is not None:
            masks.append(ci.evaluate_mask(qa_n))
        if bf is not None and qa_w is not None:
            for j in range(qa_w.shape[1]):
                masks.append(bf.evaluate_mask(qa_w[:, j : j + 1]))
        mask = combine_masks(*masks)
        scores = q @ item_embs_t  # [B, N+1]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        # Item 0 is padding — exclude.
        scores[:, 0] = float("-inf")
        topk_ids = torch.topk(scores, K_eff, dim=1).indices  # [B, K_eff]
        out[batch_idx, :K_eff] = topk_ids.cpu()

    if ci is not None:
        del ci
    if bf is not None:
        del bf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ----- driver ----------------------------------------------------------------


def _load_filter_assets(
    filter_kind: str,
    fcfg: FilterCfg,
    data_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Loads ``(item_attrs_narrow, item_attrs_wide, clause_is_reverse)`` for
    a filter_kind. Paths come from the config. None for sides this kind
    doesn't use.
    """
    item_attrs_narrow: torch.Tensor | None = None
    item_attrs_wide: torch.Tensor | None = None
    clause_is_reverse: torch.Tensor | None = None

    if filter_kind == "clause":
        if fcfg.attrs_path is None:
            raise ValueError(f"filter_kind={filter_kind} requires attrs_path")
        item_attrs_narrow = torch.load(
            str(_resolve(data_dir, fcfg.attrs_path)), map_location=device
        )
        if fcfg.reverse_path:
            clause_is_reverse = torch.load(
                str(_resolve(data_dir, fcfg.reverse_path)), map_location=device
            )
    elif filter_kind == "bloom":
        if fcfg.attrs_path is None:
            raise ValueError(f"filter_kind={filter_kind} requires attrs_path")
        item_attrs_wide = torch.load(
            str(_resolve(data_dir, fcfg.attrs_path)), map_location=device
        )
    elif filter_kind == "combined":
        if fcfg.attrs_narrow is None or fcfg.attrs_wide is None:
            raise ValueError(
                f"filter_kind=combined requires attrs_narrow and attrs_wide"
            )
        item_attrs_narrow = torch.load(
            str(_resolve(data_dir, fcfg.attrs_narrow)), map_location=device
        )
        item_attrs_wide = torch.load(
            str(_resolve(data_dir, fcfg.attrs_wide)), map_location=device
        )
        if fcfg.reverse_path:
            clause_is_reverse = torch.load(
                str(_resolve(data_dir, fcfg.reverse_path)), map_location=device
            )

    return item_attrs_narrow, item_attrs_wide, clause_is_reverse


def _resolve(data_dir: Path, path_str: str) -> Path:
    """Treat YAML paths as cwd-relative if absolute-or-cwd-relative paths
    point at a real file; otherwise resolve them against ``data_dir``."""
    p = Path(path_str).expanduser()
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    return (data_dir / p.name).resolve()


@click.command()
@click.option(
    "--config",
    "config_path",
    type=str,
    required=True,
    help="YAML config path (see conf/goodreads-d*.yaml).",
)
@click.option(
    "--algorithms",
    "algos_override",
    multiple=True,
    type=str,
    default=(),
    help="Replace (do not merge into) the YAML's algorithms list.",
)
@click.option(
    "--filter-kind",
    "filter_kind_filter",
    type=str,
    default=None,
    help="Run only this filter_kind (e.g. clause). Defaults to all.",
)
@click.option(
    "--sweep",
    "sweep_filter",
    type=str,
    default=None,
    help="Run only this sweep name. Defaults to all.",
)
@click.option(
    "--output",
    "output_override",
    type=str,
    default=None,
    help="Override the YAML's output path.",
)
def main(
    config_path: str,
    algos_override: tuple[str, ...],
    filter_kind_filter: str | None,
    sweep_filter: str | None,
    output_override: str | None,
) -> None:
    cfg = load_eval_config(Path(config_path))
    if algos_override:
        cfg.algorithms = list(algos_override)
    if output_override:
        cfg.output = output_override
    if cfg.filters is None:
        raise click.UsageError("config has no `filters:` block; this is the goodreads harness")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    ckpt_path = Path(cfg.checkpoint)
    data_path = Path(cfg.data_dir)
    dev = torch.device(cfg.device)

    with open(data_path / "item_id_map.json") as f:
        num_items = len(json.load(f))
    logger.info("num_items={}", num_items)

    model = load_model_for_eval(ckpt_path, num_items=num_items, device=dev)
    item_embs = model.get_output_embeddings().weight.detach().to(dev).contiguous()
    item_embs[0] = 0.0
    logger.info("item_embs shape={} dtype={}", tuple(item_embs.shape), item_embs.dtype)

    eval_parquet = data_path / f"{cfg.split}.parquet"
    eval_split_path = data_path / "eval_split.parquet"
    out_path = (
        Path(cfg.output)
        if cfg.output
        else ckpt_path.parent / "eval_goodreads_retrieval.json"
    )
    gt_dir = data_path / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    queries, targets, n_targets = encode_queries(
        model,
        eval_parquet,
        max_length=cfg.encode.max_seq_length,
        encode_batch_size=cfg.encode.batch_size,
        num_workers=cfg.encode.num_workers,
        device=dev,
    )
    logger.info("encoded queries: {} users, dim={}", queries.shape[0], queries.shape[1])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----- load eval_split.parquet (per-user query attrs) -------------------
    qa_narrow_all: torch.Tensor | None = None
    qa_wide_1: torch.Tensor | None = None
    qa_wide_2: torch.Tensor | None = None
    if eval_split_path.exists():
        eval_split = pl.read_parquet(eval_split_path)
        if eval_split.height != queries.shape[0]:
            raise RuntimeError(
                f"eval_split rows={eval_split.height} ≠ test rows={queries.shape[0]}; "
                "regen eval_split.parquet via `goodreads.py attrs`"
            )
        qa_narrow_all = torch.tensor(
            eval_split["query_attrs_narrow"].to_list(), dtype=torch.long
        )
        qa_wide_1 = torch.tensor(
            eval_split["query_attrs_wide_1shelf"].to_list(), dtype=torch.long
        )
        qa_wide_2 = torch.tensor(
            eval_split["query_attrs_wide_2shelf"].to_list(), dtype=torch.long
        )
        logger.info(
            "loaded eval_split.parquet: qa_narrow={} qa_wide_1={} qa_wide_2={}",
            tuple(qa_narrow_all.shape), tuple(qa_wide_1.shape), tuple(qa_wide_2.shape),
        )
    else:
        logger.warning("no eval_split.parquet at {} — only filter_kind=none is runnable", eval_split_path)

    rows: list[dict] = []
    K_GT = max(cfg.ks)

    for filter_kind, fcfg in cfg.filters.items():
        if filter_kind_filter and filter_kind != filter_kind_filter:
            continue

        # ---- filter assets for this kind (shared across all cells) ----
        item_attrs_narrow, item_attrs_wide, clause_is_reverse = _load_filter_assets(
            filter_kind, fcfg, data_path, dev
        )
        ci, bf = build_filter_modules(
            filter_kind,
            item_attrs_narrow=item_attrs_narrow,
            item_attrs_wide=item_attrs_wide,
            clause_is_reverse=clause_is_reverse,
            bloom_m_bits=fcfg.m_bits,
            bloom_k_hash=fcfg.k_hash,
            device=dev,
        )
        if ci is not None or bf is not None:
            logger.info(
                "  filter modules built: ClauseIndex={} BloomFilter={}",
                ci is not None,
                bf is not None,
            )

        for sweep in fcfg.sweeps:
            if sweep_filter and sweep.name != sweep_filter:
                continue
            logger.info("=== filter_kind={} sweep={} ===", filter_kind, sweep.name)

            # ---- per-sweep query attrs ----
            if filter_kind == "none":
                qa_n_sweep = None
                qa_w_sweep = None
                skip_mask = None
            else:
                qa_n_sweep, qa_w_sweep, skip_mask = _build_sweep_qa(
                    sweep, filter_kind, qa_narrow_all, qa_wide_1, qa_wide_2
                )

            n_users = queries.shape[0]
            n_kept = (
                int((~skip_mask).sum().item()) if skip_mask is not None else n_users
            )
            logger.info("  kept users: {} / {}", n_kept, n_users)

            # ---- oracle (cached per-sweep) for filtered cells ----
            oracle_topk: torch.Tensor | None = None
            if filter_kind != "none":
                gt_path = gt_dir / f"gt_topk_{sweep.name}.pt"
                if gt_path.exists():
                    oracle_topk = torch.load(str(gt_path), map_location="cpu")
                    if oracle_topk.shape != (n_users, K_GT):
                        logger.warning(
                            "stale oracle at {} (shape={}); recomputing",
                            gt_path,
                            tuple(oracle_topk.shape),
                        )
                        oracle_topk = None
                if oracle_topk is None:
                    logger.info("  building filtered oracle (K_GT={})", K_GT)
                    oracle_topk = compute_filtered_oracle(
                        item_embs,
                        queries,
                        qa_n_sweep,
                        qa_w_sweep,
                        skip_mask,
                        ci,
                        bf,
                        K_GT=K_GT,
                        device=dev,
                    )
                    torch.save(oracle_topk, str(gt_path))
                    logger.info("  saved oracle → {}", gt_path)

            # ---- per-algo loop ----
            for algo in cfg.algorithms:
                algo_params = cfg.algo_params.get(algo, {})

                # Per-(algo, k) build matches yambda benchmark.py — silvertorch
                # k-means runs once per k, but the FilterModules are reused
                # across all cells of the kind (built above).
                index_mem = 0.0
                first_build = True
                for k in cfg.ks:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    mem_before_k = cuda_allocated_mib()
                    try:
                        forward_k, modules_k, is_cpu = build_filtered_algorithm(
                            algo,
                            item_embs,
                            k=k,
                            filter_kind=filter_kind,
                            sweep=sweep,
                            ci=ci,
                            bf=bf,
                            clause_is_reverse=clause_is_reverse,
                            item_attrs_wide=item_attrs_wide,
                            bloom_m_bits=fcfg.m_bits,
                            bloom_k_hash=fcfg.k_hash,
                            algo_params=algo_params,
                        )
                    except SilvertorchSkippedOnNarrow:
                        if first_build:
                            logger.info(
                                "  skipping silvertorch on narrow sweep {}", sweep.name
                            )
                            rows.append(
                                {
                                    "suite": "goodreads_filter",
                                    "cell": f"{filter_kind}_{sweep.name}",
                                    "filter_kind": filter_kind,
                                    "sweep": sweep.name,
                                    "impl": algo,
                                    "skipped": True,
                                    "reason": "silvertorch_skipped_on_narrow",
                                }
                            )
                        first_build = False
                        continue
                    first_build = False
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    index_mem = 0.0 if is_cpu else cuda_allocated_mib() - mem_before_k

                    # ---- quality ----
                    if filter_kind == "none":
                        # Held-out test target — matches yambda harness.
                        recall, ndcg = quality_pass_cached(
                            forward_k,
                            queries,
                            targets,
                            n_targets,
                            k=k,
                            device=dev,
                            desc=f"{filter_kind}/{sweep.name}/{algo} k={k}",
                            qa_narrow=qa_n_sweep,
                            qa_wide=qa_w_sweep,
                            skip_mask=skip_mask,
                        )
                    else:
                        assert oracle_topk is not None
                        ot_k = oracle_topk[:, :k].contiguous()
                        nt_k = torch.full((n_users,), k, dtype=torch.long)
                        recall, ndcg = quality_pass_cached(
                            forward_k,
                            queries,
                            ot_k,
                            nt_k,
                            k=k,
                            device=dev,
                            desc=f"{filter_kind}/{sweep.name}/{algo} k={k}",
                            qa_narrow=qa_n_sweep,
                            qa_wide=qa_w_sweep,
                            skip_mask=skip_mask,
                        )

                    # ---- perf ----
                    for bs in cfg.batch_sizes:
                        med, p20, p80, peak, scratch = perf_pass_cached(
                            forward_k,
                            queries,
                            batch_size=bs,
                            device=dev,
                            is_cpu=is_cpu,
                            seed=cfg.seed,
                            qa_narrow=qa_n_sweep,
                            qa_wide=qa_w_sweep,
                            skip_mask=skip_mask,
                        )
                        rows.append(
                            {
                                "suite": "goodreads_filter",
                                "cell": f"{filter_kind}_{sweep.name}_bs{bs}_k{k}",
                                "filter_kind": filter_kind,
                                "sweep": sweep.name,
                                "impl": algo,
                                "device": "cpu" if is_cpu else "cuda",
                                "seed": cfg.seed,
                                "batch_size": bs,
                                "k": k,
                                "n_users_kept": n_kept,
                                "median_ms": med,
                                "p20_ms": p20,
                                "p80_ms": p80,
                                "peak_mem_mib": peak,
                                "index_mem_mib": index_mem,
                                "fwd_scratch_mib": scratch,
                                "recall_kind": "held_out_target"
                                if filter_kind == "none"
                                else "filtered_fullscan_oracle",
                                f"recall@{k}": recall,
                                f"ndcg@{k}": ndcg,
                                "extra": {
                                    "params": {
                                        str(pk): str(pv)
                                        for pk, pv in algo_params.items()
                                    },
                                    "fullscan_post_filter": (
                                        algo == "torch_fullscan"
                                        and filter_kind != "none"
                                    ),
                                },
                            }
                        )
                        logger.info(
                            "{}/{}/{} k={} bs={} median={:.3f}ms p20={:.3f} p80={:.3f} "
                            "peak={:.1f}MiB recall={:.4f} ndcg={:.4f}",
                            filter_kind, sweep.name, algo, k, bs,
                            med, p20, p80, peak, recall, ndcg,
                        )
                    # Each (algo, k) cell gets its own forward; drop refs
                    # before the next cell's mem_before snapshot.
                    modules_k.clear()
                    del forward_k, modules_k
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Drop FilterModules between filter_kinds — bloom_sigs is ~12 MiB at
        # m_bits=1024 / 1.5M items, but ClauseIndex's attrs buffer can be
        # several hundred MiB.
        del ci, bf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("wrote {} rows to {}", len(rows), out_path)


if __name__ == "__main__":
    main()
