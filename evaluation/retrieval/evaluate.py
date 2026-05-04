"""Unified retrieval benchmark driver.

One config-driven entry point spanning yambda, goodreads, and arxiv. Dispatch
is implicit: config shape decides whether to encode queries from a SASRec
checkpoint vs load pre-encoded text embeddings, and whether to run the filter
sweep loop vs a single unfiltered cell.

| Config shape                              | Mode                                   |
|-------------------------------------------|----------------------------------------|
| `checkpoint` set, `query_emb_path` unset  | Encode queries via SASRec (yambda/gr)  |
| `query_emb_path` set, `checkpoint` unset  | Load pre-encoded text embs (arxiv)     |
| `filters: null`                           | Quality run, single unfiltered cell    |
| `filters: {clause, bloom}`                | Filter run, per-kind sweeps            |

Usage::

    uv run evaluate --config conf/500m/d128-quality.yaml
    uv run evaluate --config conf/goodreads/d128-quality.yaml
    uv run evaluate --config conf/goodreads/d128-filter.yaml
    uv run evaluate --config conf/arxiv/d256-filter.yaml
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import click
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from retrieval.algos import build_algorithm, build_filter
from retrieval.bench_primitives import (
    cuda_allocated_mib,
    encode_queries,
    load_model_for_eval,
    perf_pass_cached,
    quality_pass_cached,
)
from retrieval.config import EvalConfig, FilterCfg, FilterSweepCfg, load_eval_config
from retrieve.interfaces import FilterModule

# Per-dataset narrow clause counts. Both happen to be 5 today (genre/lang/
# format/year/author for goodreads; main_cat/license/year/n_versions/author
# for arxiv). Inferred from the loaded item_attrs_narrow.pt at runtime —
# no config field needed.

EXPECTED_DOC_PREFIX = "search_document: "
EXPECTED_QUERY_PREFIX = "search_query: "


# ----- arxiv-only helpers -----------------------------------------------------


def assert_arxiv_prefixes(content_dir: Path) -> None:
    """Catch silent prefix swaps that degrade arxiv recall by ~5–15% with no error."""
    text_meta_path = content_dir / "text_emb.meta.json"
    query_meta_path = content_dir / "query_emb.meta.json"
    if not text_meta_path.exists() or not query_meta_path.exists():
        logger.warning(
            "missing meta.json sidecars at {} — skipping prefix-drift assertion",
            content_dir,
        )
        return
    with open(text_meta_path) as f:
        text_meta = json.load(f)
    with open(query_meta_path) as f:
        query_meta = json.load(f)
    if text_meta.get("prefix") != EXPECTED_DOC_PREFIX:
        raise RuntimeError(
            f"text_emb.meta.json prefix={text_meta.get('prefix')!r} "
            f"!= expected {EXPECTED_DOC_PREFIX!r} — re-encode items"
        )
    if query_meta.get("prefix") != EXPECTED_QUERY_PREFIX:
        raise RuntimeError(
            f"query_emb.meta.json prefix={query_meta.get('prefix')!r} "
            f"!= expected {EXPECTED_QUERY_PREFIX!r} — re-encode queries"
        )
    logger.info("prefixes OK: doc={!r}, query={!r}", EXPECTED_DOC_PREFIX, EXPECTED_QUERY_PREFIX)


# ----- embeddings -------------------------------------------------------------


def load_pre_encoded_arxiv(
    cfg: EvalConfig, data_path: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Arxiv path: load `text_emb.pt` (items) and `query_emb.pt` (queries) on disk.

    Returns ``(item_embs [N+1, D] on device, queries [N_users, D] on cpu,
    targets [N_users, 1] on cpu, num_targets [N_users] on cpu)``.
    """
    content_dir = data_path / "content"
    assert_arxiv_prefixes(content_dir)

    text_emb_path = content_dir / "text_emb.pt"
    item_embs = torch.load(str(text_emb_path), map_location=device)
    item_embs[0] = 0.0
    # fp16 on disk → fp32 on device for oracle math
    item_embs = item_embs.float().contiguous()
    item_embs = F.normalize(item_embs, dim=-1)
    logger.info(
        "item_embs shape={} dtype={} (fp16 on disk → fp32 on device)",
        tuple(item_embs.shape),
        item_embs.dtype,
    )

    query_emb_path = (
        Path(cfg.query_emb_path) if cfg.query_emb_path else content_dir / "query_emb.pt"
    )
    heldout_path = data_path / "heldout.parquet"
    if not heldout_path.exists():
        raise FileNotFoundError(f"missing {heldout_path}")
    if not query_emb_path.exists():
        raise FileNotFoundError(f"missing {query_emb_path}")

    heldout = pl.read_parquet(heldout_path)
    n_users = heldout.height
    target_ids = torch.tensor(heldout["item_id"].to_list(), dtype=torch.long).unsqueeze(-1)
    n_targets = torch.ones(n_users, dtype=torch.long)

    queries = torch.load(str(query_emb_path), map_location="cpu")
    if queries.shape[0] != n_users:
        raise RuntimeError(
            f"query_emb rows={queries.shape[0]} ≠ heldout rows={n_users}; "
            "regen via `arxiv encode_queries`"
        )
    if queries.shape[1] != item_embs.shape[1]:
        raise RuntimeError(
            f"query_emb dim={queries.shape[1]} ≠ item_embs dim={item_embs.shape[1]}; "
            "encoder mismatch — re-run encode_text + encode_queries with the same truncate_dim"
        )
    queries = F.normalize(queries.float(), dim=-1)
    return item_embs, queries, target_ids, n_targets


def load_sasrec_embeddings(
    cfg: EvalConfig, data_path: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Path]:
    """Yambda / goodreads path: load SASRec checkpoint, encode every test query.

    Returns ``(item_embs, queries, targets, num_targets, ckpt_path)``.
    """
    ckpt_path = Path(cfg.checkpoint)
    with open(data_path / "item_id_map.json") as f:
        num_items = len(json.load(f))
    logger.info("num_items={}", num_items)

    model = load_model_for_eval(ckpt_path, num_items=num_items, device=device)
    item_embs = model.get_output_embeddings().weight.detach().to(device).contiguous()
    item_embs[0] = 0.0
    logger.info("item_embs shape={} dtype={}", tuple(item_embs.shape), item_embs.dtype)

    eval_parquet = data_path / f"{cfg.split}.parquet"
    queries, targets, n_targets = encode_queries(
        model,
        eval_parquet,
        max_length=cfg.encode.max_seq_length,
        encode_batch_size=cfg.encode.batch_size,
        num_workers=cfg.encode.num_workers,
        device=device,
    )
    logger.info("encoded queries: {} users, dim={}", queries.shape[0], queries.shape[1])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return item_embs, queries, targets, n_targets, ckpt_path


# ----- query attributes -------------------------------------------------------


def load_query_attrs(
    eval_split_path: Path, n_queries: int
) -> torch.Tensor | None:
    """Load per-user ``query_attrs_narrow`` from ``eval_split.parquet``.

    Returns None if the parquet is absent (then only ``filter_kind=none``
    is runnable). The wide-shelf columns in the parquet (``_1shelf`` /
    ``_2shelf``) are unused by the current bench — kept on disk for now
    in case wide-bloom sweeps come back.
    """
    if not eval_split_path.exists():
        logger.warning(
            "no eval_split.parquet at {} — only filter_kind=none is runnable",
            eval_split_path,
        )
        return None
    eval_split = pl.read_parquet(eval_split_path)
    if eval_split.height != n_queries:
        raise RuntimeError(
            f"eval_split rows={eval_split.height} ≠ queries={n_queries}; "
            "regen eval_split.parquet via the dataset CLI's `attrs` subcommand"
        )
    qa_narrow = torch.tensor(eval_split["query_attrs_narrow"].to_list(), dtype=torch.long)
    logger.info("loaded eval_split.parquet: qa_narrow={}", tuple(qa_narrow.shape))
    return qa_narrow


def build_sweep_qa(
    sweep: FilterSweepCfg,
    filter_kind: str,
    qa_narrow_all: torch.Tensor | None,
    n_clauses: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Per-sweep query attribute synthesis.

    Returns ``(qa_narrow_sweep, skip_mask)``. ``skip_mask`` is True for
    users to drop (target had no surviving narrow clauses). Inactive
    clauses are coded as ``-1`` — ExactAttributeFilter treats that as
    "match anything"; BloomFilter as "no bits queried". Reverse semantics
    live in the index, not in the query.
    """
    if filter_kind not in ("clause", "bloom") or not sweep.active_clauses:
        return None, None
    if qa_narrow_all is None:
        raise ValueError(f"sweep {sweep.name!r} needs qa_narrow but eval_split has none")
    qa_n_sweep = qa_narrow_all.clone()
    inactive_mask = torch.ones(n_clauses, dtype=torch.bool)
    for c in sweep.active_clauses:
        if not (0 <= c < n_clauses):
            raise ValueError(f"sweep {sweep.name!r}: active clause {c} out of range")
        inactive_mask[c] = False
    if inactive_mask.any():
        qa_n_sweep[:, inactive_mask] = -1
    skip_mask = (qa_n_sweep == -1).all(dim=1)
    return qa_n_sweep, skip_mask


# ----- ground-truth oracle (filter sweeps only) -------------------------------


def load_filter_assets(
    filter_kind: str,
    fcfg: FilterCfg,
    data_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Load ``(item_attrs_narrow, clause_is_reverse)`` for a filter_kind.

    Both `clause` and `bloom` filter_kinds run over the same narrow
    attribute tensor; only the algo on top differs. The wide-shelf
    tensor is currently unused — see goodreads-filter-eval.md.
    """
    item_attrs_narrow: torch.Tensor | None = None
    clause_is_reverse: torch.Tensor | None = None

    if filter_kind in ("clause", "bloom"):
        if fcfg.attrs_path is None:
            raise ValueError(f"filter_kind={filter_kind} requires attrs_path")
        item_attrs_narrow = torch.load(
            str(resolve_path(data_dir, fcfg.attrs_path)), map_location=device
        )
        if fcfg.reverse_path:
            clause_is_reverse = torch.load(
                str(resolve_path(data_dir, fcfg.reverse_path)), map_location=device
            )

    return item_attrs_narrow, clause_is_reverse


def resolve_path(data_dir: Path, path_str: str) -> Path:
    """Treat YAML paths as cwd-relative if they point at a real file; otherwise
    resolve them against ``data_dir`` (so ``data/<dataset>/foo.pt`` works
    whether the harness runs from `evaluation/` or anywhere else)."""
    p = Path(path_str).expanduser()
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    return (data_dir / p.name).resolve()


@torch.inference_mode()
def compute_filtered_oracle(
    item_embs: torch.Tensor,
    queries: torch.Tensor,
    qa_narrow_sweep: torch.Tensor | None,
    skip_mask: torch.Tensor | None,
    filter_mod: FilterModule | None,
    K_GT: int,
    *,
    batch_size: int = 64,
    device: torch.device,
) -> torch.Tensor:
    """Brute-force filtered FullScan: returns ``[N_users, K_GT]`` int64 ids.

    Skipped rows get all -1. `id 0` is masked out (padding row of `item_embs`).
    ``filter_mod`` must be an *exact* mask source — i.e. ``ExactAttributeFilter``
    even on bloom-suite runs, so bloom's false positives do not leak
    into the ground truth.
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
        mask = (
            filter_mod.evaluate_mask(qa_n)
            if (filter_mod is not None and qa_n is not None)
            else None
        )
        scores = q @ item_embs_t
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        scores[:, 0] = float("-inf")
        topk = torch.topk(scores, K_eff, dim=1)
        # When the filter passes fewer than K_eff items, the bottom slots tie
        # at -inf and torch.topk picks the lowest-indexed padding items
        # (0, 1, 2, ...). Force those to -1 so they don't get scored as real
        # ground-truth candidates against the algos' -1 padding.
        topk_ids = torch.where(
            torch.isfinite(topk.values),
            topk.indices,
            torch.full_like(topk.indices, -1),
        )
        out[batch_idx, :K_eff] = topk_ids.cpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ----- param sweep ------------------------------------------------------------


def expand_param_combos(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """List-valued entries become sweep axes (cartesian product); scalars stay
    fixed. Empty dict yields ``[{}]`` so callers can iterate uniformly.
    """
    swept_keys = [k for k, v in raw.items() if isinstance(v, list)]
    fixed = {k: v for k, v in raw.items() if not isinstance(v, list)}
    if not swept_keys:
        return [dict(fixed)]
    swept_vals = [raw[k] for k in swept_keys]
    return [
        {**fixed, **dict(zip(swept_keys, combo, strict=True))}
        for combo in itertools.product(*swept_vals)
    ]


def is_valid_combo(algo: str, params: dict[str, Any]) -> bool:
    """Skip combos the underlying algo would assert on."""
    if algo == "silvertorch":
        n_lists = params.get("n_lists")
        n_probe = params.get("n_probe")
        if n_lists is not None and n_probe is not None and n_probe > n_lists:
            return False
    return True


# ----- driver -----------------------------------------------------------------


@click.command()
@click.option("--config", "config_path", type=str, required=True)
@click.option("--algorithms", "algos_override", multiple=True, type=str, default=())
@click.option(
    "--filter-kind",
    "filter_kinds",
    multiple=True,
    type=str,
    default=(),
    help="Restrict to one or more filter_kinds (repeat the flag); empty = run all.",
)
@click.option("--sweep", "sweep_filter", type=str, default=None)
@click.option("--output", "output_override", type=str, default=None)
@click.option(
    "--skip-quality",
    is_flag=True,
    default=False,
    help="Skip the bs=1 quality stream and report recall=ndcg=NaN. "
    "Perf timing rows are still emitted. Useful for fast latency/memory sweeps.",
)
def main(
    config_path: str,
    algos_override: tuple[str, ...],
    filter_kinds: tuple[str, ...],
    sweep_filter: str | None,
    output_override: str | None,
    skip_quality: bool,
) -> None:
    cfg = load_eval_config(Path(config_path))
    if algos_override:
        cfg.algorithms = list(algos_override)
    if output_override:
        cfg.output = output_override

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        # Disable TF32 so the oracle (cuBLAS `q @ E_t`) and the algos
        # (per-impl Triton GEMM, generally fp32) compute scores in the
        # same precision. Otherwise exact-mask algos like
        # `linr_v1_filter_mask` show ~1e-3 recall drift vs the oracle
        # on narrow filters where top-K boundaries land on items
        # within TF32's 10-bit mantissa noise band.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    data_path = Path(cfg.data_dir)
    dev = torch.device(cfg.device)

    # 1. Load items + queries — branch on config shape.
    use_pre_encoded = cfg.query_emb_path is not None or cfg.checkpoint is None
    ckpt_path: Path | None = None
    if use_pre_encoded:
        item_embs, queries, targets, n_targets = load_pre_encoded_arxiv(cfg, data_path, dev)
    else:
        item_embs, queries, targets, n_targets, ckpt_path = load_sasrec_embeddings(
            cfg, data_path, dev
        )

    # 2. Default output path.
    if cfg.output:
        out_path = Path(cfg.output)
    elif ckpt_path is not None:
        out_path = ckpt_path.parent / "evaluate.json"
    else:
        out_path = data_path / "evaluate.json"

    # 3. Optional per-query attributes (None for yambda).
    qa_narrow_all: torch.Tensor | None = None
    if cfg.filters is not None:
        qa_narrow_all = load_query_attrs(
            data_path / "eval_split.parquet", queries.shape[0]
        )

    # 4. Run the sweep.
    rows = run_sweep(
        cfg,
        item_embs,
        queries,
        targets,
        n_targets,
        qa_narrow_all,
        data_path=data_path,
        device=dev,
        filter_kinds=filter_kinds,
        sweep_filter=sweep_filter,
        skip_quality=skip_quality,
    )

    # 5. Write JSON.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("wrote {} rows to {}", len(rows), out_path)


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
    """Loop over (filter_kind, sweep, algo, k, batch_size) and emit rows.

    Yambda (cfg.filters is None) iterates a single synthetic
    ``("none", FilterSweepCfg(name="full_scan"))`` cell so the loop body
    stays uniform. Filtered datasets (goodreads/arxiv) iterate the
    configured filter sweeps; both clause and bloom filter_kinds run
    over the same narrow attribute tensor — only the algo differs.
    """
    rows: list[dict] = []
    K_GT = max(cfg.ks)
    suite = "yambda" if cfg.filters is None else "filter"

    if cfg.filters is None:
        filter_iter: list[tuple[str, FilterCfg]] = [
            ("none", FilterCfg(sweeps=[FilterSweepCfg(name="full_scan")]))
        ]
    else:
        filter_iter = list(cfg.filters.items())

    n_clauses = 0  # set when we first load item_attrs_narrow

    gt_dir = data_path / "gt"
    if cfg.filters is not None:
        gt_dir.mkdir(parents=True, exist_ok=True)

    for filter_kind, fcfg in filter_iter:
        if filter_kinds and filter_kind not in filter_kinds:
            continue

        # Optional subsample for filter sweeps only (goodreads has 313k test
        # users; the bs=1 quality stream is wall-clock-dominant). Unfiltered
        # cell keeps all users so the yambda parity gate still holds.
        if (
            filter_kind != "none"
            and cfg.filter_users_limit is not None
            and cfg.filter_users_limit < queries.shape[0]
        ):
            n_keep = int(cfg.filter_users_limit)
            queries_f = queries[:n_keep].contiguous()
            targets_f = targets[:n_keep].contiguous()
            n_targets_f = n_targets[:n_keep].contiguous()
            qa_narrow_f = qa_narrow_all[:n_keep] if qa_narrow_all is not None else None
            logger.info(
                "  filter_users_limit={}: subsampling {}→{} users for filter sweeps",
                n_keep,
                queries.shape[0],
                n_keep,
            )
        else:
            queries_f = queries
            targets_f = targets
            n_targets_f = n_targets
            qa_narrow_f = qa_narrow_all

        filter_mod: FilterModule | None = None
        oracle_filter: FilterModule | None = None
        item_attrs_narrow = clause_is_reverse = None
        if cfg.filters is not None:
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
            # Oracle always uses exact ExactAttributeFilter semantics, even on
            # filter_kind="bloom" — bloom's false positives must NOT leak
            # into the ground truth. On filter_kind="clause" the wired
            # filter_mod is already an ExactAttributeFilter; reuse it. On
            # bloom we build a separate exact filter over the same attrs.
            if filter_kind == "clause":
                oracle_filter = filter_mod
            elif filter_kind == "bloom":
                oracle_filter = build_filter(
                    "clause",
                    item_attrs_narrow=item_attrs_narrow,
                    clause_is_reverse=clause_is_reverse,
                    device=device,
                )
            if filter_mod is not None:
                logger.info(
                    "  filter module built: {} (oracle: {})",
                    type(filter_mod).__name__,
                    type(oracle_filter).__name__ if oracle_filter is not None else "none",
                )
            if item_attrs_narrow is not None:
                n_clauses = int(item_attrs_narrow.shape[1])

        for sweep in fcfg.sweeps:
            if sweep_filter and sweep.name != sweep_filter:
                continue
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

            # Build (and cache) the filtered-FullScan oracle for filtered cells.
            # Skipped when --skip-quality is on: oracle is only used to score
            # recall, never for perf timing or skip-mask synthesis.
            oracle_topk: torch.Tensor | None = None
            if cfg.filters is not None and filter_kind != "none" and not skip_quality:
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
                        queries_f,
                        qa_n_sweep,
                        skip_mask,
                        oracle_filter,
                        K_GT=K_GT,
                        device=device,
                    )
                    torch.save(oracle_topk, str(gt_path))
                    logger.info("  saved oracle → {}", gt_path)

            for algo in cfg.algorithms:
                raw_params = cfg.algo_params.get(algo, {})
                combos = expand_param_combos(raw_params)
                for params in combos:
                    if not is_valid_combo(algo, params):
                        logger.warning("skipping invalid combo {}: {}", algo, params)
                        continue

                    for k in cfg.ks:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                        mem_before = cuda_allocated_mib()

                        try:
                            algo_obj = build_algorithm(
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
                            continue

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        index_mem = 0.0 if algo_obj.is_cpu else cuda_allocated_mib() - mem_before

                        if skip_quality:
                            recall, ndcg = float("nan"), float("nan")
                        elif cfg.filters is None or filter_kind == "none":
                            recall, ndcg = quality_pass_cached(
                                algo_obj.forward,
                                queries_f,
                                targets_f,
                                n_targets_f,
                                k=k,
                                device=device,
                                desc=f"{filter_kind}/{sweep.name}/{algo} k={k}",
                                qa_narrow=qa_n_sweep,
                                skip_mask=skip_mask,
                            )
                        else:
                            assert oracle_topk is not None
                            ot_k = oracle_topk[:, :k].contiguous()
                            nt_k = torch.full((n_users,), k, dtype=torch.long)
                            recall, ndcg = quality_pass_cached(
                                algo_obj.forward,
                                queries_f,
                                ot_k,
                                nt_k,
                                k=k,
                                device=device,
                                desc=f"{filter_kind}/{sweep.name}/{algo} k={k}",
                                qa_narrow=qa_n_sweep,
                                skip_mask=skip_mask,
                            )

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
                            row = {
                                "suite": suite,
                                "cell": f"{filter_kind}_{sweep.name}_bs{bs}_k{k}",
                                "filter_kind": filter_kind,
                                "sweep": sweep.name,
                                "impl": algo,
                                "device": "cpu" if algo_obj.is_cpu else "cuda",
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
                                f"recall@{k}": recall,
                                f"ndcg@{k}": ndcg,
                                "extra": {
                                    "params": {str(pk): str(pv) for pk, pv in params.items()},
                                },
                            }
                            rows.append(row)
                            logger.info(
                                "{}/{}/{} k={} bs={} median={:.3f}ms p20={:.3f} p80={:.3f} "
                                "peak={:.1f}MiB recall={:.4f} ndcg={:.4f}",
                                filter_kind,
                                sweep.name,
                                algo,
                                k,
                                bs,
                                med,
                                p20,
                                p80,
                                peak,
                                recall,
                                ndcg,
                            )
                        algo_obj.modules.clear()
                        del algo_obj
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        if cfg.filters is not None:
            del filter_mod, oracle_filter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return rows


if __name__ == "__main__":
    main()


__all__ = ["main"]
