"""Unified retrieval benchmark driver.

One config-driven entry point spanning yambda, goodreads, and arxiv. Dispatch
is implicit: config shape decides whether to encode queries from a SASRec
checkpoint vs load pre-encoded text embeddings, and whether to run the filter
sweep loop vs a single unfiltered cell.

| Config shape                              | Mode                                   |
|-------------------------------------------|----------------------------------------|
| `checkpoint` set, `query_emb_path` unset  | Encode queries via SASRec (yambda/gr)  |
| `query_emb_path` set, `checkpoint` unset  | Load pre-encoded text embs (arxiv)     |
| `filters: null`                           | No filter loop (yambda)                |
| `filters: {none, clause, bloom, combined}`| Filter sweeps (goodreads, arxiv)       |

Usage::

    uv run evaluate --config conf/500m-d128.yaml
    uv run evaluate --config conf/goodreads-d128-drop0.5-id.yaml
    uv run evaluate --config conf/arxiv-d256.yaml
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

from retrieval.algo_registry import (
    SilvertorchSkippedOnNarrow,
    build_algorithm,
    build_filter_modules,
    build_filtered_algorithm,
    synthesize_query_attrs_narrow,
)
from retrieval.bench_primitives import (
    cuda_allocated_mib,
    encode_queries,
    load_model_for_eval,
    perf_pass_cached,
    quality_pass_cached,
)
from retrieval.config import EvalConfig, FilterCfg, FilterSweepCfg, load_eval_config
from retrieve.layers.filters import combine_masks

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
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Load per-user `(qa_narrow, qa_wide_1shelf, qa_wide_2shelf)` from
    `eval_split.parquet`. Returns triple of `None` if file is absent.
    """
    if not eval_split_path.exists():
        logger.warning(
            "no eval_split.parquet at {} — only filter_kind=none is runnable",
            eval_split_path,
        )
        return None, None, None
    eval_split = pl.read_parquet(eval_split_path)
    if eval_split.height != n_queries:
        raise RuntimeError(
            f"eval_split rows={eval_split.height} ≠ queries={n_queries}; "
            "regen eval_split.parquet via the dataset CLI's `attrs` subcommand"
        )
    qa_narrow = torch.tensor(eval_split["query_attrs_narrow"].to_list(), dtype=torch.long)
    qa_wide_1 = torch.tensor(eval_split["query_attrs_wide_1shelf"].to_list(), dtype=torch.long)
    qa_wide_2 = torch.tensor(eval_split["query_attrs_wide_2shelf"].to_list(), dtype=torch.long)
    logger.info(
        "loaded eval_split.parquet: qa_narrow={} qa_wide_1={} qa_wide_2={}",
        tuple(qa_narrow.shape),
        tuple(qa_wide_1.shape),
        tuple(qa_wide_2.shape),
    )
    return qa_narrow, qa_wide_1, qa_wide_2


def build_sweep_qa(
    sweep: FilterSweepCfg,
    filter_kind: str,
    qa_narrow_all: torch.Tensor | None,
    qa_wide_1: torch.Tensor | None,
    qa_wide_2: torch.Tensor | None,
    n_clauses: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Per-sweep query attribute synthesis.

    Returns ``(qa_narrow_sweep, qa_wide_sweep, skip_mask)``. `skip_mask` is
    True for users to drop (target had no surviving narrow clauses or empty
    wide bag).
    """
    qa_n_sweep: torch.Tensor | None = None
    qa_w_sweep: torch.Tensor | None = None
    skip_mask: torch.Tensor | None = None

    if filter_kind in ("clause", "combined") and sweep.active_clauses:
        if qa_narrow_all is None:
            raise ValueError(f"sweep {sweep.name!r} needs qa_narrow but eval_split has none")
        qa_n_sweep = synthesize_query_attrs_narrow(qa_narrow_all, sweep, n_clauses=n_clauses)
        sweep_skip = (qa_n_sweep == -1).all(dim=1)
        skip_mask = sweep_skip if skip_mask is None else (skip_mask | sweep_skip)

    if filter_kind in ("bloom", "combined") and sweep.query_attrs_field:
        field = sweep.query_attrs_field
        if field == "query_attrs_wide_1shelf":
            if qa_wide_1 is None:
                raise ValueError("sweep references query_attrs_wide_1shelf but none loaded")
            qa_w_sweep = qa_wide_1.unsqueeze(-1)
        elif field == "query_attrs_wide_2shelf":
            if qa_wide_2 is None:
                raise ValueError("sweep references query_attrs_wide_2shelf but none loaded")
            qa_w_sweep = qa_wide_2
        else:
            raise ValueError(f"unknown query_attrs_field: {field}")
        sweep_skip = (qa_w_sweep == -1).any(dim=-1)
        skip_mask = sweep_skip if skip_mask is None else (skip_mask | sweep_skip)

    return qa_n_sweep, qa_w_sweep, skip_mask


# ----- ground-truth oracle (filter sweeps only) -------------------------------


def load_filter_assets(
    filter_kind: str,
    fcfg: FilterCfg,
    data_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Load ``(item_attrs_narrow, item_attrs_wide, clause_is_reverse)`` for a
    filter_kind. Paths come from the config; None for sides this kind doesn't
    use.
    """
    item_attrs_narrow: torch.Tensor | None = None
    item_attrs_wide: torch.Tensor | None = None
    clause_is_reverse: torch.Tensor | None = None

    if filter_kind == "clause":
        if fcfg.attrs_path is None:
            raise ValueError(f"filter_kind={filter_kind} requires attrs_path")
        item_attrs_narrow = torch.load(
            str(resolve_path(data_dir, fcfg.attrs_path)), map_location=device
        )
        if fcfg.reverse_path:
            clause_is_reverse = torch.load(
                str(resolve_path(data_dir, fcfg.reverse_path)), map_location=device
            )
    elif filter_kind == "bloom":
        if fcfg.attrs_path is None:
            raise ValueError(f"filter_kind={filter_kind} requires attrs_path")
        item_attrs_wide = torch.load(
            str(resolve_path(data_dir, fcfg.attrs_path)), map_location=device
        )
    elif filter_kind == "combined":
        if fcfg.attrs_narrow is None or fcfg.attrs_wide is None:
            raise ValueError("filter_kind=combined requires attrs_narrow and attrs_wide")
        item_attrs_narrow = torch.load(
            str(resolve_path(data_dir, fcfg.attrs_narrow)), map_location=device
        )
        item_attrs_wide = torch.load(
            str(resolve_path(data_dir, fcfg.attrs_wide)), map_location=device
        )
        if fcfg.reverse_path:
            clause_is_reverse = torch.load(
                str(resolve_path(data_dir, fcfg.reverse_path)), map_location=device
            )

    return item_attrs_narrow, item_attrs_wide, clause_is_reverse


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
    qa_wide_sweep: torch.Tensor | None,
    skip_mask: torch.Tensor | None,
    ci,
    bf,
    K_GT: int,
    *,
    batch_size: int = 64,
    device: torch.device,
) -> torch.Tensor:
    """Brute-force filtered FullScan: returns ``[N_users, K_GT]`` int64 ids.

    Skipped rows get all -1. `id 0` is masked out (padding row of `item_embs`).
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
        scores = q @ item_embs_t
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        scores[:, 0] = float("-inf")
        topk_ids = torch.topk(scores, K_eff, dim=1).indices
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


# ----- algo eligibility -------------------------------------------------------


def algo_skip(filter_kind: str, algo: str) -> str | None:
    """Return a non-empty reason string if `(filter_kind, algo)` should skip;
    None otherwise.

    The two rules:
    * filter_kind == "none" + linr_v2_filter_compact → no candidate source.
    * filter_kind != "none" + torch_fullscan → equals the filtered-FullScan
      oracle by construction (recall=1.0); ``linr_v2_filter_compact`` covers
      the same exact-filter cell with a more relevant perf profile.
    """
    if filter_kind == "none" and algo == "linr_v2_filter_compact":
        return "linr_v2_filter_compact requires a filter"
    if filter_kind != "none" and algo == "torch_fullscan":
        return "torch_fullscan equals filtered-FullScan oracle on filter sweeps"
    return None


# ----- driver -----------------------------------------------------------------


@click.command()
@click.option("--config", "config_path", type=str, required=True)
@click.option("--algorithms", "algos_override", multiple=True, type=str, default=())
@click.option("--filter-kind", "filter_kind_filter", type=str, default=None)
@click.option("--sweep", "sweep_filter", type=str, default=None)
@click.option("--output", "output_override", type=str, default=None)
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

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

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
    qa_wide_1: torch.Tensor | None = None
    qa_wide_2: torch.Tensor | None = None
    if cfg.filters is not None:
        qa_narrow_all, qa_wide_1, qa_wide_2 = load_query_attrs(
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
        qa_wide_1,
        qa_wide_2,
        data_path=data_path,
        device=dev,
        filter_kind_filter=filter_kind_filter,
        sweep_filter=sweep_filter,
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
    qa_wide_1: torch.Tensor | None,
    qa_wide_2: torch.Tensor | None,
    *,
    data_path: Path,
    device: torch.device,
    filter_kind_filter: str | None = None,
    sweep_filter: str | None = None,
) -> list[dict]:
    """Loop over (filter_kind, sweep, algo, k, batch_size) and emit rows.

    Yambda (cfg.filters is None) iterates a single synthetic
    ``("none", FilterSweepCfg(name="full_scan"))`` cell so the loop body
    stays uniform; that path uses `build_algorithm` directly. Filtered
    datasets (goodreads/arxiv) iterate the actual filter sweeps and use
    `build_filtered_algorithm`.
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
        if filter_kind_filter and filter_kind != filter_kind_filter:
            continue

        ci = bf = None
        item_attrs_narrow = item_attrs_wide = clause_is_reverse = None
        if cfg.filters is not None:
            item_attrs_narrow, item_attrs_wide, clause_is_reverse = load_filter_assets(
                filter_kind, fcfg, data_path, device
            )
            ci, bf = build_filter_modules(
                filter_kind,
                item_attrs_narrow=item_attrs_narrow,
                item_attrs_wide=item_attrs_wide,
                clause_is_reverse=clause_is_reverse,
                bloom_m_bits=fcfg.m_bits,
                bloom_k_hash=fcfg.k_hash,
                device=device,
            )
            if ci is not None or bf is not None:
                logger.info(
                    "  filter modules built: ClauseIndex={} BloomFilter={}",
                    ci is not None,
                    bf is not None,
                )
            if item_attrs_narrow is not None:
                n_clauses = int(item_attrs_narrow.shape[1])

        for sweep in fcfg.sweeps:
            if sweep_filter and sweep.name != sweep_filter:
                continue
            logger.info("=== filter_kind={} sweep={} ===", filter_kind, sweep.name)

            if cfg.filters is None or filter_kind == "none":
                qa_n_sweep = qa_w_sweep = skip_mask = None
            else:
                qa_n_sweep, qa_w_sweep, skip_mask = build_sweep_qa(
                    sweep, filter_kind, qa_narrow_all, qa_wide_1, qa_wide_2, n_clauses
                )

            n_users = queries.shape[0]
            n_kept = int((~skip_mask).sum().item()) if skip_mask is not None else n_users
            logger.info("  kept users: {} / {}", n_kept, n_users)

            # Build (and cache) the filtered-FullScan oracle for filtered cells.
            oracle_topk: torch.Tensor | None = None
            if cfg.filters is not None and filter_kind != "none":
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
                        device=device,
                    )
                    torch.save(oracle_topk, str(gt_path))
                    logger.info("  saved oracle → {}", gt_path)

            for algo in cfg.algorithms:
                reason = algo_skip(filter_kind, algo)
                if reason is not None:
                    logger.debug("  skipping {}: {}", algo, reason)
                    continue

                raw_params = cfg.algo_params.get(algo, {})
                combos = expand_param_combos(raw_params)
                for params in combos:
                    if not is_valid_combo(algo, params):
                        logger.warning("skipping invalid combo {}: {}", algo, params)
                        continue

                    first_build = True
                    for k in cfg.ks:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                        mem_before = cuda_allocated_mib()

                        try:
                            if cfg.filters is None:
                                forward, modules, is_cpu = build_algorithm(
                                    algo, item_embs, k=k, params=params
                                )
                            else:
                                forward, modules, is_cpu = build_filtered_algorithm(
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
                                    algo_params=params,
                                )
                        except SilvertorchSkippedOnNarrow:
                            if first_build:
                                logger.info(
                                    "  skipping silvertorch on narrow sweep {}", sweep.name
                                )
                                rows.append(
                                    {
                                        "suite": suite,
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
                        index_mem = 0.0 if is_cpu else cuda_allocated_mib() - mem_before

                        if cfg.filters is None or filter_kind == "none":
                            recall, ndcg = quality_pass_cached(
                                forward,
                                queries,
                                targets,
                                n_targets,
                                k=k,
                                device=device,
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
                                forward,
                                queries,
                                ot_k,
                                nt_k,
                                k=k,
                                device=device,
                                desc=f"{filter_kind}/{sweep.name}/{algo} k={k}",
                                qa_narrow=qa_n_sweep,
                                qa_wide=qa_w_sweep,
                                skip_mask=skip_mask,
                            )

                        for bs in cfg.batch_sizes:
                            med, p20, p80, peak, scratch = perf_pass_cached(
                                forward,
                                queries,
                                batch_size=bs,
                                device=device,
                                is_cpu=is_cpu,
                                seed=cfg.seed,
                                qa_narrow=qa_n_sweep,
                                qa_wide=qa_w_sweep,
                                skip_mask=skip_mask,
                            )
                            row = {
                                "suite": suite,
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
                        modules.clear()
                        del forward, modules
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        if cfg.filters is not None:
            del ci, bf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return rows


if __name__ == "__main__":
    main()


__all__ = ["main"]
