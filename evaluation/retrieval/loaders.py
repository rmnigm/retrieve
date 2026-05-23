"""Disk and config-driven loaders for the retrieval benchmark.

Two query/item-embedding paths feed the same downstream sweep:

| Config shape                              | Loader                       |
|-------------------------------------------|------------------------------|
| `checkpoint` set, `query_emb_path` unset  | ``load_sasrec_embeddings``   |
| `query_emb_path` set, `checkpoint` unset  | ``load_pre_encoded_arxiv``   |

``load_item_and_queries`` is the single entry the driver calls — it
dispatches on cfg shape so the driver doesn't have to.

Filter sweeps additionally need per-query attributes (``load_query_attrs``)
and per-item attribute tensors (``load_filter_assets``); both live here so
all parquet/torch.load calls are in one place.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger

from retrieval.bench_tools import encode_queries, load_model_for_eval
from retrieval.config import EvalConfig, FilterCfg, FilterSweepCfg

EXPECTED_DOC_PREFIX = "search_document: "
EXPECTED_QUERY_PREFIX = "search_query: "


# ----- path utility -----------------------------------------------------------


def resolve_path(data_dir: Path, path_str: str) -> Path:
    """Treat YAML paths as cwd-relative if they point at a real file; otherwise
    resolve them against ``data_dir`` (so ``data/<dataset>/foo.pt`` works
    whether the harness runs from ``evaluation/`` or anywhere else)."""
    p = Path(path_str).expanduser()
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    return (data_dir / p.name).resolve()


# ----- arxiv-only sanity check ------------------------------------------------


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


# ----- embedding loaders ------------------------------------------------------


def _load_sharded_text_emb(
    shard_index_path: Path, device: torch.device
) -> torch.Tensor:
    """Reassemble ``content/text_emb_shard_*.pt`` into one ``[N+1, D]`` tensor.

    Synth catalogs from ``evaluation/datasets/synth_arxiv.py`` write the item
    embeddings sharded so the per-file size stays under torch's implicit
    serialization ceilings; the sidecar ``shard_index.json`` lists shard
    offsets and lengths.
    """
    with open(shard_index_path) as f:
        idx = json.load(f)
    n = int(idx["n_items_plus_one"])
    d = int(idx["dim"])
    dtype = getattr(torch, idx["dtype"])
    if n > 100_000_000 and device.type == "cuda":
        logger.warning(
            "shard_index reports n={:,} > 100M; this catalog needs ~{:.0f} GB "
            "on device at {} ({} bytes per element). If load OOMs, switch to "
            "the int8/1-bit path described in docs/plans/linr-int8-quantization.md.",
            n, n * d * dtype.itemsize / 1e9, idx["dtype"], dtype.itemsize,
        )
    out = torch.empty((n, d), dtype=dtype, device=device)
    content_dir = shard_index_path.parent
    for s in idx["shards"]:
        shard_path = content_dir / s["filename"]
        start = int(s["start_id"])
        n_rows = int(s["n_rows"])
        shard = torch.load(str(shard_path), map_location=device)
        if shard.shape != (n_rows, d):
            raise RuntimeError(
                f"{shard_path}: shape {tuple(shard.shape)} != ({n_rows}, {d})"
            )
        out[start : start + n_rows].copy_(shard)
        del shard
    logger.info(
        "loaded sharded text_emb: {} shards → [{}, {}] dtype={}",
        len(idx["shards"]), n, d, idx["dtype"],
    )
    return out


def load_pre_encoded_arxiv(
    cfg: EvalConfig, data_path: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Arxiv path: load ``text_emb.pt`` (items) and ``query_emb.pt`` (queries) on disk.

    Detects sharded synth catalogs via ``content/shard_index.json`` and
    reassembles them on the fly. Otherwise loads the single-file layout
    written by the upstream arxiv ETL.

    Returns ``(item_embs [N+1, D] on device, queries [N_users, D] on cpu,
    targets [N_users, 1] on cpu, num_targets [N_users] on cpu)``.
    """
    content_dir = data_path / cfg.content_subdir
    assert_arxiv_prefixes(content_dir)

    shard_index_path = content_dir / "shard_index.json"
    if shard_index_path.exists():
        item_embs = _load_sharded_text_emb(shard_index_path, device)
    else:
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
            "encoder mismatch — re-run encode_text + encode_queries with matching truncate_dims"
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


def load_item_and_queries(
    cfg: EvalConfig, data_path: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Path | None]:
    """Dispatch on ``cfg`` shape: pre-encoded arxiv vs SASRec checkpoint.

    Returns ``(item_embs, queries, targets, n_targets, ckpt_path_or_None)``.
    Output paths in the driver default to ``ckpt_path.parent / evaluate.json``
    when a checkpoint is involved; arxiv runs return ``None`` and fall back
    to ``data_dir``.
    """
    use_pre_encoded = cfg.query_emb_path is not None or cfg.checkpoint is None
    if use_pre_encoded:
        item_embs, queries, targets, n_targets = load_pre_encoded_arxiv(cfg, data_path, device)
        return item_embs, queries, targets, n_targets, None
    return load_sasrec_embeddings(cfg, data_path, device)


# ----- query attributes -------------------------------------------------------


def load_query_attrs(eval_split_path: Path, n_queries: int) -> torch.Tensor | None:
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


# ----- filter assets ----------------------------------------------------------


def load_filter_assets(
    filter_kind: str,
    fcfg: FilterCfg,
    data_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Load ``(item_attrs_narrow, clause_is_reverse)`` for a filter_kind.

    Both ``clause`` and ``bloom`` filter_kinds run over the same narrow
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


__all__ = [
    "assert_arxiv_prefixes",
    "build_sweep_qa",
    "load_filter_assets",
    "load_item_and_queries",
    "load_pre_encoded_arxiv",
    "load_query_attrs",
    "load_sasrec_embeddings",
    "resolve_path",
]
