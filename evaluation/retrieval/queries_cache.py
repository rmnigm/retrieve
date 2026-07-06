"""Disk cache for the SASRec encode pass.

``retrieval.cli.run_evaluation`` (``uv run run-evaluation``) spawns one
``evaluate`` subprocess per algo in a config. Without this cache each
subprocess would re-encode every query
from scratch (~1–2 min on goodreads' 313k users). Arxiv configs have
no checkpoint so they bypass the cache — their embeddings are already
disk-resident via ``load_pre_encoded_arxiv``.

Cache lives at ``<ckpt-dir>/encoded_queries_<split>.pt`` and bundles
``item_embs`` (CPU side) together with ``queries / targets / n_targets``.
Keyed by ``(ckpt mtime, max_seq_length, users_limit)``; stale entries
recompute and overwrite. ``users_limit`` is applied **before** caching
so the cache size scales with the (typically tiny) limited user set,
not the full test split — critical on 100k+ user datasets.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from loguru import logger

from retrieval.config import EvalConfig
from retrieval.loaders import load_item_and_queries, load_sasrec_embeddings

# Skip caching if the estimated cache would consume more than this fraction
# of the cache filesystem's free space. Encoding is cheap compared to filling
# the disk and crashing the wrapper mid-run; subsequent algos just re-encode.
_CACHE_FREE_FRACTION = 0.7
_CACHE_RESERVE_BYTES = 4 * 2**30  # always keep at least 4 GB free


def load_or_cache_queries(
    cfg: EvalConfig, data_path: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Path | None]:
    """Returns ``(item_embs, queries, targets, n_targets, ckpt_path)``.

    Cache hit → no model load. Cache miss → ``load_sasrec_embeddings`` runs
    the full encode pass and the result is persisted before returning.
    """
    if cfg.checkpoint is None:
        return load_item_and_queries(cfg, data_path, device)

    ckpt_path = Path(cfg.checkpoint)
    cache_path = ckpt_path.parent / f"encoded_queries_{cfg.split}.pt"
    ckpt_mtime = ckpt_path.stat().st_mtime
    max_seq = cfg.encode.max_seq_length
    users_limit = cfg.users_limit

    if cache_path.exists():
        # Blob is dict-of-tensors + scalar cache keys (see torch.save below),
        # so the safe weights_only load path handles it.
        blob = torch.load(str(cache_path), map_location="cpu", weights_only=True)
        if (
            blob.get("ckpt_mtime") == ckpt_mtime
            and blob.get("max_seq_length") == max_seq
            and blob.get("users_limit") == users_limit
        ):
            logger.info("loaded encoded_queries from cache: {}", cache_path)
            return (
                blob["item_embs"].to(device).contiguous(),
                blob["queries"],
                blob["targets"],
                blob["n_targets"],
                ckpt_path,
            )
        logger.info(
            "encoded_queries cache mismatch (ckpt_mtime/max_seq/users_limit changed); recomputing"
        )

    item_embs, queries, targets, n_targets, ckpt_path = load_sasrec_embeddings(
        cfg, data_path, device
    )
    if users_limit is not None and users_limit < queries.shape[0]:
        n = int(users_limit)
        logger.info(
            "  users_limit={}: trimming cache to {} users (was {})",
            users_limit,
            n,
            queries.shape[0],
        )
        queries = queries[:n].contiguous()
        targets = targets[:n].contiguous()
        n_targets = n_targets[:n].contiguous()
    est_bytes = sum(
        t.element_size() * t.numel()
        for t in (item_embs, queries, targets, n_targets)
    )
    free_bytes = shutil.disk_usage(str(cache_path.parent)).free
    budget_bytes = max(0, int(free_bytes * _CACHE_FREE_FRACTION) - _CACHE_RESERVE_BYTES)
    if est_bytes > budget_bytes:
        logger.warning(
            "  skipping cache write: est={:.1f}GB > budget={:.1f}GB (free={:.1f}GB); "
            "subsequent algos will re-encode",
            est_bytes / 2**30, budget_bytes / 2**30, free_bytes / 2**30,
        )
    else:
        torch.save(
            {
                "item_embs": item_embs.cpu(),
                "queries": queries,
                "targets": targets,
                "n_targets": n_targets,
                "ckpt_mtime": ckpt_mtime,
                "max_seq_length": max_seq,
                "users_limit": users_limit,
            },
            str(cache_path),
        )
        logger.info("wrote encoded_queries → {} ({:.1f}GB)", cache_path, est_bytes / 2**30)
    return item_embs, queries, targets, n_targets, ckpt_path


__all__ = ["load_or_cache_queries"]
