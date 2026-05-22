"""Disk cache for the SASRec encode pass.

``run_per_algo.sh`` spawns one ``evaluate`` subprocess per algo in a
config. Without this cache each subprocess would re-encode every query
from scratch (~1–2 min on goodreads' 313k users). Arxiv configs have
no checkpoint so they bypass the cache — their embeddings are already
disk-resident via ``load_pre_encoded_arxiv``.

Cache lives at ``<ckpt-dir>/encoded_queries_<split>.pt`` and bundles
``item_embs`` (CPU side) together with ``queries / targets / n_targets``.
Keyed by ``(ckpt mtime, max_seq_length)``; stale entries recompute and
overwrite.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from retrieval.config import EvalConfig
from retrieval.loaders import load_item_and_queries, load_sasrec_embeddings


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

    if cache_path.exists():
        blob = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        if blob["ckpt_mtime"] == ckpt_mtime and blob["max_seq_length"] == max_seq:
            logger.info("loaded encoded_queries from cache: {}", cache_path)
            return (
                blob["item_embs"].to(device).contiguous(),
                blob["queries"],
                blob["targets"],
                blob["n_targets"],
                ckpt_path,
            )

    item_embs, queries, targets, n_targets, ckpt_path = load_sasrec_embeddings(
        cfg, data_path, device
    )
    torch.save(
        {
            "item_embs": item_embs.cpu(),
            "queries": queries,
            "targets": targets,
            "n_targets": n_targets,
            "ckpt_mtime": ckpt_mtime,
            "max_seq_length": max_seq,
        },
        str(cache_path),
    )
    logger.info("wrote encoded_queries → {}", cache_path)
    return item_embs, queries, targets, n_targets, ckpt_path


__all__ = ["load_or_cache_queries"]
