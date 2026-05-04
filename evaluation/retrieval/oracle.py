"""Filtered-FullScan ground truth for filter-bench cells.

The oracle is a brute-force ``q @ E_t`` top-K_GT, computed once per (sweep,
filter_kind) pair and cached on disk. Algos under test are scored against
this oracle, not against held-out targets, because filter sweeps deliberately
restrict the candidate set: held-out targets often fall outside the filtered
catalog and would tank recall regardless of algo quality.

``filter_mod`` MUST be an *exact* mask source (``ExactAttributeFilter``).
Bloom's false positives must NOT leak into the ground truth — bloom-suite
runs build a separate exact filter over the same attrs at the call site.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from retrieve.interfaces import FilterModule


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

    Skipped rows get all -1. ``id 0`` is masked out (padding row of
    ``item_embs``). ``filter_mod`` must be an *exact* mask source — i.e.
    ``ExactAttributeFilter`` even on bloom-suite runs, so bloom's false
    positives do not leak into the ground truth.
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


def load_or_build_oracle(
    gt_dir: Path,
    sweep_name: str,
    n_users: int,
    K_GT: int,
    *,
    item_embs: torch.Tensor,
    queries: torch.Tensor,
    qa_narrow_sweep: torch.Tensor | None,
    skip_mask: torch.Tensor | None,
    oracle_filter: FilterModule | None,
    device: torch.device,
) -> torch.Tensor:
    """Load cached oracle from disk; recompute and cache on shape mismatch.

    Disk cache lives at ``<gt_dir>/gt_topk_<sweep_name>.pt``. A stale cache
    (different ``n_users`` or ``K_GT``) is recomputed and overwritten — the
    common cause is changing ``content_subdir`` between runs (item_embs
    differ → oracle scores differ).
    """
    gt_path = gt_dir / f"gt_topk_{sweep_name}.pt"
    oracle_topk: torch.Tensor | None = None
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
            qa_narrow_sweep,
            skip_mask,
            oracle_filter,
            K_GT=K_GT,
            device=device,
        )
        torch.save(oracle_topk, str(gt_path))
        logger.info("  saved oracle → {}", gt_path)
    return oracle_topk


__all__ = ["compute_filtered_oracle", "load_or_build_oracle"]
