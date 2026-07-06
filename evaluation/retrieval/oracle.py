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

import hashlib
import sys
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from retrieve.interfaces import FilterModule

_FP_SAMPLE_ROWS = 64


def _oracle_fingerprint(
    item_embs: torch.Tensor,
    queries: torch.Tensor,
    qa_narrow_sweep: torch.Tensor | None,
    k_gt: int,
) -> str:
    """Cheap deterministic content hash: shapes + dtypes + a fixed row sample.

    Sampling (vs hashing 3M×256 fp32 fully) keeps this <10 ms; linspace rows
    catch dim changes, re-encodes, attr regens, and checkpoint swaps — any of
    which perturb sampled bytes. Not adversarially robust; doesn't need to be."""
    h = hashlib.sha256()
    tensors = [item_embs, queries] + ([qa_narrow_sweep] if qa_narrow_sweep is not None else [])
    for t in tensors:
        h.update(repr((tuple(t.shape), str(t.dtype))).encode())
        idx = torch.linspace(0, t.shape[0] - 1, steps=min(_FP_SAMPLE_ROWS, t.shape[0])).long()
        h.update(t[idx].detach().float().cpu().contiguous().numpy().tobytes())
    h.update(str(k_gt).encode())
    return h.hexdigest()


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

    Skipped rows get all -1. ``filter_mod`` must be an *exact* mask
    source — i.e. ``ExactAttributeFilter`` even on bloom-suite runs, so
    bloom's false positives do not leak into the ground truth.

    Indices are 0-indexed positions in the (already pad-row-dropped)
    ``item_embs`` — the loaders strip the training-side padding row
    before any retrieval-time tensor leaves the harness.
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

    for s in tqdm(
        range(0, keep_idx.numel(), batch_size),
        desc="oracle",
        leave=False,
        disable=not sys.stderr.isatty(),
    ):
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
    K_GT: int,
    *,
    item_embs: torch.Tensor,
    queries: torch.Tensor,
    qa_narrow_sweep: torch.Tensor | None,
    skip_mask: torch.Tensor | None,
    oracle_filter: FilterModule | None,
    device: torch.device,
) -> torch.Tensor:
    """Load cached oracle from disk; recompute on content-fingerprint mismatch.

    Disk cache lives at ``<gt_dir>/gt_topk_v3_<sweep_name>.pt`` as a dict
    blob ``{"topk", "fingerprint", "k_gt"}``. The fingerprint hashes the
    post-``users_limit`` tensors the oracle is actually built from, so
    same-shape content changes — a different ``content_subdir`` dim off
    the same ``data_dir``, regenerated attrs, a retrained checkpoint —
    invalidate the cache instead of silently reusing stale ground truth
    (per-dim ``gt_subdir`` remains cheap defense-in-depth, not a
    correctness requirement). Legacy ``gt_topk_v2_`` bare-tensor caches
    are ignored by name and can be deleted; a bare tensor or stale dict
    found at the v3 path is recomputed and overwritten, never migrated
    in-place.
    """
    gt_path = gt_dir / f"gt_topk_v3_{sweep_name}.pt"
    fp = _oracle_fingerprint(item_embs, queries, qa_narrow_sweep, K_GT)
    if gt_path.exists():
        blob = torch.load(str(gt_path), map_location="cpu", weights_only=True)
        if isinstance(blob, dict) and blob.get("fingerprint") == fp:
            logger.info("  loaded oracle from cache: {}", gt_path)
            return blob["topk"]
        logger.warning("stale/legacy oracle at {} (fingerprint mismatch); recomputing", gt_path)
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
    torch.save({"topk": oracle_topk, "fingerprint": fp, "k_gt": K_GT}, str(gt_path))
    logger.info("  saved oracle → {}", gt_path)
    return oracle_topk


__all__ = ["compute_filtered_oracle", "load_or_build_oracle"]
