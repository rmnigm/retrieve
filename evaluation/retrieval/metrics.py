"""Retrieval quality metrics for recommendation evaluation.

Supports both single-target (leave-last-out) and multi-target (time-based
split) evaluation. All per-query functions return [B] tensors.

Target format: ``targets [B, T]`` padded with -1, ``num_targets [B]``.
For single-target evaluation, T=1 and num_targets is all ones.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor


def _hits_mask(candidate_ids: Tensor, targets: Tensor, k: int) -> Tensor:
    """Boolean mask of which top-k candidates are relevant.

    Args:
        candidate_ids: [B, K_max] retrieved item IDs.
        targets: [B, T] ground-truth item IDs, padded with -1.
        k: cutoff.

    Returns:
        [B, k] bool tensor — True where candidate matches any target.
    """
    topk = candidate_ids[:, :k]  # [B, k]
    # [B, k, 1] == [B, 1, T] -> [B, k, T] -> any over T -> [B, k]
    return (topk.unsqueeze(2) == targets.unsqueeze(1)).any(dim=2)


def recall_at_k(
    candidate_ids: Tensor,
    targets: Tensor,
    num_targets: Tensor,
    k: int,
) -> Tensor:
    """Recall@K — fraction of relevant items retrieved in top-k.

    For single-target this equals hit rate.

    Returns:
        [B] float tensor.
    """
    hits = _hits_mask(candidate_ids, targets, k)  # [B, k]
    n_hits = hits.sum(dim=1).float()  # [B]
    return n_hits / num_targets.float().clamp(min=1)


def precision_at_k(
    candidate_ids: Tensor,
    targets: Tensor,
    num_targets: Tensor,
    k: int,
) -> Tensor:
    """Precision@K — fraction of top-k that are relevant.

    Returns:
        [B] float tensor.
    """
    hits = _hits_mask(candidate_ids, targets, k)  # [B, k]
    return hits.sum(dim=1).float() / k


def mrr_at_k(
    candidate_ids: Tensor,
    targets: Tensor,
    num_targets: Tensor,
    k: int,
) -> Tensor:
    """Mean Reciprocal Rank@K — 1/rank of first relevant item in top-k.

    Returns:
        [B] float tensor, 0.0 if no relevant item found.
    """
    hits = _hits_mask(candidate_ids, targets, k)  # [B, k]
    found = hits.any(dim=1)  # [B]
    # argmax returns index of first True (0 if none, masked below)
    rank = hits.float().argmax(dim=1) + 1  # [B], 1-indexed
    return (1.0 / rank.float()) * found.float()


def ndcg_at_k(
    candidate_ids: Tensor,
    targets: Tensor,
    num_targets: Tensor,
    k: int,
) -> Tensor:
    """NDCG@K — normalized discounted cumulative gain.

    IDCG is computed per query based on the number of relevant items:
    IDCG = sum_{i=1}^{min(num_targets, k)} 1/log2(i+1).

    Returns:
        [B] float tensor.
    """
    hits = _hits_mask(candidate_ids, targets, k)  # [B, k]
    positions = torch.arange(
        1, k + 1, device=candidate_ids.device, dtype=torch.float32
    )
    discounts = 1.0 / torch.log2(positions + 1)  # [k]
    dcg = (hits.float() * discounts.unsqueeze(0)).sum(dim=1)  # [B]

    # IDCG: best possible DCG given num_targets relevant items
    # ideal_hits[i] = 1 if i < min(num_targets, k) else 0
    ideal_hits = (
        positions.unsqueeze(0) <= num_targets.unsqueeze(1).float().clamp(max=k)
    )  # [B, k]
    idcg = (ideal_hits.float() * discounts.unsqueeze(0)).sum(dim=1)  # [B]

    return dcg / idcg.clamp(min=1e-8)


_METRIC_FNS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def accumulate_metrics(
    candidate_ids: Tensor,
    targets: Tensor,
    num_targets: Tensor,
    ks: list[int],
    accum: dict[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    """Compute per-query metrics and append to accumulator.

    Call once per batch. Values are moved to CPU and stored as Python
    floats so GPU memory stays bounded.

    Args:
        candidate_ids: [B, K_max] retrieved item IDs.
        targets: [B, T] ground-truth item IDs, padded with -1.
        num_targets: [B] number of valid targets per query.
        ks: list of cutoff values.
        accum: existing accumulator dict, or None to create a new one.

    Returns:
        Updated accumulator.
    """
    if accum is None:
        accum = defaultdict(list)

    for k in ks:
        for name, fn in _METRIC_FNS.items():
            key = f"{name}@{k}"
            values = fn(candidate_ids, targets, num_targets, k)
            accum[key].extend(values.cpu().tolist())

    return accum


def finalize_metrics(accum: dict[str, list[float]]) -> dict[str, float]:
    """Average accumulated per-query metric values.

    Args:
        accum: accumulator from accumulate_metrics.

    Returns:
        Dict mapping metric names to mean values.
    """
    results = {}
    for key, values in sorted(accum.items()):
        results[key] = sum(values) / len(values) if values else 0.0
    return results
