"""Retrieval quality metrics as device-side running sums (§2.4). ``per_row``'s float32
arithmetic and op order are frozen: the golden cells were scored with it."""

from __future__ import annotations

import torch
from torch import Tensor

METRICS = ("recall", "ndcg", "precision", "mrr")


def _hits(ids: Tensor, targets: Tensor) -> Tensor:
    """``[B, K]`` bool: candidate at rank r is one of the row's (non-``-1``) targets."""
    eq = ids.unsqueeze(2) == targets.unsqueeze(1)  # [B, K, T]
    valid = (ids.unsqueeze(2) != -1) & (targets.unsqueeze(1) != -1)
    return (eq & valid).any(dim=2)


def per_row(hits: Tensor, num_targets: Tensor, k: int) -> dict[str, Tensor]:
    """The four per-row metrics ([B] float32) at cutoff ``k`` from ``hits[:, :k]``."""
    h = hits[:, :k]
    n_hits = h.sum(dim=1).float()
    found = h.any(dim=1)
    rank = h.float().argmax(dim=1) + 1  # 1-indexed rank of the first hit (masked by found)
    positions = torch.arange(1, k + 1, device=h.device, dtype=torch.float32)
    discounts = 1.0 / torch.log2(positions + 1)
    dcg = (h.float() * discounts.unsqueeze(0)).sum(dim=1)
    ideal = positions.unsqueeze(0) <= num_targets.unsqueeze(1).float().clamp(max=k)
    idcg = (ideal.float() * discounts.unsqueeze(0)).sum(dim=1)
    return {
        "recall": n_hits / num_targets.float().clamp(min=1),
        "ndcg": dcg / idcg.clamp(min=1e-8),
        "precision": n_hits / k,
        "mrr": (1.0 / rank.float()) * found.float(),
    }


def accumulator(ks: list[int], device: torch.device | str) -> dict:
    """Zeroed running sums for every ``<metric>@<k>`` plus the row count ``n``."""
    acc: dict = {
        f"{m}@{k}": torch.zeros((), dtype=torch.float64, device=device) for k in ks for m in METRICS
    }
    acc["n"] = 0
    return acc


def accumulate(
    acc: dict,
    ids: Tensor,
    targets: Tensor,
    num_targets: Tensor | None = None,
    *,
    ranked: bool = False,
) -> dict[int, Tensor]:
    """Add one chunk's per-row metrics to ``acc`` (in place, no host sync). Returns the
    chunk's per-row recall ``{k: [B] float32}``, the values summed into ``recall@k``."""
    ks = sorted({int(key.split("@")[1]) for key in acc if "@" in key})
    recall: dict[int, Tensor] = {}
    if num_targets is None and not ranked:
        num_targets = (targets != -1).sum(dim=1)
    hits = None if ranked else _hits(ids, targets)
    for k in ks:
        if ranked:
            t = targets[:, :k]
            rows = per_row(_hits(ids[:, :k], t), (t != -1).sum(dim=1), k)
        else:
            rows = per_row(hits, num_targets, k)
        for m in METRICS:
            acc[f"{m}@{k}"] += rows[m].double().sum()
        recall[k] = rows["recall"]
    acc["n"] += int(ids.shape[0])
    return recall


def finalize(acc: dict) -> dict[str, float | None]:
    """Means over the accumulated rows — one device→host copy for all keys."""
    keys = [key for key in acc if key != "n"]
    values = torch.stack([acc[key] for key in keys]).div(max(acc["n"], 1)).tolist() if keys else []
    return null_if_empty({**dict(zip(keys, values, strict=True)), "n": acc["n"]})


def null_if_empty(side: dict) -> dict:
    """Every metric ``None`` when ``n == 0``: no scored row has no mean, never a 0.0 a report
    would average in as a miss."""
    return ({m: None for m in side if m != "n"} | {"n": 0}) if side.get("n") == 0 else side


def drop_self(ids: Tensor, self_ids: Tensor) -> Tensor:
    """``[B, K]`` → ``[B, K - 1]``: each row without its (at most one) ``self_ids`` entry, order
    kept; a row without it loses its last entry instead. ``-1`` padding stays ``-1``."""
    is_self = (ids == self_ids.unsqueeze(1)) & (ids != -1)
    is_self[:, -1] |= ~is_self.any(dim=1)
    return ids[~is_self].reshape(ids.shape[0], ids.shape[1] - 1)


def jaccard_at_k(ids_a: Tensor, ids_b: Tensor, k: int) -> float:
    """Mean per-row Jaccard of the top-``k`` id sets (``-1`` ignored; two empty sets → 1.0).
    The harness's cross-backend wiring check (``jaccard_vs_first@k``)."""
    a, b = ids_a[:, :k], ids_b[:, :k]
    inter = _hits(a, b).sum(dim=1).double()
    union = (a != -1).sum(dim=1) + (b != -1).sum(dim=1) - inter
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter)).mean().item()
