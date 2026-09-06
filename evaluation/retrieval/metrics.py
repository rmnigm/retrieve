"""Retrieval quality metrics as device-side running sums (harness v2, H §2.4).

The quality pass streams chunks of ``(ids [B, K_max], targets [B, T])`` through
``accumulate``; every ``k`` in ``ks`` is scored from the one top-``K_max`` list
(the algos return ``torch.topk``-sorted rows, so the top-``k`` prefix *is* the
top-``k`` result). Sums live on the device as float64 and are read back once,
in ``finalize`` — one sync per pass instead of one per chunk.

Targets are ``[B, T]`` int64 padded with ``-1``; ``-1`` on either side never
counts as a hit. Two target modes:

* fixed (held-out items): the target set is the same at every ``k``; pass
  ``num_targets`` (or let it be derived as the count of non-``-1`` entries);
* ``ranked=True`` (oracle top-``K_max`` list): the target set at ``k`` is the
  oracle's own top-``k`` prefix and ``num_targets`` is its non-``-1`` count —
  the old harness's per-``k`` ``nt_k`` semantics, so oracle recall stays
  golden-comparable.

The per-row arithmetic in ``per_row`` is the old ``metrics.py`` verbatim
(float32, same op order); ``tests/test_metrics.py`` asserts the running-sum
means equal the old per-row means to 1e-9. The four ``*_at_k`` functions and
``accumulate_metrics`` / ``finalize_metrics`` are the old per-row API, kept
for ``passes.py`` until C3 deletes it.
"""

from __future__ import annotations

from collections import defaultdict

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
        f"{m}@{k}": torch.zeros((), dtype=torch.float64, device=device)
        for k in ks
        for m in METRICS
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
) -> None:
    """Add one chunk's per-row metrics to ``acc`` (in place, no host sync)."""
    ks = sorted({int(key.split("@")[1]) for key in acc if "@" in key})
    if ranked:
        for k in ks:
            t = targets[:, :k]
            rows = per_row(_hits(ids[:, :k], t), (t != -1).sum(dim=1), k)
            for m in METRICS:
                acc[f"{m}@{k}"] += rows[m].double().sum()
    else:
        if num_targets is None:
            num_targets = (targets != -1).sum(dim=1)
        hits = _hits(ids, targets)
        for k in ks:
            rows = per_row(hits, num_targets, k)
            for m in METRICS:
                acc[f"{m}@{k}"] += rows[m].double().sum()
    acc["n"] += int(ids.shape[0])


def finalize(acc: dict) -> dict[str, float]:
    """Means over the accumulated rows — one device→host copy for all keys."""
    keys = [key for key in acc if key != "n"]
    n = max(acc["n"], 1)
    values = torch.stack([acc[key] for key in keys]).div(n).tolist() if keys else []
    out = dict(zip(keys, values))
    out["n"] = acc["n"]
    return out


def jaccard_at_k(ids_a: Tensor, ids_b: Tensor, k: int) -> float:
    """Mean per-row Jaccard of the top-``k`` id sets (``-1`` ignored; two empty sets → 1.0).
    The harness's cross-backend wiring check (H §2.4 ``jaccard_vs_first@k``)."""
    a, b = ids_a[:, :k], ids_b[:, :k]
    inter = _hits(a, b).sum(dim=1).double()
    union = (a != -1).sum(dim=1) + (b != -1).sum(dim=1) - inter
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter)).mean().item()


# ----- old per-row API (consumed by passes.py and the pre-v2 tests; C3 deletes) -----


def recall_at_k(candidate_ids: Tensor, targets: Tensor, num_targets: Tensor, k: int) -> Tensor:
    return per_row(_hits(candidate_ids[:, :k], targets), num_targets, k)["recall"]


def precision_at_k(candidate_ids: Tensor, targets: Tensor, num_targets: Tensor, k: int) -> Tensor:
    return per_row(_hits(candidate_ids[:, :k], targets), num_targets, k)["precision"]


def mrr_at_k(candidate_ids: Tensor, targets: Tensor, num_targets: Tensor, k: int) -> Tensor:
    return per_row(_hits(candidate_ids[:, :k], targets), num_targets, k)["mrr"]


def ndcg_at_k(candidate_ids: Tensor, targets: Tensor, num_targets: Tensor, k: int) -> Tensor:
    return per_row(_hits(candidate_ids[:, :k], targets), num_targets, k)["ndcg"]


def accumulate_metrics(
    candidate_ids: Tensor,
    targets: Tensor,
    num_targets: Tensor,
    ks: list[int],
    accum: dict[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    """Old API: per-row values appended as Python floats (one sync per call)."""
    if accum is None:
        accum = defaultdict(list)
    for k in ks:
        rows = per_row(_hits(candidate_ids[:, :k], targets), num_targets, k)
        for m in METRICS:
            accum[f"{m}@{k}"].extend(rows[m].cpu().tolist())
    return accum


def finalize_metrics(accum: dict[str, list[float]]) -> dict[str, float]:
    """Old API: plain means of the appended per-row values."""
    return {key: sum(v) / len(v) if v else 0.0 for key, v in sorted(accum.items())}
