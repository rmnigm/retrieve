"""The two losses over one shared candidate id vector: ``queries [P, D]`` against their
positives ``pos_ids [P]`` and ``neg_ids [K]`` shared by every row."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gbce_loss(
    queries: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    table: torch.Tensor,
    num_items: int,
    t: float,
) -> torch.Tensor:
    pos_scores = (queries * table[pos_ids]).sum(-1, keepdim=True)
    neg_scores = queries @ table[neg_ids].T

    alpha = neg_ids.shape[0] / (num_items - 1)
    beta = alpha * ((1 - 1 / alpha) * t + 1 / alpha)
    eps = 1e-10
    pos_probs = torch.clamp(torch.sigmoid(pos_scores.double()), eps, 1 - eps)
    pos_adj = torch.clamp(pos_probs.pow(-beta), 1 + eps, torch.finfo(torch.float64).max)
    pos_transformed = (
        torch.clamp(1.0 / (pos_adj - 1), eps, torch.finfo(torch.float64).max).log().float()
    )

    logits = torch.cat([pos_transformed, neg_scores.float()], dim=1)
    labels = torch.zeros_like(logits)
    labels[:, 0] = 1.0
    return F.binary_cross_entropy_with_logits(logits, labels)


def sampled_softmax_loss(
    queries: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    table: torch.Tensor,
    temperature: float,
    normalize: bool,
) -> torch.Tensor:
    """Cross-entropy of the positive against the shared candidates; a candidate equal to the
    row's own positive (an accidental hit) is masked out."""
    with torch.autocast(queries.device.type, enabled=False):
        q, pos, neg = queries.float(), table[pos_ids].float(), table[neg_ids].float()
        if normalize:
            q, pos, neg = F.normalize(q, dim=-1), F.normalize(pos, dim=-1), F.normalize(neg, dim=-1)
        neg_logits = (q @ neg.T).masked_fill(neg_ids[None, :] == pos_ids[:, None], float("-inf"))
        logits = torch.cat([(q * pos).sum(-1, keepdim=True), neg_logits], dim=1) / temperature
        return F.cross_entropy(logits, torch.zeros_like(pos_ids))
