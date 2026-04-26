from __future__ import annotations

import torch
import torch.nn.functional as F


def gbce_loss(
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    output_embeddings: torch.nn.Embedding,
    uniform_negatives: torch.Tensor,
    gbce_t: float = 0.75,
) -> torch.Tensor:
    queries = hidden_states[mask]
    pos_ids = target_ids[mask]
    pos_scores = (queries * output_embeddings(pos_ids)).sum(-1, keepdim=True)

    neg_ids = uniform_negatives[mask]
    negs_per_pos = neg_ids.shape[1]
    num_items = output_embeddings.num_embeddings - 1
    neg_embs = output_embeddings(neg_ids)
    neg_scores = torch.einsum("pd,pkd->pk", queries, neg_embs)

    alpha = negs_per_pos / (num_items - 1)
    beta = alpha * ((1 - 1 / alpha) * gbce_t + 1 / alpha)
    eps = 1e-10
    pos_probs = torch.clamp(torch.sigmoid(pos_scores.double()), eps, 1 - eps)
    pos_adj = torch.clamp(pos_probs.pow(-beta), 1 + eps, torch.finfo(torch.float64).max)
    pos_transformed = (
        torch.clamp(1.0 / (pos_adj - 1), eps, torch.finfo(torch.float64).max)
        .log()
        .float()
    )

    logits = torch.cat([pos_transformed, neg_scores], dim=1)
    labels = torch.zeros_like(logits)
    labels[:, 0] = 1.0
    return F.binary_cross_entropy_with_logits(logits, labels)
