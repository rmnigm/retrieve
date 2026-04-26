from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import RetrievalModule
from retrieve.layers.utils.quantize import quantize_int8


class IVF_INT8_ANN(RetrievalModule):
    """Tensor-native IVF + INT8 ANN retrieval."""

    centroids: Tensor
    item_codes: Tensor
    item_scales: Tensor
    padded_cluster_items: Tensor
    cluster_sizes: Tensor

    def __init__(
        self,
        k: int,
        n_lists: int,
        n_probe: int,
        n_iter: int = 10,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.k = k
        self.n_lists = n_lists
        self.n_probe = n_probe
        self.n_iter = n_iter
        self.seed = seed

    def register_index(self, item_embs: Tensor) -> None:
        n, _ = item_embs.shape
        if self.n_lists > n:
            raise ValueError(f"n_lists ({self.n_lists}) cannot exceed N ({n}).")
        if self.n_probe > self.n_lists:
            raise ValueError(f"n_probe ({self.n_probe}) cannot exceed n_lists ({self.n_lists}).")

        centroids, assignments = _kmeans_torch(item_embs, self.n_lists, self.n_iter, self.seed)
        cluster_sizes = torch.bincount(assignments, minlength=self.n_lists)
        max_size = int(cluster_sizes.max().item())

        sort_idx = torch.argsort(assignments)
        sorted_clusters = assignments[sort_idx]
        offsets = torch.zeros(self.n_lists + 1, dtype=torch.long, device=item_embs.device)
        offsets[1:] = cluster_sizes.cumsum(0)

        padded = torch.full(
            (self.n_lists, max_size),
            -1,
            dtype=torch.long,
            device=item_embs.device,
        )
        within_slot = torch.arange(n, device=item_embs.device) - offsets[sorted_clusters]
        padded[sorted_clusters, within_slot] = sort_idx

        codes, scales = quantize_int8(item_embs)

        self.register_buffer("centroids", centroids)
        self.register_buffer("item_codes", codes)
        self.register_buffer("item_scales", scales)
        self.register_buffer("padded_cluster_items", padded)
        self.register_buffer("cluster_sizes", cluster_sizes)

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
        candidate_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if candidate_ids is not None:
            return self._forward_candidates(query, candidate_ids)
        return self._forward_full(query, mask)

    def _forward_full(
        self,
        query: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        b = query.shape[0]

        cent_scores = query @ self.centroids.t()
        _, probe_ids = torch.topk(cent_scores, self.n_probe, dim=1)  # [B, n_probe]

        probed = self.padded_cluster_items[probe_ids]  # [B, n_probe, max_size]
        flat_items = probed.reshape(b, -1)  # [B, P]
        valid = flat_items >= 0
        safe = flat_items.clamp(min=0)

        codes = self.item_codes[safe].float()  # [B, P, D]
        scales = self.item_scales[safe]  # [B, P]
        scores = torch.einsum("bd,bpd->bp", query, codes) * scales
        scores = scores.masked_fill(~valid, float("-inf"))

        if mask is not None:
            item_mask = mask.gather(1, safe) & valid
            scores = scores.masked_fill(~item_mask, float("-inf"))

        actual_k = min(self.k, scores.shape[1])
        topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
        topk_ids = flat_items.gather(1, topk_local)
        return topk_ids, topk_scores

    def _forward_candidates(
        self,
        query: Tensor,
        candidate_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cand_codes = self.item_codes[candidate_ids]
        cand_scales = self.item_scales[candidate_ids]
        cand_embs = cand_codes.float() * cand_scales.unsqueeze(-1)
        scores = torch.bmm(query.unsqueeze(1), cand_embs.transpose(1, 2)).squeeze(1)

        actual_k = min(self.k, scores.shape[1])
        topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
        topk_ids = candidate_ids.gather(1, topk_local)
        return topk_ids, topk_scores


def _kmeans_torch(
    embs: Tensor,
    n_lists: int,
    n_iter: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    n, d = embs.shape
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)[:n_lists]
    centroids = embs[perm].clone().float()

    chunk = max(1, min(n, 1 << 14))

    for _ in range(n_iter):
        assignments = _assign_chunked(embs, centroids, chunk)

        new_sums = torch.zeros_like(centroids)
        counts = torch.zeros(n_lists, dtype=torch.float, device=embs.device)
        new_sums.index_add_(0, assignments, embs.float())
        counts.index_add_(0, assignments, torch.ones(n, dtype=torch.float, device=embs.device))
        non_empty = counts > 0
        new_centroids = torch.where(
            non_empty.unsqueeze(1),
            new_sums / counts.clamp(min=1).unsqueeze(1),
            centroids,
        )
        centroids = new_centroids

    assignments = _assign_chunked(embs, centroids, chunk)
    return centroids, assignments


def _assign_chunked(embs: Tensor, centroids: Tensor, chunk: int) -> Tensor:
    n = embs.shape[0]
    assignments = torch.empty(n, dtype=torch.long, device=embs.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = torch.cdist(embs[start:end].float(), centroids)
        assignments[start:end] = d.argmin(dim=1)
    return assignments


def build_ivf_int8(
    item_embs: Tensor,
    k: int,
    *,
    n_lists: int,
    n_probe: int,
    n_iter: int = 10,
    seed: int = 0,
) -> IVF_INT8_ANN:
    module = IVF_INT8_ANN(k=k, n_lists=n_lists, n_probe=n_probe, n_iter=n_iter, seed=seed)
    module.register_index(item_embs)
    return module
