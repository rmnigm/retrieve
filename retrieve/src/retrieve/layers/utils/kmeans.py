from __future__ import annotations

import torch
from torch import Tensor


class KMeansTorch:
    """Lloyd's k-means with chunked centroid assignment; ``fit`` returns (centroids [n_lists, D],
    assignments [N])."""

    def __init__(self, n_lists: int, n_iter: int = 10, seed: int = 0) -> None:
        self.n_lists = n_lists
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, embs: Tensor) -> tuple[Tensor, Tensor]:
        n, _ = embs.shape
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        perm = torch.randperm(n, generator=g)[: self.n_lists]
        centroids = embs[perm].clone().float()

        chunk = max(1, min(n, 1 << 14))

        for _ in range(self.n_iter):
            assignments = self._assign_chunked(embs, centroids, chunk)

            new_sums = torch.zeros_like(centroids)
            counts = torch.zeros(self.n_lists, dtype=torch.float, device=embs.device)
            new_sums.index_add_(0, assignments, embs.float())
            counts.index_add_(0, assignments, torch.ones(n, dtype=torch.float, device=embs.device))
            non_empty = counts > 0
            centroids = torch.where(
                non_empty.unsqueeze(1),
                new_sums / counts.clamp(min=1).unsqueeze(1),
                centroids,
            )

        assignments = self._assign_chunked(embs, centroids, chunk)
        return centroids, assignments

    @staticmethod
    def _assign_chunked(embs: Tensor, centroids: Tensor, chunk: int) -> Tensor:
        n = embs.shape[0]
        assignments = torch.empty(n, dtype=torch.long, device=embs.device)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            d = torch.cdist(embs[start:end].float(), centroids)
            assignments[start:end] = d.argmin(dim=1)
        return assignments
