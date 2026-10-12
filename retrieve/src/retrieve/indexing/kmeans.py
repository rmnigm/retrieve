from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


def _chunk(n: int) -> int:
    return max(1, min(n, 1 << 14))


KMeansInit = Literal["random", "kmeans++"]


class KMeans:
    """Lloyd's k-means with chunked centroid assignment; ``fit`` returns (centroids [n_lists, D],
    assignments [N]). ``init="random"`` seeds from ``n_lists`` distinct rows; ``init="kmeans++"``
    seeds by D² sampling (Arthur & Vassilvitskii 2007, the paper's choice — SilverTorch §4.1),
    one pass over the index per centroid, no host sync per step.

    ``fit`` is **bit-for-bit reproducible run to run** on a given device: both initialisations
    draw from a CPU generator seeded with ``seed``, the assignment step is a deterministic
    ``cdist`` + ``argmin``, and the centroid-update reduction is order-fixed (see
    ``_cluster_sums``). Reproducibility is a gate requirement — SilverTorch's IVF layout, and
    therefore every quality number it produces, is a pure function of these centroids."""

    def __init__(
        self, n_lists: int, n_iter: int = 10, seed: int = 0, init: KMeansInit = "random"
    ) -> None:
        if init not in ("random", "kmeans++"):
            raise ValueError(f"init must be 'random' or 'kmeans++', got {init!r}")
        self.n_lists = n_lists
        self.n_iter = n_iter
        self.seed = seed
        self.init = init

    def fit(self, embs: Tensor) -> tuple[Tensor, Tensor]:
        n, _ = embs.shape
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        if self.init == "kmeans++":
            centroids = self._seed_kmeanspp(embs, g)
        else:
            perm = torch.randperm(n, generator=g)[: self.n_lists]
            centroids = embs[perm].clone().float()

        chunk = _chunk(n)

        for _ in range(self.n_iter):
            assignments = self.assign(embs, centroids)

            new_sums, counts = self._cluster_sums(embs, assignments, self.n_lists, chunk)
            non_empty = counts > 0
            centroids = torch.where(
                non_empty.unsqueeze(1),
                new_sums / counts.clamp(min=1).unsqueeze(1),
                centroids,
            )

        return centroids, self.assign(embs, centroids)

    def _seed_kmeanspp(self, embs: Tensor, g: torch.Generator) -> Tensor:
        """D² sampling: the first centroid uniform, each next one drawn with probability
        proportional to its squared distance to the nearest centroid so far. One ``mv`` over
        the index per centroid (``‖x‖² − 2x·c + ‖c‖²``, clamped at 0); the draw is a CPU
        uniform from ``g`` located by ``searchsorted`` on the device-side cumsum
        (``right=True`` so a point already chosen, at zero mass, is never drawn again)."""
        x = embs.float()
        n = x.shape[0]
        x_norm = (x * x).sum(dim=1)
        centroids = torch.empty(self.n_lists, x.shape[1], dtype=torch.float32, device=x.device)
        centroids[0] = x[torch.randint(n, (1,), generator=g)[0]]
        d2 = (x_norm - 2 * torch.mv(x, centroids[0]) + centroids[0].dot(centroids[0])).clamp_min(0)
        for j in range(1, self.n_lists):
            cum = torch.cumsum(d2, dim=0)
            target = torch.rand((), generator=g) * cum[-1:]
            idx = torch.searchsorted(cum, target, right=True).clamp_max(n - 1)[0]
            c = x[idx]
            centroids[j] = c
            d2 = torch.minimum(d2, (x_norm - 2 * torch.mv(x, c) + c.dot(c)).clamp_min(0))
        return centroids

    @staticmethod
    def _cluster_sums(
        embs: Tensor, assignments: Tensor, n_lists: int, chunk: int
    ) -> tuple[Tensor, Tensor]:
        """Per-cluster sums ``[n_lists, D]`` (float32) and counts ``[n_lists]`` (float32), with a
        reduction order fixed by the shapes rather than by the scheduler: ``index_add_``'s float
        atomics are not reproducible on CUDA, a one-hot GEMM's reduction order is. float64, so it
        is exact past float32 and no caller's TF32 setting can demote it (kernels.md § KMeans
        centroid update)."""
        n, d = embs.shape
        counts = torch.bincount(assignments, minlength=n_lists)
        sums = torch.zeros(n_lists, d, dtype=torch.float64, device=embs.device)
        # 1 << 23 float64 = 64 MiB of one-hot, so a large n_lists costs panels, not memory.
        width = max(1, min(chunk, (1 << 23) // max(1, n_lists)))

        for start in range(0, n, width):
            end = min(start + width, n)
            onehot = torch.zeros(n_lists, end - start, dtype=torch.float64, device=embs.device)
            onehot.scatter_(0, assignments[start:end].unsqueeze(0), 1.0)
            sums.addmm_(onehot, embs[start:end].double())

        return sums.float(), counts.to(torch.float32)

    @staticmethod
    def assign(embs: Tensor, centroids: Tensor) -> Tensor:
        """Nearest centroid per row, ``[N]`` int64; chunked ``cdist`` + ``argmin``."""
        n = embs.shape[0]
        chunk = _chunk(n)
        assignments = torch.empty(n, dtype=torch.long, device=embs.device)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            d = torch.cdist(embs[start:end].float(), centroids)
            assignments[start:end] = d.argmin(dim=1)
        return assignments
