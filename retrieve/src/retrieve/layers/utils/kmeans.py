from __future__ import annotations

import torch
from torch import Tensor


class KMeansTorch:
    """Lloyd's k-means with chunked centroid assignment; ``fit`` returns (centroids [n_lists, D],
    assignments [N]).

    ``fit`` is **bit-for-bit reproducible run to run** on a given device: the initialisation
    draws from a CPU generator seeded with ``seed``, the assignment step is a deterministic
    ``cdist`` + ``argmin``, and the centroid-update reduction is order-fixed (see
    ``_cluster_sums``). Reproducibility is a gate requirement — SilverTorch's IVF layout, and
    therefore every quality number it produces, is a pure function of these centroids."""

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

            new_sums, counts = self._cluster_sums(embs, assignments, self.n_lists, chunk)
            non_empty = counts > 0
            centroids = torch.where(
                non_empty.unsqueeze(1),
                new_sums / counts.clamp(min=1).unsqueeze(1),
                centroids,
            )

        assignments = self._assign_chunked(embs, centroids, chunk)
        return centroids, assignments

    @staticmethod
    def _cluster_sums(
        embs: Tensor, assignments: Tensor, n_lists: int, chunk: int
    ) -> tuple[Tensor, Tensor]:
        """Per-cluster sums ``[n_lists, D]`` (float32) and counts ``[n_lists]`` (float32), with a
        reduction order fixed by the shapes rather than by the scheduler.

        The obvious form — ``torch.zeros(n_lists, D).index_add_(0, assignments, embs)`` — is
        **not reproducible on CUDA**: ``index_add_`` reduces with floating-point atomics, so the
        summation order of a cluster's members depends on block scheduling, and two identical
        builds drift apart (roadmap C4 measured 1.8e-2 in the centroids after 10 Lloyd
        iterations — one assignment flipped early cascades through the rest). Here instead:

        1. ``bincount`` gives the counts. Integer atomics are exact whatever the order.
        2. The sums are a **one-hot GEMM** in float64, ``onehot[n_lists, W] @ embs[W, D]``, over
           panels of ``W`` items, with the panel partials accumulated into ``sums`` in panel
           order (``addmm_``, one accumulation per panel, issued sequentially on the stream). A
           GEMM's reduction order is a function of its shapes alone — no atomics — so it repeats
           bit for bit on the same device, and the panel loop is a Python ``for``.

        float64 rather than float32 for two reasons, at 8% more time on the A100 (DMMA runs at
        the same rate as the float32 SIMT path here): the accumulation is then exact well past
        float32's resolution, and — unlike a float32 matmul — it cannot be silently demoted to
        TF32 by a caller's ``torch.set_float32_matmul_precision("high")``, which would cost 13
        mantissa bits of ``embs`` and swamp the drift this is fixing. Nothing global is touched.

        Cost is one extra ``[n_lists, N, D]``-shaped matmul per Lloyd iteration — the same FLOPs
        as the ``cdist`` assignment step that precedes it, so ``fit`` runs at most ~2x its old
        wall time whatever ``n_lists`` is (measured 1.3x at N=200k, D=128, n_lists=1024). The
        panel width is capped so the one-hot buffer stays at 64 MiB regardless of ``n_lists``."""
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
    def _assign_chunked(embs: Tensor, centroids: Tensor, chunk: int) -> Tensor:
        n = embs.shape[0]
        assignments = torch.empty(n, dtype=torch.long, device=embs.device)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            d = torch.cdist(embs[start:end].float(), centroids)
            assignments[start:end] = d.argmin(dim=1)
        return assignments
