"""The IVF layout every ``SilverTorch`` backend registers — cluster-sorted CSR — derived from one
k-means assignment, and the probe width its scorers write.

The sort is a stable ``argsort`` so that items inside a cluster keep ascending-id order and the
slot order is a function of the assignment alone, not of the sort implementation
(torch's CUDA sort is stable for segments > 4096 and was measured stable below it too — bit-
identical buffers and outputs on every regime, roadmap B2,
docs/plans/official-silvertorch-artifacts/wp3/argsort_stable_probe.txt).
"""

from __future__ import annotations

import torch
from torch import Tensor


def csr_layout(assignments: Tensor, n_lists: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cluster-sorted CSR: ``(sort_perm [N]`` sorted position →
    original id, ``inv_perm [N]`` original id → sorted position, ``cluster_offsets [n_lists+1]``,
    ``cluster_sizes [n_lists])``."""
    cluster_sizes = torch.bincount(assignments, minlength=n_lists)
    sort_perm = torch.argsort(assignments, stable=True)
    inv_perm = torch.empty_like(sort_perm)
    inv_perm[sort_perm] = torch.arange(sort_perm.numel(), device=sort_perm.device)
    offsets = torch.zeros(n_lists + 1, dtype=torch.long, device=assignments.device)
    offsets[1:] = cluster_sizes.cumsum(0)
    return sort_perm, inv_perm, offsets, cluster_sizes


def probe_width(cluster_sizes: Tensor, n_probe: int) -> int:
    """The scorer's static output width: the sum of the ``n_probe`` largest clusters, an upper
    bound on any row's probed items (the probed clusters are distinct). One host sync, at
    register / ``set_query_params`` / load time."""
    return int(cluster_sizes.topk(n_probe).values.sum().item())
