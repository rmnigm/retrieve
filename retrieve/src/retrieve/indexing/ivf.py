"""The two IVF cluster layouts ``SilverTorch`` registers, derived from one k-means assignment.

Both sort items with a stable ``argsort`` so that items inside a cluster keep ascending-id order
and the slot order is a function of the assignment alone, not of the sort implementation
(torch's CUDA sort is stable for segments > 4096 and was measured stable below it too — bit-
identical buffers and outputs on every regime, roadmap B2,
docs/plans/official-silvertorch-artifacts/wp3/argsort_stable_probe.txt).
"""

from __future__ import annotations

import torch
from torch import Tensor


def csr_layout(assignments: Tensor, n_lists: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cluster-sorted CSR — the official scorer's layout: ``(sort_perm [N]`` sorted position →
    original id, ``inv_perm [N]`` original id → sorted position, ``cluster_offsets [n_lists+1]``,
    ``cluster_sizes [n_lists])``."""
    cluster_sizes = torch.bincount(assignments, minlength=n_lists)
    sort_perm = torch.argsort(assignments, stable=True)
    inv_perm = torch.empty_like(sort_perm)
    inv_perm[sort_perm] = torch.arange(sort_perm.numel(), device=sort_perm.device)
    offsets = torch.zeros(n_lists + 1, dtype=torch.long, device=assignments.device)
    offsets[1:] = cluster_sizes.cumsum(0)
    return sort_perm, inv_perm, offsets, cluster_sizes


def padded_layout(assignments: Tensor, n_lists: int) -> tuple[Tensor, Tensor]:
    """Padded cluster table — the Triton / torch layout: ``(padded_cluster_items [n_lists,
    max_size]`` with ``-1`` padding, ``cluster_sizes [n_lists])``. Slot order inside a cluster
    is the CSR's."""
    sort_perm, _, offsets, cluster_sizes = csr_layout(assignments, n_lists)
    n = assignments.shape[0]
    max_size = int(cluster_sizes.max().item())
    sorted_clusters = assignments[sort_perm]
    padded = torch.full((n_lists, max_size), -1, dtype=torch.long, device=assignments.device)
    within_slot = torch.arange(n, device=assignments.device) - offsets[sorted_clusters]
    padded[sorted_clusters, within_slot] = sort_perm
    return padded, cluster_sizes
