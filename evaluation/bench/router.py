"""The local-pass-rate router (roadmap V-ROUTER; docs/system/evaluation.md § The router arm): a
harness arm, not a library module. An unfiltered SilverTorch pre-probe returns each query's
top-100; the share of them its filter admits is the local pass rate ``l_q``. Rows with
``l_q < lq_threshold`` go to exact LiNR V2, the others to filtered SilverTorch (IVF recall
follows ``l_q``: EXHIBITS idea #2). The pre-probe runs inside ``forward``, so its cost is part of
the arm's timing. The split is data-dependent, so the arm is eager only."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from retrieve.interfaces import FilterModule

PRE_K = 100  # l_q's neighbourhood: the share of the unfiltered top-100 that passes


class Router(nn.Module):
    capturable = False

    def __init__(
        self,
        *,
        pre: nn.Module,
        ivf: nn.Module,
        exact: nn.Module,
        filter: FilterModule,
        lq_threshold: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= lq_threshold <= 1.0:
            raise ValueError(f"lq_threshold must lie in [0, 1], got {lq_threshold!r}")
        self.pre, self.ivf, self.exact, self.filter = pre, ivf, exact, filter
        self.lq_threshold = float(lq_threshold)
        self.backend = ivf.backend
        self.last_lq: Tensor | None = None  # [B] of the last forward, read by the quality pass
        self.last_exact: Tensor | None = None

    @property
    def k(self) -> int:
        return self.ivf.k

    @k.setter
    def k(self, k: int) -> None:
        self.ivf.k = k
        self.exact.k = k

    def set_query_params(self, *, n_probe: int) -> None:
        self.ivf.set_query_params(n_probe=n_probe)

    def local_pass_rate(self, query: Tensor, query_clause_attrs: Tensor) -> Tensor:
        ids, _ = self.pre(query)
        found = ids >= 0
        passing = self.filter.evaluate_subset(query_clause_attrs, ids.clamp_min(0)) & found
        return passing.sum(dim=1).float() / PRE_K

    def forward(self, query: Tensor, query_clause_attrs: Tensor) -> tuple[Tensor, Tensor]:
        lq = self.local_pass_rate(query, query_clause_attrs)
        exact = lq < self.lq_threshold
        self.last_lq, self.last_exact = lq, exact
        ids = torch.full((query.shape[0], self.k), -1, dtype=torch.long, device=query.device)
        scores = torch.full(ids.shape, float("-inf"), dtype=torch.float32, device=query.device)
        for branch, rows in (
            (self.exact, exact.nonzero().reshape(-1)),
            (self.ivf, (~exact).nonzero().reshape(-1)),
        ):
            if rows.numel():
                i, s = branch(query.index_select(0, rows), query_clause_attrs.index_select(0, rows))
                ids.index_copy_(0, rows, i.long())
                scores.index_copy_(0, rows, s.float())
        return ids, scores
