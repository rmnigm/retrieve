"""SilverTorch — IVF + INT8 ANN, three valid construction modes.

* ``filter_kind="bloom"``: codesigned bloom-fused IVF. Item bloom
  signatures over the narrow attrs are baked into the IVF at register
  time and filter during the probe; the query forward passes
  ``query_clause_attrs=qa_narrow`` so the kernel checks bloom bits inline.
* ``filter_kind="clause"``: codesigned exact-clause-fused IVF. Item
  clause attributes are stored as ``[N, C, A_max]`` int64 alongside the
  IVF and the AND/OR/XOR predicate is evaluated inside the probe — same
  fused-filter shape as the bloom path, no false positives. Enabled by
  the ``codesigned_probe_score_exact`` kernel.
* ``filter_kind="none"``: plain IVF + INT8, no filter.

When ``filter_kind="clause"``, the wrapper threads ``clause_is_reverse``
into ``SilverTorch.register_index`` so sweeps with reverse predicates
(e.g., Goodreads ``c1_lang_reverse``) are evaluated correctly via the
codesigned exact-clause kernel's per-clause XOR. Without this kwarg the
kernel would treat every clause as a positive equality, collapsing
recall to ~0 on reverse-clause sweeps.
"""

from __future__ import annotations

from torch import Tensor, nn

from retrieve import SilverTorch
from retrieve.interfaces import Backend

from ._helpers import collect_modules


class SilvertorchAlgo(nn.Module):
    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        filter_kind: str = "none",
        item_attrs_narrow: Tensor | None = None,
        clause_is_reverse: Tensor | None = None,
        n_lists: int = 1024,
        n_probe: int = 24,
        n_iter: int = 10,
        m_bits: int = 1024,
        k_hash: int = 5,
        seed: int = 0,
        backend: Backend = "triton",
    ) -> None:
        super().__init__()
        if filter_kind not in ("none", "bloom", "clause"):
            raise ValueError(
                f"silvertorch supports filter_kind in (none, bloom, clause); "
                f"got {filter_kind!r}."
            )
        self._filter_kind = filter_kind
        device = item_embs.device
        if filter_kind == "bloom":
            if item_attrs_narrow is None:
                raise ValueError(
                    "silvertorch on filter_kind=bloom needs item_attrs_narrow "
                    "for the codesigned IVF build"
                )
            self.idx = SilverTorch(
                k=k,
                n_lists=n_lists,
                n_probe=n_probe,
                filter="bloom",
                m_bits=m_bits,
                k_hash=k_hash,
                n_iter=n_iter,
                seed=seed,
                backend=backend,
            ).to(device)
            self.idx.register_index(item_embs, item_clause_attrs=item_attrs_narrow)
        elif filter_kind == "clause":
            if item_attrs_narrow is None:
                raise ValueError(
                    "silvertorch on filter_kind=clause needs item_attrs_narrow "
                    "for the codesigned exact-clause IVF build"
                )
            self.idx = SilverTorch(
                k=k,
                n_lists=n_lists,
                n_probe=n_probe,
                filter="exact",
                n_iter=n_iter,
                seed=seed,
                backend=backend,
            ).to(device)
            self.idx.register_index(
                item_embs,
                item_clause_attrs=item_attrs_narrow,
                clause_is_reverse=clause_is_reverse,
            )
        else:
            self.idx = SilverTorch(
                k=k,
                n_lists=n_lists,
                n_probe=n_probe,
                n_iter=n_iter,
                seed=seed,
                backend=backend,
            ).to(device)
            self.idx.register_index(item_embs)
        self.algo_modules = collect_modules(self.idx, filter_mod=None)
        self.compile(dynamic=True, mode="reduce-overhead")

    def forward(self, q: Tensor, qa_narrow: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self._filter_kind in ("bloom", "clause"):
            assert qa_narrow is not None
            return self.idx(q, query_clause_attrs=qa_narrow)
        return self.idx(q, query_clause_attrs=None)
