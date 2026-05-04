"""SilverTorch — IVF + INT8 ANN, two valid construction modes.

* ``filter_kind="bloom"``: codesigned bloom-fused IVF. Item bloom
  signatures over the narrow attrs are baked into the IVF at register
  time and filter during the probe; the query forward passes
  ``query_clause_attrs=qa_narrow`` so the kernel checks bloom bits inline.
* ``filter_kind="none"``: plain IVF + INT8, no filter.

``filter_kind="clause"`` is rejected at construction. Post-mask IVF
probe (mask gated outside the kernel after probe) systematically
under-recalls because masked-in items outside the ``n_probe`` nearest
clusters never get scored — silvertorch is meant to do the filtering
*inside* the probe (codesigned bloom) or not at all.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from retrieve import SilverTorch


class SilvertorchAlgo:
    is_cpu = False

    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        filter_kind: str = "none",
        item_attrs_narrow: Tensor | None = None,
        n_lists: int = 1024,
        n_probe: int = 24,
        n_iter: int = 10,
        m_bits: int = 1024,
        k_hash: int = 5,
        seed: int = 0,
    ) -> None:
        if filter_kind not in ("none", "bloom"):
            raise ValueError(
                f"silvertorch supports filter_kind in (none, bloom); "
                f"got {filter_kind!r} — use codesigned bloom or no filter."
            )
        self._fused = filter_kind == "bloom"
        device = item_embs.device
        if self._fused:
            if item_attrs_narrow is None:
                raise ValueError(
                    "silvertorch on filter_kind=bloom needs item_attrs_narrow "
                    "for the codesigned IVF build"
                )
            self.idx = SilverTorch(
                k=k,
                n_lists=n_lists,
                n_probe=n_probe,
                m_bits=m_bits,
                k_hash=k_hash,
                n_iter=n_iter,
                seed=seed,
            ).to(device)
            self.idx.register_index(item_embs, item_clause_attrs=item_attrs_narrow)
        else:
            self.idx = SilverTorch(
                k=k,
                n_lists=n_lists,
                n_probe=n_probe,
                n_iter=n_iter,
                seed=seed,
            ).to(device)
            self.idx.register_index(item_embs)
        self.modules: list[nn.Module] = [self.idx]

    def forward(self, q: Tensor, qa_narrow: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self._fused:
            assert qa_narrow is not None
            return self.idx(q, query_clause_attrs=qa_narrow, mask=None)
        return self.idx(q, query_clause_attrs=None, mask=None)
