from __future__ import annotations

import torch
from torch import Tensor

from retrieve.kernels.triton.linr.oporp_1bit_match_topk import oporp_1bit_match_topk
from retrieve.layers.linr.one_bit_knn import OneBitKNN
from retrieve.layers.utils.compact import compact_mask


class OneBitKNNTriton(OneBitKNN):
    """1-bit Sign-OPORP scoring with the fused Triton kernel.

    All three paths (full, masked, candidates) route through
    ``oporp_1bit_match_topk``: full path scans every item with contiguous
    int64-word loads; masked / candidate paths use indirect loads through a
    positive-indices buffer. Same operation in either case — XOR + popcount
    + ``D - 2 * hamming`` — so there's no separate dequant or fp32 dot.

    Unlike the fp32 paths, this module does not branch on pass rate: 1-bit
    popcount is roughly 10× cheaper per item than an fp32 dot, so the gather
    overhead never crosses the dense matmul break-even and a single sparse
    path is fine.
    """

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
        candidate_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        query_bits = self._project_query(query)

        if candidate_ids is not None:
            counts = torch.full(
                (candidate_ids.shape[0],),
                candidate_ids.shape[1],
                dtype=torch.long,
                device=query.device,
            )
            return oporp_1bit_match_topk(
                query_bits,
                self.item_bits,
                self.k,
                positive_indices=candidate_ids,
                counts=counts,
            )

        if mask is None:
            return oporp_1bit_match_topk(query_bits, self.item_bits, self.k)

        positive_indices, counts = compact_mask(mask)
        if int(counts.max().item()) == 0:
            b = query.shape[0]
            device = query.device
            return (
                torch.full((b, self.k), -1, dtype=torch.long, device=device),
                torch.full((b, self.k), float("-inf"), device=device),
            )
        return oporp_1bit_match_topk(
            query_bits,
            self.item_bits,
            self.k,
            positive_indices=positive_indices,
            counts=counts,
        )
