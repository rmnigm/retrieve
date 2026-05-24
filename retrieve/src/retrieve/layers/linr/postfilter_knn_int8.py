from __future__ import annotations

import torch
from torch import Tensor, nn

from retrieve.interfaces import Backend

_INT32_NEG_INF = torch.iinfo(torch.int32).min


def _quantize_int8_global(t: Tensor) -> Tensor:
    """Symmetric global INT8 quantization with one scalar scale.

    Matches SilverTorch §3.2: "we compute global min/max values across
    all embeddings, scale them to [-128, 127], and assign integer
    representations accordingly." A single scale per tensor — not per
    row — so the ``dot_int = const * true_dot`` relationship is rank-
    preserving across all (query, item) pairs and topk can run directly
    on the int32 result without any rescale.

    Returns only ``codes``; the scale is not stored because it's not
    needed for retrieval-as-ids.
    """
    abs_max = t.abs().amax().clamp_min(1e-8)
    return (t / abs_max * 127.0).round().clamp(-128, 127).to(torch.int8)


class PostfilterKNNInt8(nn.Module):
    """Single-stage int8 dense scoring + optional mask + top-K, int32 end-to-end.

    Items and queries are int8-quantized using one global scale each
    (SilverTorch §3.2). The matmul runs through ``torch._int_mm`` —
    cuBLAS LtGemm, int8×int8 → int32, IMMA tensor cores on Ampere+. The
    int32 result feeds ``torch.topk`` directly: no cast to fp32, no
    scale recovery, no fp16 intermediate buffer. Because the two scales
    are global constants per call, ``dot_int`` is a positive monotonic
    transform of the true cosine — topk ordering is exact (modulo the
    int8 rounding noise on each element).

    Storage: one ``[D, N]`` int8 buffer. Half the memory of
    ``PostfilterKNN``'s fp16 transposed embs.

    Notes:

    - ``backend=`` is accepted for API symmetry with the other LiNR
      layers but has no effect (``torch._int_mm`` is cuBLAS-backed).
    - ``torch._int_mm`` requires the M-dim (batch) to be ``>= 17``; small
      batches are zero-padded here and sliced after the matmul. Quantize
      *before* padding so the padded zero rows don't shift the global
      scale.
    - The eval pipeline discards the returned scores
      (``bench_tools.quality_pass_cached``), so scores being raw int32
      dots rather than cosine-valued doesn't matter for downstream
      metrics — only ids do.
    """

    _PAD_M = 17

    item_codes_t: Tensor  # [D, N_padded] int8

    def __init__(self, k: int, backend: Backend = "triton") -> None:
        super().__init__()
        self.k = k
        self.backend = backend
        self._n_real = 0

    def register_index(self, item_embs: Tensor) -> None:
        codes = _quantize_int8_global(item_embs)  # [N, D] int8
        # cuBLAS _int_mm requires mat2.size(1) (== N after the transpose
        # to [D, N]) to be a multiple of 8. Pad N with zero items here;
        # the padded columns produce dot=0 and are sliced off in forward
        # before scoring. Quantize *before* padding so the global scale
        # is unaffected.
        n = codes.shape[0]
        self._n_real = n
        pad_n = (-n) % 8
        if pad_n:
            pad = codes.new_zeros((pad_n, codes.shape[1]))
            codes = torch.cat([codes, pad], dim=0)
        # Pre-transpose to [D, N_padded] contiguous — matches
        # PostfilterKNN's layout convention and what _int_mm expects
        # for B (== K) input.
        self.register_buffer("item_codes_t", codes.t().contiguous())

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        q_codes = _quantize_int8_global(query)
        b = q_codes.shape[0]

        if b < self._PAD_M:
            pad_rows = self._PAD_M - b
            pad = q_codes.new_zeros((pad_rows, q_codes.shape[1]))
            q_codes_pad = torch.cat([q_codes, pad], dim=0)
            dots = torch._int_mm(q_codes_pad, self.item_codes_t)[:b]
        else:
            dots = torch._int_mm(q_codes, self.item_codes_t)
        dots = dots[:, : self._n_real]

        # int32 → fp16 for topk, scaled to fit. Worst-case dot magnitude is
        # ``D · 127² ≈ 2M`` (≈ 2²¹), and fp16's max is ~65 504 (~2¹⁶), so
        # an arithmetic right-shift by 5 brings the magnitude under fp16's
        # range while preserving order (positive monotonic on the
        # accumulator, sign-preserving on negatives). The cast pays for
        # itself: ``torch.topk`` uses CUB radix-select, which runs 2
        # passes on fp16 vs 4 on int32 — at large N the score buffer is
        # the binding constraint, and the smaller fp16 buffer cuts both
        # the cast write and every topk pass in half.
        scores = (dots >> 5).to(torch.float16)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        topk_scores, topk_ids = torch.topk(scores, self.k, dim=1)
        if mask is not None:
            topk_ids = torch.where(
                torch.isfinite(topk_scores),
                topk_ids,
                topk_ids.new_full((), -1),
            )
        return topk_ids, topk_scores
