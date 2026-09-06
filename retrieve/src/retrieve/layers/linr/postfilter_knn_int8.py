from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import LinrBackend, RetrievalModule, check_backend
from retrieve.layers.utils.topk import masked_topk


def _quantize_int8_global(t: Tensor) -> Tensor:
    """Symmetric global INT8 quantization, one scalar scale (SilverTorch §3.2): rank-preserving
    (``dot_int = const * true_dot``), so topk runs directly on the int32 result. Returns codes
    only."""
    abs_max = t.abs().amax().clamp_min(1e-8)
    return (t / abs_max * 127.0).round().clamp(-128, 127).to(torch.int8)


class PostfilterKNNInt8(RetrievalModule):
    """Single-stage int8 dense scoring + optional mask + top-K, int32 end-to-end (SilverTorch §3.2):
    ``torch._int_mm`` (int8×int8 → int32) feeds ``torch.topk`` directly with no fp32
    intermediate, and the global scales make ``dot_int`` a monotonic transform of cosine, so
    topk ordering is exact up to ties introduced by the ``>>5`` range compression (boundary
    ties are quality-equivalent). Storage is one ``[D, N]`` int8 buffer, half of
    ``PostfilterKNN``; ``backend=`` has no effect."""

    # cuBLAS LtGemm's int8 kernel requires M >= 17 (see docs/system/kernels.md →
    # PostfilterKNNInt8); smaller batches are zero-padded to this M and sliced back.
    _PAD_M = 17

    item_codes_t: Tensor  # [D, N_padded] int8

    def __init__(self, k: int, backend: LinrBackend = "triton") -> None:
        super().__init__()
        check_backend(backend, LinrBackend)
        self.k = k
        self.backend = backend
        self._n_real = 0

    def register_index(self, item_embs: Tensor) -> None:
        codes = _quantize_int8_global(item_embs)  # [N, D] int8
        # _int_mm needs N (after transpose) a multiple of 8; pad with zero items (sliced off in
        # forward), quantize before padding so the global scale is unaffected.
        n = codes.shape[0]
        self._n_real = n
        pad_n = (-n) % 8
        if pad_n:
            pad = codes.new_zeros((pad_n, codes.shape[1]))
            codes = torch.cat([codes, pad], dim=0)
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

        # int32 → fp16 for topk: >>5 brings worst-case |dot| ≈ D·127² (~2²¹) under fp16's ~2¹⁶ range
        # while preserving order; fp16 also halves CUB radix-select passes (2 vs 4).
        scores = (dots >> 5).to(torch.float16)
        if mask is not None:
            return masked_topk(scores, self.k, valid=mask)
        return masked_topk(scores, self.k)
