from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from retrieve.layers.utils.quantize import quantize_int8


@dataclass(frozen=True)
class CodesignedProbeScoreExactConfig:
    block_p: int
    num_warps: int
    num_stages: int = 3


# Single default the library ships with. Mirrors ``codesigned_probe_score``'s
# A100 tuning; re-tune on a new arch via ``evaluation/scripts/tune_kernels.py``.
DEFAULT_CONFIG = CodesignedProbeScoreExactConfig(block_p=256, num_warps=4)


@triton.jit
def _codesigned_probe_score_exact_kernel(
    q_codes_ptr,
    q_scales_ptr,
    flat_items_ptr,
    item_codes_ptr,
    item_attrs_ptr,
    is_reverse_ptr,
    query_attrs_ptr,
    out_scores_ptr,
    global_scale,
    P: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    A_MAX: tl.constexpr,
    stride_qcb,
    stride_qcd,
    stride_qs,
    stride_fb,
    stride_fp,
    stride_cn,
    stride_cd,
    stride_ian,
    stride_iac,
    stride_iaa,
    stride_qab,
    stride_qac,
    stride_ob,
    stride_op,
    BLOCK_P: tl.constexpr,
):
    # tile on axis-0 (CUDA grid_x ≤ 2^31), batch on axis-1 (grid_y ≤ 65535):
    # n_probe × max_cluster_size can be millions, which would overflow grid_y.
    tile_id = tl.program_id(0)
    bid = tl.program_id(1)

    p_off = tile_id * BLOCK_P + tl.arange(0, BLOCK_P)
    p_valid = p_off < P

    d_off = tl.arange(0, D)
    # Query is pre-quantized in the wrapper — symmetric per-row int8 with a
    # fp32 scale, paying one amax + div per batch row outside the loop.
    q_codes = tl.load(q_codes_ptr + bid * stride_qcb + d_off * stride_qcd)
    q_scale = tl.load(q_scales_ptr + bid * stride_qs)

    item_ids = tl.load(
        flat_items_ptr + bid * stride_fb + p_off * stride_fp,
        mask=p_valid,
        other=-1,
    ).to(tl.int64)
    # Masked load returned -1 for OOB lanes, so id>=0 already implies p_valid.
    valid = item_ids >= 0
    safe_ids = tl.where(valid, item_ids, 0)

    # --- exact-clause filter (AND across C, OR over A_max within each clause) ---
    # Per-item gather is C × A_max int64 — at C=2, A_max=2 that's 32 B/item vs
    # the bloom variant's W × 8 = 128 B/item at M=1024. The clause loop is
    # static-unrolled by Triton; the AND/OR/XOR chain stays in registers.
    keep = valid
    for c in tl.static_range(C):
        q_c = tl.load(query_attrs_ptr + bid * stride_qab + c * stride_qac)
        rev_c = tl.load(is_reverse_ptr + c).to(tl.int1)

        clause_match = tl.full([BLOCK_P], 0, tl.int1)
        for a in tl.static_range(A_MAX):
            ia = tl.load(
                item_attrs_ptr
                + safe_ids * stride_ian
                + c * stride_iac
                + a * stride_iaa,
                mask=valid,
                other=-1,
            )
            clause_match = clause_match | (ia == q_c)

        clause_match = clause_match ^ rev_c
        # ``q_c == -1`` marks an inactive clause — always passes.
        inactive = q_c == -1
        clause_match = clause_match | inactive

        keep = keep & clause_match

    codes = tl.load(
        item_codes_ptr + safe_ids[:, None] * stride_cn + d_off[None, :] * stride_cd,
        mask=keep[:, None],
        other=0,
    )

    # int8 × int8 → int32 matmul (paper §4.2). dp4a path identical to the
    # bloom-variant kernel; see codesigned_probe_score.py for the bandwidth
    # rationale.
    q_codes_2d = q_codes[None, :]
    codes_T = tl.trans(codes)
    dots_2d = tl.dot(q_codes_2d, codes_T, out_dtype=tl.int32)
    dots_i32 = tl.sum(dots_2d, axis=0)

    dots = dots_i32.to(tl.float32) * q_scale * global_scale
    dots = tl.where(keep, dots, float("-inf"))

    tl.store(
        out_scores_ptr + bid * stride_ob + p_off * stride_op,
        dots,
        mask=p_valid,
    )


def _codesigned_probe_score_exact_impl(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    k: int,
    *,
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
    config: CodesignedProbeScoreExactConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused phase-2+3 with an **exact-clause** filter instead of Bloom.

    Sibling of ``codesigned_probe_score``: same int8×int8 → int32 dp4a path,
    same ``-inf`` masking + top-K epilogue. The filter is the exact-clause
    predicate from ``clause_mask`` — AND across ``C`` clauses, OR across
    ``A_max`` attribute values per clause, XOR with ``clause_is_reverse``,
    OR with the ``q_c == -1`` inactive-clause sentinel.

    For each (query, probed-item) cell: evaluate the clause predicate
    against ``item_clause_attrs[item_id]``, and (if it passes) score
    ``(item_codes[id] · q_codes[b])_i32 * q_scale[b] * global_scale``. Items
    failing the filter or with id == -1 get score ``-inf``.

    The per-item clause gather (``[BLOCK_P, C, A_max]`` int64) never touches
    HBM — it lives in registers/SRAM, same memory win as the bloom variant.

    Inputs:
        query:               [B, D]            fp32 — quantized to int8 in the wrapper.
        flat_probed_items:   [B, P]            int64 (-1 padding for empty cluster slots).
        item_codes:          [N, D]            int8 — symmetric per-tensor codes.
        global_scale:        Python float — single scale, paired with item_codes.
        item_clause_attrs:   [N, C, A_max]     int64 — per-item clause values (-1 pad).
        clause_is_reverse:   [C]               bool — invert match per clause.
        query_clause_attrs:  [B, C]            int64 — per-query clause values (-1 inactive).

    Returns ``(ids[B, K], scores[B, K])``. Requires ``P >= k`` (the layer's
    ``__init__`` asserts ``k <= n_probe * max_cluster_size``); per-row "no
    candidate passed" cells already get ``-inf`` / ``-1`` from the kernel +
    ``flat_probed_items`` padding semantics.

    Eager entry point for tune scripts and parity tests. The compiled
    path goes through the ``@triton_op`` wrapper ``codesigned_probe_score_exact``
    which mirrors this body inline so that ``wrap_triton`` is textually in
    the decorated function's source — torch.export's kernel registry
    requires that.
    """
    if query.dim() != 2 or flat_probed_items.dim() != 2:
        raise ValueError("query must be [B, D] and flat_probed_items [B, P]")
    if item_codes.dtype != torch.int8:
        raise TypeError(f"item_codes must be int8, got {item_codes.dtype}")
    if item_clause_attrs.dim() != 3:
        raise ValueError("item_clause_attrs must be [N, C, A_max]")
    if query_clause_attrs.dim() != 2:
        raise ValueError("query_clause_attrs must be [B, C]")

    b, d = query.shape
    p = flat_probed_items.shape[1]
    _, c, a_max = item_clause_attrs.shape
    b_q, c_q = query_clause_attrs.shape
    if c != c_q:
        raise ValueError(f"clause-count mismatch: items C={c}, query C={c_q}")
    if b_q != b:
        raise ValueError(f"batch mismatch: query B={b}, query_clause_attrs B={b_q}")
    if clause_is_reverse.shape != (c,):
        raise ValueError(f"clause_is_reverse must be [{c}], got {tuple(clause_is_reverse.shape)}")

    # Per-batch query int8 quantization. One amax + scalar div per row.
    q_codes, q_scales = quantize_int8(query)
    q_codes = q_codes.contiguous()
    q_scales = q_scales.contiguous()

    flat_probed_items = flat_probed_items.contiguous()
    item_codes = item_codes.contiguous()
    item_clause_attrs = item_clause_attrs.contiguous()
    # Triton can't load native torch.bool; mirror clause_mask.py.
    clause_is_reverse = clause_is_reverse.contiguous().to(torch.int8)
    query_clause_attrs = query_clause_attrs.contiguous()

    # `torch.empty` is safe: the kernel writes every slot in [0, P) — either a
    # real dot product or -inf for filtered/padding lanes — so downstream topk
    # sees deterministic values.
    all_scores = torch.empty((b, p), dtype=torch.float32, device=query.device)

    cfg = config if config is not None else DEFAULT_CONFIG

    # Tile axis on grid_x (≤ 2^31) since num_tiles can exceed grid_y/grid_z's
    # 65535 limit at large n_probe × max_cluster_size.
    grid = (triton.cdiv(p, cfg.block_p), b)

    _codesigned_probe_score_exact_kernel[grid](
        q_codes,
        q_scales,
        flat_probed_items,
        item_codes,
        item_clause_attrs,
        clause_is_reverse,
        query_clause_attrs,
        all_scores,
        float(global_scale),
        P=p,
        D=d,
        C=c,
        A_MAX=a_max,
        stride_qcb=q_codes.stride(0),
        stride_qcd=q_codes.stride(1),
        stride_qs=q_scales.stride(0),
        stride_fb=flat_probed_items.stride(0),
        stride_fp=flat_probed_items.stride(1),
        stride_cn=item_codes.stride(0),
        stride_cd=item_codes.stride(1),
        stride_ian=item_clause_attrs.stride(0),
        stride_iac=item_clause_attrs.stride(1),
        stride_iaa=item_clause_attrs.stride(2),
        stride_qab=query_clause_attrs.stride(0),
        stride_qac=query_clause_attrs.stride(1),
        stride_ob=all_scores.stride(0),
        stride_op=all_scores.stride(1),
        BLOCK_P=cfg.block_p,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    # P (= n_probe × max_cluster_size) >= k by layer-construction assert,
    # so topk(k) works directly with no min/pad path.
    topk_scores, topk_local = torch.topk(all_scores, k, dim=1)
    topk_ids = flat_probed_items.gather(1, topk_local)
    return topk_ids, topk_scores


@triton_op("retrieve::codesigned_probe_score_exact", mutates_args=())
def codesigned_probe_score_exact(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Production ``@triton_op`` for exact-clause-filtered int8 ANN scoring.

    Mirrors ``_codesigned_probe_score_exact_impl`` inline so ``wrap_triton``
    appears textually in the decorated source — required by torch.export's
    kernel registry. Uses ``DEFAULT_CONFIG``; tune scripts and parity tests
    that need a non-default config call ``_codesigned_probe_score_exact_impl``
    directly.

    Layer-construction asserts ``k <= n_probe * max_cluster_size`` so
    ``torch.topk(all_scores, k)`` always has >= k lanes with no pad tail.
    """
    b, d = query.shape
    p = flat_probed_items.shape[1]
    _, c, a_max = item_clause_attrs.shape

    q_codes, q_scales = quantize_int8(query)
    q_codes = q_codes.contiguous()
    q_scales = q_scales.contiguous()
    flat_probed_items = flat_probed_items.contiguous()
    item_codes = item_codes.contiguous()
    item_clause_attrs = item_clause_attrs.contiguous()
    # Triton can't load native torch.bool; mirror clause_mask.py.
    clause_is_reverse = clause_is_reverse.contiguous().to(torch.int8)
    query_clause_attrs = query_clause_attrs.contiguous()

    all_scores = torch.empty((b, p), dtype=torch.float32, device=query.device)

    def grid(meta):
        return (triton.cdiv(p, meta["BLOCK_P"]), b)

    wrap_triton(_codesigned_probe_score_exact_kernel)[grid](
        q_codes,
        q_scales,
        flat_probed_items,
        item_codes,
        item_clause_attrs,
        clause_is_reverse,
        query_clause_attrs,
        all_scores,
        float(global_scale),
        P=p,
        D=d,
        C=c,
        A_MAX=a_max,
        stride_qcb=q_codes.stride(0),
        stride_qcd=q_codes.stride(1),
        stride_qs=q_scales.stride(0),
        stride_fb=flat_probed_items.stride(0),
        stride_fp=flat_probed_items.stride(1),
        stride_cn=item_codes.stride(0),
        stride_cd=item_codes.stride(1),
        stride_ian=item_clause_attrs.stride(0),
        stride_iac=item_clause_attrs.stride(1),
        stride_iaa=item_clause_attrs.stride(2),
        stride_qab=query_clause_attrs.stride(0),
        stride_qac=query_clause_attrs.stride(1),
        stride_ob=all_scores.stride(0),
        stride_op=all_scores.stride(1),
        BLOCK_P=DEFAULT_CONFIG.block_p,
        num_warps=DEFAULT_CONFIG.num_warps,
        num_stages=DEFAULT_CONFIG.num_stages,
    )

    topk_scores, topk_local = torch.topk(all_scores, k, dim=1)
    topk_ids = flat_probed_items.gather(1, topk_local)
    return topk_ids, topk_scores
