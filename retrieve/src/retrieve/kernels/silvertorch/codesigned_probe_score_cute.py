"""Host side of the CuTe DSL SilverTorch backend (kernels in cute/codesigned_probe_score.py).

Mirrors ``codesigned_probe_score_cuda.py`` one to one: lazy import + compile of the DSL
kernels, eager ``_impl``s for the tuner and parity tests, and three
``@torch.library.custom_op`` wrappers — plain, bloom-filtered and exact-clause-filtered
scoring. It consumes the same transposed ``bloom_sigs_t`` and writes the same
cluster-major 1-bit mask layout as the C++ backend (``words_per_cluster``,
``build_transposed_sigs`` and the ``_prep``s are imported from it, not copied), so a cute
checkpoint is byte-identical to a cuda checkpoint. Importing this module never imports
``cutlass``, so machines without the ``cute`` extra (or a GPU) import and collect freely.

Every ``TORCH_CHECK`` of the ``.cu`` launchers has a Python twin here — the DSL kernels
have no device-side checks at all. Design, layout contract and constraints:
docs/system/kernels.md § codesigned_probe_score_cuda.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass

import torch
from torch import Tensor

from retrieve.kernels.silvertorch.codesigned_probe_score_cuda import (
    _cps_cuda_prep,
    _cpse_cuda_prep,
    build_transposed_sigs,
    words_per_cluster,
)

__all__ = [
    "CodesignedProbeScoreCuteConfig",
    "CuteMissing",
    "DEFAULT_CONFIG",
    "build_transposed_sigs",
    "codesigned_probe_score_bloom_cute",
    "codesigned_probe_score_cute",
    "codesigned_probe_score_exact_cute",
    "ensure_built",
    "is_available",
    "words_per_cluster",
]


@dataclass(frozen=True)
class CodesignedProbeScoreCuteConfig:
    block_p: int  # probed items per block (phase-3 scoring kernel)
    num_warps: int  # warps per block; each warp owns block_p / num_warps items
    # Items a segment keeps in flight per loop iteration; 1, 2 or 4 (a Constexpr of the
    # specialized scorer, so the compile cache enumerates the values). 1 is the original
    # one-item-per-iteration loop. Memory-level parallelism only: the arithmetic, and
    # therefore bit-exactness, is identical at any value.
    unroll: int = 1


# Re-tune on the target arch via `uv run tune-kernels codesigned-probe-score-cute` (and
# its `-exact-cute` twin, which shares this line) and paste the printed line here.
# A100 (2026-09-02, plan §5): the landscape is flat — no config won more than 2 of the
# 11 regimes of the two sweeps — so the reconciled pick is the geometric-mean winner
# over all 11 (1.056x the per-regime best, worst 1.144x; the cuda default (128, 8, 1)
# sits at 1.086x / 1.311x); a re-sweep after the host-overhead trims put every top
# config within 1.3 % of each other, and this one wins the bloom B=16 kernel outright
# (39.8 µs vs 51.2 at (128, 8, 1)). unroll=4 costs 48 registers/thread (34 at
# unroll=1) — no occupancy cliff.
DEFAULT_CONFIG = CodesignedProbeScoreCuteConfig(block_p=256, num_warps=8, unroll=4)


class CuteMissing(ImportError):
    """The CuTe DSL cannot run here at all — ``nvidia-cutlass-dsl`` (the ``cute``
    extra) is not installed, or torch sees no CUDA device.

    Deliberately distinct from a plain ``ImportError``: a *missing* DSL is a "you cannot
    run this here" condition that tests skip on, while a DSL that is present and then
    failed to compile a kernel is a bug that must fail loudly — the same split
    ``ToolchainMissing`` makes for the C++ backend."""


# Explicit memo of the one-shot load outcome: the device module, or the exception it
# failed with — same reasoning as the cuda module's ``_EXT_MEMO``.
_DEV_MEMO: object | None = None


def _import_dev(verbose: bool):
    """One import + first-compile attempt; returns the module or the exception to
    memoize (never raises)."""
    if not torch.cuda.is_available():
        return CuteMissing("torch reports no CUDA device, so the CuTe DSL kernels cannot run here.")
    if torch.cuda.get_device_capability() < (8, 0):
        # The segment reduction is `redux.sync` (sm_80+); unlike the C++ backend there is
        # no butterfly fallback (plan §4), so pre-Ampere is "cannot run here", not a bug.
        return CuteMissing(
            "the CuTe DSL kernels need an sm_80+ GPU (redux.sync); found "
            f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}."
        )
    try:
        from retrieve.kernels.silvertorch.cute import codesigned_probe_score as dev
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] in ("cutlass", "cuda"):
            return CuteMissing(
                "nvidia-cutlass-dsl is not installed; sync the `cute` extra "
                f"(`uv sync --all-packages --extra cute`). Underlying error: {e}"
            )
        return e
    except Exception as e:  # the DSL imported but its own initialization failed
        failure = ImportError(f"CuTe DSL device module failed to import:\n{e}")
        failure.__cause__ = e
        return failure
    # Compile one representative scorer now so a broken DSL fails loudly and early —
    # the D=128 no-filter build at the default unroll, which the layer's first plain
    # query then reuses (~130 ms per specialization, no disk cache).
    unroll = DEFAULT_CONFIG.unroll
    t0 = time.perf_counter()
    try:
        dev.compile_score(8, False, unroll, device=torch.cuda.current_device())
    except Exception as e:
        failure = ImportError(
            f"CuTe DSL failed to compile cps_score_kernel<8, false, {unroll}>:\n{e}"
        )
        failure.__cause__ = e
        return failure
    if verbose:
        dt = time.perf_counter() - t0
        print(f"cute: compiled cps_score_kernel<8, false, {unroll}> in {dt:.2f}s")
    return dev


def _load_dev(verbose: bool = False):
    """Import the device module and compile its first kernel (memoized).

    Raises ``CuteMissing`` when the DSL is absent or there is no GPU, and a plain
    ``ImportError`` carrying the DSL error when it is present and failed."""
    global _DEV_MEMO
    if _DEV_MEMO is None:
        _DEV_MEMO = _import_dev(verbose)
    if isinstance(_DEV_MEMO, BaseException):
        raise _DEV_MEMO
    return _DEV_MEMO


def ensure_built(verbose: bool = False) -> None:
    """Import the DSL and compile the ``D=128`` scorer now — runbook sanity step."""
    _load_dev(verbose=verbose)


def is_available() -> bool:
    """True iff CUDA and the DSL are present and the first kernel compiles.

    A capability probe: it flattens "no DSL" and "the compile broke" into one
    ``False``. Tests want those apart — see ``CuteMissing``."""
    try:
        _load_dev()
    except ImportError:
        return False
    return True


_CLAUSE_TABLE = {(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)}

# Host-side launch cost matters here: a DSL launch has no C++ launcher to hide behind,
# so these two helpers are the cheapest correct forms (plan §5, host-overhead table).
# `torch._C._cuda_getCurrentRawStream` is what inductor's generated code uses (0.2 µs;
# `torch.cuda.current_stream().cuda_stream` is ~5 µs).
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _stream_handle(device_index: int) -> int:
    if _raw_stream is not None:
        return _raw_stream(device_index)
    return torch.cuda.current_stream(device_index).cuda_stream


def _device_guard(device_index: int):
    """``torch.cuda.device(...)`` only when the tensors' device is not already current
    (the guard costs ~2 µs; the check ~0.3 µs)."""
    if device_index == torch.cuda.current_device():
        return contextlib.nullcontext()
    return torch.cuda.device(device_index)


def _check_i64_contiguous(**tensors: Tensor) -> None:
    for name, t in tensors.items():
        if t.dtype != torch.int64:
            raise TypeError(f"{name} must be int64, got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _bloom_partial_mask_cute_impl(
    query_bits: Tensor,
    bloom_sigs_t: Tensor,
    probe_ids: Tensor,
    max_size: int,
) -> Tensor:
    """Phase 2 alone (eager, for tests): 1-bit-per-item masks over the probed clusters.

    Returns ``[B, n_probe * wpc]`` int64; bit ``s % 64`` of word ``pi * wpc + s // 64``
    is the verdict for slot ``s`` of the ``pi``-th probed cluster. Ids are trusted as
    in-range layer buffers, exactly as in the ``.cu``."""
    dev = _load_dev()
    query_bits, bloom_sigs_t = query_bits.contiguous(), bloom_sigs_t.contiguous()
    probe_ids = probe_ids.contiguous()
    _check_i64_contiguous(query_bits=query_bits, bloom_sigs_t=bloom_sigs_t, probe_ids=probe_ids)
    if not query_bits.is_cuda:
        raise ValueError("query_bits must be a CUDA tensor")
    b, w = query_bits.shape
    n_probe = probe_ids.shape[1]
    wpc = words_per_cluster(max_size)
    if bloom_sigs_t.shape[0] != w * 64:
        raise ValueError("bloom_sigs_t rows must equal m_bits = W*64")
    if bloom_sigs_t.shape[1] % wpc != 0:
        raise ValueError("bloom_sigs_t width must be n_lists*wpc")
    if probe_ids.shape[0] != b:
        raise ValueError("batch mismatch")
    if b > 65535:
        raise ValueError(f"B exceeds grid.y limit, got {b}")
    mask_words = n_probe * wpc
    mask = torch.empty((b, mask_words), dtype=torch.int64, device=query_bits.device)

    device = query_bits.device.index
    with _device_guard(device):
        dev.compile_bloom_mask(device=device)(
            query_bits.data_ptr(),
            bloom_sigs_t.data_ptr(),
            probe_ids.data_ptr(),
            mask.data_ptr(),
            bloom_sigs_t.shape[1],
            mask_words,
            n_probe,
            wpc,
            w,
            b,
            _stream_handle(device),
        )
    return mask


def _clause_partial_mask_cute_impl(
    flat_items: Tensor,
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
    max_size: int,
) -> Tensor:
    """Phase 2 alone (eager, for tests): 1-bit-per-item masks over the probed clusters
    from the exact AND-of-OR clause predicate.

    Same output layout as ``_bloom_partial_mask_cute_impl`` — ``[B, n_probe * wpc]``
    int64 — so phase 3 consumes either one unchanged. Ids are read from ``flat_items``
    (``P = n_probe * max_size``), which is why exact mode needs no cluster-major
    attribute copy and no extra registered buffer."""
    dev = _load_dev()
    flat_items, item_clause_attrs = flat_items.contiguous(), item_clause_attrs.contiguous()
    query_clause_attrs = query_clause_attrs.contiguous()
    # The kernel reads torch.bool storage as bytes; .to() is a no-op when already bool.
    clause_is_reverse = clause_is_reverse.contiguous().to(torch.bool)
    _check_i64_contiguous(
        flat_items=flat_items, item_attrs=item_clause_attrs, query_attrs=query_clause_attrs
    )
    if not flat_items.is_cuda:
        raise ValueError("flat_items must be a CUDA tensor")
    if item_clause_attrs.dim() != 3:
        raise ValueError("item_attrs must be [N, C, A_max]")
    if query_clause_attrs.dim() != 2:
        raise ValueError("query_attrs must be [B, C]")
    b, p = flat_items.shape
    c, a_max = item_clause_attrs.shape[1], item_clause_attrs.shape[2]
    if max_size <= 0 or p % max_size != 0:
        raise ValueError("P must be n_probe * max_size")
    if query_clause_attrs.shape[1] != c:
        raise ValueError("clause-count mismatch: items C != query C")
    if clause_is_reverse.shape != (c,):
        raise ValueError("is_reverse must be [C]")
    if query_clause_attrs.shape[0] != b:
        raise ValueError("batch mismatch")
    if b > 65535:
        raise ValueError(f"B exceeds grid.y limit, got {b}")
    wpc = words_per_cluster(max_size)
    mask_words = (p // max_size) * wpc
    mask = torch.empty((b, mask_words), dtype=torch.int64, device=flat_items.device)

    # Register fast path for C <= 4, A in {1, 2, 4}, C*A <= 8; else the (0, 0) loop.
    cc, aa = (c, a_max) if (c, a_max) in _CLAUSE_TABLE else (0, 0)
    device = flat_items.device.index
    with _device_guard(device):
        dev.compile_clause_mask(cc, aa, device=device)(
            flat_items.data_ptr(),
            item_clause_attrs.data_ptr(),
            clause_is_reverse.data_ptr(),
            query_clause_attrs.data_ptr(),
            mask.data_ptr(),
            p,
            max_size,
            mask_words,
            wpc,
            c,
            a_max,
            b,
            _stream_handle(device),
        )
    return mask


def _cps_cute_scores(
    q_codes: Tensor,
    q_scales: Tensor,
    mask: Tensor,
    flat_items: Tensor,
    item_codes: Tensor,
    out_scores: Tensor,
    *,
    global_scale: float,
    has_mask: bool,
    max_size: int,
    cfg: CodesignedProbeScoreCuteConfig,
) -> None:
    """Phase 3: the ``cps_scores`` launcher of the ``.cu`` — validation, the
    ``D -> (SEG | generic) x HAS_MASK x UNROLL`` dispatch, and the launch into
    ``out_scores`` (every slot in [0, P) is written: a real dot or ``-inf``)."""
    dev = _load_dev()
    if not q_codes.is_cuda:
        raise ValueError("q_codes must be a CUDA tensor")
    for name, t in (
        ("q_codes", q_codes),
        ("q_scales", q_scales),
        ("mask", mask),
        ("flat_items", flat_items),
        ("item_codes", item_codes),
        ("out_scores", out_scores),
    ):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if q_codes.dtype != torch.int8 or item_codes.dtype != torch.int8:
        raise TypeError("q_codes and item_codes must be int8")
    if q_scales.dtype != torch.float32 or out_scores.dtype != torch.float32:
        raise TypeError("q_scales and out_scores must be float32")
    if flat_items.dtype != torch.int64 or mask.dtype != torch.int64:
        raise TypeError("flat_items and mask must be int64")
    b, d = q_codes.shape
    p = flat_items.shape[1]
    if d % 4 != 0:
        raise ValueError(f"D must be a multiple of 4 for dp4a packing, got {d}")
    if item_codes.shape[1] != d:
        raise ValueError("item_codes D mismatch")
    if flat_items.shape[0] != b or out_scores.shape[0] != b:
        raise ValueError("batch mismatch")
    if out_scores.shape[1] != p:
        raise ValueError("out_scores P mismatch")
    if b > 65535:
        raise ValueError(f"B exceeds grid.y limit, got {b}")
    if not 1 <= cfg.num_warps <= 32:
        raise ValueError("num_warps must be in [1, 32]")
    if cfg.block_p <= 0 or cfg.block_p % cfg.num_warps != 0:
        raise ValueError("block_p must be a positive multiple of num_warps")
    if cfg.unroll not in (1, 2, 4):
        raise ValueError(f"unroll must be 1, 2 or 4, got {cfg.unroll}")
    wpc = 1
    if has_mask:
        if max_size <= 0 or p % max_size != 0:
            raise ValueError("P must be n_probe * max_size")
        wpc = words_per_cluster(max_size)
        if mask.shape != (b, (p // max_size) * wpc):
            raise ValueError("mask must be [B, n_probe*wpc]")
    mask_words = mask.shape[1]

    # int4 row loads need 16 B-aligned bases (a storage-offset view may not be);
    # SEG = D/16 for the table D, anything else takes the generic kernel, whose
    # 32-bit word loads still need 4 B (the kernels read the int8 rows as Int32).
    q_addr, codes_addr = q_codes.data_ptr(), item_codes.data_ptr()
    if q_addr % 4 or codes_addr % 4:
        raise ValueError("q_codes and item_codes must start at 4-byte aligned addresses")
    vec_ok = q_addr % 16 == 0 and codes_addr % 16 == 0
    seg = {64: 4, 128: 8, 256: 16}.get(d if vec_ok else 0)
    device = q_codes.device.index
    with _device_guard(device):
        if seg is not None:
            launch = dev.compile_score(seg, has_mask, cfg.unroll, device=device)
            tail = ()
        else:
            launch = dev.compile_score_generic(has_mask, device=device)  # ignores unroll
            tail = (d // 4,)
        launch(
            q_addr,
            q_scales.data_ptr(),
            mask.data_ptr(),
            flat_items.data_ptr(),
            codes_addr,
            out_scores.data_ptr(),
            float(global_scale),
            p,
            max_size,
            mask_words,
            wpc,
            b,
            cfg.block_p,
            cfg.num_warps,
            *tail,
            _stream_handle(device),
        )


def _cps_cute_score_topk(
    q_codes: Tensor,
    q_scales: Tensor,
    mask: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    all_scores: Tensor,
    *,
    global_scale: float,
    has_mask: bool,
    max_size: int,
    k: int,
    cfg: CodesignedProbeScoreCuteConfig,
) -> tuple[Tensor, Tensor]:
    """Phase 3 + phase 4, shared by every eager ``_impl`` here: launch the masked dp4a
    scorer over the prepared buffers, then the host top-K epilogue.

    The scoring kernel is filter-agnostic — bloom, exact and no-filter differ only in what
    (if anything) filled ``mask`` — so this tail is written once."""
    _cps_cute_scores(
        q_codes,
        q_scales,
        mask,
        flat_probed_items,
        item_codes.contiguous(),
        all_scores,
        global_scale=global_scale,
        has_mask=has_mask,
        max_size=max_size,
        cfg=cfg,
    )
    # Same epilogue as the Triton _cps_finish: P >= k is guaranteed by the layer's
    # index-build assert, so topk(k) needs no min/pad path.
    topk_scores, topk_local = torch.topk(all_scores, k, dim=1)
    topk_ids = flat_probed_items.gather(1, topk_local)
    return topk_ids, topk_scores


def _codesigned_probe_score_cute_impl(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    k: int,
    *,
    query_bits: Tensor | None = None,
    bloom_sigs_t: Tensor | None = None,
    probe_ids: Tensor | None = None,
    config: CodesignedProbeScoreCuteConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Eager entry point for tune scripts / parity tests; the compiled path goes through
    the custom ops. Bloom inputs are the transposed index + probed cluster ids, where the
    Triton ``_codesigned_probe_score_impl`` takes row-wise signatures. Requires P >= k."""
    cfg = config if config is not None else DEFAULT_CONFIG
    q_codes, q_scales, flat_probed_items, all_scores, has_mask, max_size = _cps_cuda_prep(
        query,
        flat_probed_items,
        item_codes,
        query_bits=query_bits,
        bloom_sigs_t=bloom_sigs_t,
        probe_ids=probe_ids,
    )
    if has_mask:
        assert query_bits is not None and bloom_sigs_t is not None and probe_ids is not None
        mask = _bloom_partial_mask_cute_impl(query_bits, bloom_sigs_t, probe_ids, max_size)
    else:
        # 1x1 int64 dummy: has_mask=False gates every mask load, so it is never
        # dereferenced (same convention as the cuda and Triton wrappers).
        mask = torch.empty(1, 1, dtype=torch.int64, device=query.device)
    return _cps_cute_score_topk(
        q_codes,
        q_scales,
        mask,
        flat_probed_items,
        item_codes,
        all_scores,
        global_scale=global_scale,
        has_mask=has_mask,
        max_size=max_size,
        k=k,
        cfg=cfg,
    )


def _codesigned_probe_score_exact_cute_impl(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    k: int,
    *,
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
    max_size: int,
    config: CodesignedProbeScoreCuteConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Eager entry point for tune scripts / parity tests; the compiled path goes through the
    custom op. Mirrors the cuda ``_codesigned_probe_score_exact_cuda_impl`` argument for
    argument (``max_size`` is the padded cluster width, so the mask kernel can lay its
    output out cluster-major). Requires P >= k."""
    cfg = config if config is not None else DEFAULT_CONFIG
    q_codes, q_scales, flat_probed_items, all_scores = _cpse_cuda_prep(
        query,
        flat_probed_items,
        item_codes,
        item_clause_attrs=item_clause_attrs,
        clause_is_reverse=clause_is_reverse,
        query_clause_attrs=query_clause_attrs,
        max_size=max_size,
    )
    mask = _clause_partial_mask_cute_impl(
        flat_probed_items,
        item_clause_attrs,
        clause_is_reverse,
        query_clause_attrs,
        max_size,
    )
    return _cps_cute_score_topk(
        q_codes,
        q_scales,
        mask,
        flat_probed_items,
        item_codes,
        all_scores,
        global_scale=global_scale,
        has_mask=True,
        max_size=max_size,
        k=k,
        cfg=cfg,
    )


@torch.library.custom_op(
    "retrieve::codesigned_probe_score_cute", mutates_args=(), device_types="cuda"
)
def codesigned_probe_score_cute(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Plain int8 ANN scoring (no attribute filter), CuTe DSL backend — sibling of the
    cuda ``retrieve::codesigned_probe_score_cuda`` op. Requires P >= k."""
    return _codesigned_probe_score_cute_impl(query, flat_probed_items, item_codes, global_scale, k)


@codesigned_probe_score_cute.register_fake
def _(query, flat_probed_items, item_codes, global_scale, k):
    b = query.shape[0]
    return (
        query.new_empty((b, k), dtype=torch.int64),
        query.new_empty((b, k), dtype=torch.float32),
    )


@torch.library.custom_op(
    "retrieve::codesigned_probe_score_bloom_cute", mutates_args=(), device_types="cuda"
)
def codesigned_probe_score_bloom_cute(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    query_bits: Tensor,
    bloom_sigs_t: Tensor,
    probe_ids: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Int8 ANN scoring behind the partial bloom filter, CuTe DSL backend — counterpart
    of the cuda ``retrieve::codesigned_probe_score_bloom_cuda`` op. A separate op rather
    than one op with an Optional, so the layer just routes."""
    return _codesigned_probe_score_cute_impl(
        query,
        flat_probed_items,
        item_codes,
        global_scale,
        k,
        query_bits=query_bits,
        bloom_sigs_t=bloom_sigs_t,
        probe_ids=probe_ids,
    )


@codesigned_probe_score_bloom_cute.register_fake
def _(query, flat_probed_items, item_codes, query_bits, bloom_sigs_t, probe_ids, global_scale, k):
    b = query.shape[0]
    return (
        query.new_empty((b, k), dtype=torch.int64),
        query.new_empty((b, k), dtype=torch.float32),
    )


@torch.library.custom_op(
    "retrieve::codesigned_probe_score_exact_cute", mutates_args=(), device_types="cuda"
)
def codesigned_probe_score_exact_cute(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
    global_scale: float,
    k: int,
    max_size: int,
) -> tuple[Tensor, Tensor]:
    """Int8 ANN scoring behind the exact AND-of-OR clause filter, CuTe DSL backend —
    counterpart of the cuda ``retrieve::codesigned_probe_score_exact_cuda`` op and
    bit-identical to it (same predicate, same int32 dot, same fp32 epilogue, same host
    top-K).

    ``max_size`` is the padded cluster width, i.e. ``P // n_probe``: the phase-2 kernel
    reads ids from ``flat_probed_items`` rather than from cluster ids, so it needs the span
    width to write the cluster-major mask layout phase 3 decodes. It is the layer's
    ``padded_cluster_items.shape[1]`` — a Python int, never a device read. Requires
    P >= k."""
    return _codesigned_probe_score_exact_cute_impl(
        query,
        flat_probed_items,
        item_codes,
        global_scale,
        k,
        item_clause_attrs=item_clause_attrs,
        clause_is_reverse=clause_is_reverse,
        query_clause_attrs=query_clause_attrs,
        max_size=max_size,
    )


@codesigned_probe_score_exact_cute.register_fake
def _(
    query,
    flat_probed_items,
    item_codes,
    item_clause_attrs,
    clause_is_reverse,
    query_clause_attrs,
    global_scale,
    k,
    max_size,
):
    b = query.shape[0]
    return (
        query.new_empty((b, k), dtype=torch.int64),
        query.new_empty((b, k), dtype=torch.float32),
    )
