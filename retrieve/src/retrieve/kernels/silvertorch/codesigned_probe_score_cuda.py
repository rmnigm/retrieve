"""Host side of the CUDA C++ SilverTorch backend (kernels in cuda/codesigned_probe_score.cu).

Lazy JIT build, eager ``_impl``s for the tuner and parity tests, and three
``@torch.library.custom_op`` wrappers mirroring the Triton ops — plain, bloom-filtered
and exact-clause-filtered scoring. The two filtered paths differ only in which phase-2
kernel fills the 1-bit-per-item mask; phase 3 is filter-agnostic. Importing this module
never triggers a build, so CPU-only machines can import and collect freely.

Design, layout contract, build requirements, and constraints:
docs/system/kernels.md § codesigned_probe_score_cuda.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from retrieve.layers.utils.quantize import quantize_int8

_CU_SOURCE = Path(__file__).resolve().parent / "cuda" / "codesigned_probe_score.cu"
_EXT_NAME = "retrieve_cps_cuda"


@dataclass(frozen=True)
class CodesignedProbeScoreCudaConfig:
    block_p: int  # probed items per block (phase-3 scoring kernel)
    num_warps: int  # warps per block; each warp owns block_p / num_warps items
    # Items a segment keeps in flight per loop iteration; 1, 2 or 4 (a template
    # parameter of the specialized scorer, so the .cu enumerates the values).
    # 1 is the original one-item-per-iteration loop. Memory-level parallelism
    # only: the arithmetic, and therefore bit-exactness, is identical at any value.
    unroll: int = 1


# Re-tune on the target arch via `uv run tune-kernels codesigned-probe-score-cuda` and
# paste the printed line here. The phase-2 mask kernel is not swept (fixed 256-thread
# blocks).
DEFAULT_CONFIG = CodesignedProbeScoreCudaConfig(block_p=256, num_warps=4, unroll=1)


class ToolchainMissing(ImportError):
    """There is no CUDA toolchain on this machine at all — no visible device, or no
    ``nvcc`` on PATH / under ``$CUDA_HOME``.

    Deliberately distinct from a plain ``ImportError``: a *missing* toolchain is a
    "you cannot run this here" condition that tests skip on, while a toolchain that
    exists and then failed to compile is a bug that must fail loudly. Silently
    skipping the second case is exactly how a broken first GPU run looks green."""


# Explicit memo of the one-shot build outcome: the extension module, or the exception
# it failed with. Deliberately not ``functools.cache``, whose key would include
# ``verbose`` — two calls differing only in verbosity would then compile twice and,
# worse, ``is_available()`` (verbose=False) would not memoize the failure that
# ``ensure_built(verbose=True)`` had already hit.
_EXT_MEMO: object | None = None


def _resolve_nvcc() -> str | None:
    """``nvcc`` from PATH, else ``$CUDA_HOME/bin/nvcc`` (the order cpp_extension uses)."""
    found = shutil.which("nvcc")
    if found:
        return found
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root:
            candidate = Path(root) / "bin" / "nvcc"
            if candidate.exists():
                return str(candidate)
    return None


def _check_nvcc_major(nvcc: str) -> None:
    """Raise ImportError if the system toolkit's major differs from the torch wheel's.

    ``cpp_extension.load`` runs torch's own ``_check_cuda_version`` only on the
    *setuptools* path, never on this JIT path — so a CUDA 11 nvcc against a cu12x
    wheel gets all the way to an opaque compile or link error. Catch it here, naming
    both versions, because that is the single most likely first-GPU-run failure."""
    wheel_cuda = torch.version.cuda
    if wheel_cuda is None:  # CPU-only torch build; the load below will say so
        return
    probe = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=False)
    match = re.search(r"release (\d+)\.(\d+)", probe.stdout)
    if match is None:  # unparseable banner — let the real build speak for itself
        return
    if int(match.group(1)) != int(wheel_cuda.split(".")[0]):
        raise ImportError(
            f"CUDA major-version mismatch: {nvcc} reports CUDA {match.group(1)}."
            f"{match.group(2)}, but this torch wheel was built against CUDA "
            f"{wheel_cuda}. The JIT extension links against the wheel's runtime, so "
            "the majors must match. Install a matching system toolkit or point "
            "CUDA_HOME at one. See docs/plans/cuda-silvertorch-handoff.md §2."
        )


def _build_ext(verbose: bool):
    """One build attempt; returns the module or the exception to memoize (never raises)."""
    if not torch.cuda.is_available():
        return ToolchainMissing(
            "torch reports no CUDA device, so the SilverTorch CUDA extension cannot "
            "be built or run here."
        )
    nvcc = _resolve_nvcc()
    if nvcc is None:
        return ToolchainMissing(
            "no `nvcc` on PATH or under $CUDA_HOME/bin. torch's cu12x wheels ship the "
            "CUDA runtime but not the compiler; a system CUDA toolkit is required. "
            "See docs/plans/cuda-silvertorch-handoff.md §2."
        )
    try:
        _check_nvcc_major(nvcc)
    except ImportError as e:
        return e

    from torch.utils.cpp_extension import load

    try:
        return load(
            name=_EXT_NAME,
            sources=[str(_CU_SOURCE)],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=verbose,
        )
    except Exception as e:  # ninja missing, nvcc error, link error, ...
        failure = ImportError(
            f"retrieve CUDA extension failed to build with {nvcc}. Requirements: "
            "`ninja` (installed by `uv sync` from retrieve's dev group) and a writable "
            "TORCH_EXTENSIONS_DIR. See docs/plans/cuda-silvertorch-handoff.md §2 / F1.\n"
            f"Underlying error:\n{e}"
        )
        failure.__cause__ = e
        return failure


def _load_ext(verbose: bool = False):
    """JIT-compile and import the extension (ninja-cached; a no-op after the first build).

    The outcome — module or exception — is memoized on the first call, so a failure is
    reported identically (and instantly) to every later caller instead of retrying a
    ~1-minute build. ``verbose`` is therefore honored only by whichever call builds
    first; the runbook's ``ensure_built(verbose=True)`` is step 1 for that reason.

    Raises ``ToolchainMissing`` when there is no toolchain to build with, and a plain
    ``ImportError`` carrying the nvcc/ninja output when there is one and it failed."""
    global _EXT_MEMO
    if _EXT_MEMO is None:
        _EXT_MEMO = _build_ext(verbose)
    if isinstance(_EXT_MEMO, BaseException):
        raise _EXT_MEMO
    return _EXT_MEMO


def ensure_built(verbose: bool = False) -> None:
    """Build (or verify the cached build of) the extension now — runbook sanity step."""
    _load_ext(verbose=verbose)


def is_available() -> bool:
    """True iff CUDA is present and the extension builds (or is already cached).

    A capability probe: it flattens "no toolchain" and "the build broke" into one
    ``False``. Tests want those apart — see ``tests/conftest.require_cps_cuda``."""
    try:
        _load_ext()
    except ImportError:
        return False
    return True


def words_per_cluster(max_size: int) -> int:
    """Mask/index words per cluster span: each cluster occupies ``ceil(max_size / 64)``
    int64 words in the transposed bloom index, so spans stay word-aligned."""
    return (max_size + 63) // 64


def build_transposed_sigs(bloom_sigs: Tensor, padded_cluster_items: Tensor) -> Tensor:
    """Rotate the bloom index into the transposed, cluster-major layout the kernel reads.

    Row ``m`` of the result is a bit-vector over padded IVF slots: bit ``s % 64`` of
    word ``c * wpc + s // 64`` is bit ``m`` of ``bloom_sigs[padded_cluster_items[c, s]]``
    (0 for ``-1`` padding). Shape ``[m_bits, n_lists * wpc]`` int64 with
    ``wpc = words_per_cluster(max_size)``; every cluster is a contiguous, word-aligned
    span. This bit layout is the contract cps_bloom_mask_kernel decodes — changing it
    means changing the kernel.

    Host-side, one-time at ``register_index``. Costs ``m_bits`` iterations of a few
    small GPU ops each (1024 at the default ``m_bits``)."""
    if bloom_sigs.dim() != 2 or padded_cluster_items.dim() != 2:
        raise ValueError("bloom_sigs must be [N, W] and padded_cluster_items [n_lists, max_size]")
    n_lists, max_size = padded_cluster_items.shape
    w = bloom_sigs.shape[1]
    wpc = words_per_cluster(max_size)
    device = bloom_sigs.device

    valid = padded_cluster_items >= 0
    safe = padded_cluster_items.clamp_min(0)
    shifts = torch.arange(64, device=device, dtype=torch.int64)
    pad_tail = wpc * 64 - max_size

    out = torch.empty(w * 64, n_lists * wpc, dtype=torch.int64, device=device)
    zero = torch.zeros((), dtype=torch.int64, device=device)
    for word in range(w):
        # Per-slot signature word, cluster-major; padding slots contribute 0 bits.
        slot_words = torch.where(valid, bloom_sigs[:, word][safe], zero)
        for j in range(64):
            bits = (slot_words >> j) & 1  # [n_lists, max_size]
            if pad_tail:
                bits = torch.nn.functional.pad(bits, (0, pad_tail))
            packed = (bits.view(n_lists, wpc, 64) << shifts).sum(dim=-1)
            out[word * 64 + j] = packed.reshape(-1)
    return out


def _cps_cuda_prep(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    *,
    query_bits: Tensor | None,
    bloom_sigs_t: Tensor | None,
    probe_ids: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, bool, int]:
    """Validation + contiguity + buffers — the CUDA mirror of the Triton ``_cps_prep``.

    Shares the ``quantize_int8`` query path so kernel inputs are byte-identical across
    backends, which is what makes the bit-exactness parity test meaningful. The score
    buffer is ``torch.empty`` because phase 3 writes every slot in [0, P)."""
    if query.dim() != 2 or flat_probed_items.dim() != 2:
        raise ValueError("query must be [B, D] and flat_probed_items [B, P]")
    if item_codes.dtype != torch.int8:
        raise TypeError(f"item_codes must be int8, got {item_codes.dtype}")
    has_mask = query_bits is not None
    if has_mask and (bloom_sigs_t is None or probe_ids is None):
        raise ValueError("bloom_sigs_t and probe_ids are required when query_bits is provided")

    b, p = flat_probed_items.shape
    max_size = 0
    if has_mask:
        assert probe_ids is not None
        n_probe = probe_ids.shape[1]
        if p % n_probe != 0:
            raise ValueError(f"P ({p}) must be n_probe ({n_probe}) x max_cluster_size")
        max_size = p // n_probe

    q_codes, q_scales = quantize_int8(query)
    q_codes, q_scales = q_codes.contiguous(), q_scales.contiguous()
    flat_probed_items = flat_probed_items.contiguous()

    all_scores = torch.empty((b, p), dtype=torch.float32, device=query.device)
    return q_codes, q_scales, flat_probed_items, all_scores, has_mask, max_size


def _bloom_partial_mask_cuda_impl(
    query_bits: Tensor,
    bloom_sigs_t: Tensor,
    probe_ids: Tensor,
    max_size: int,
) -> Tensor:
    """Phase 2 alone (eager, for tests): 1-bit-per-item masks over the probed clusters.

    Returns ``[B, n_probe * wpc]`` int64; bit ``s % 64`` of word ``pi * wpc + s // 64``
    is the verdict for slot ``s`` of the ``pi``-th probed cluster."""
    b, n_probe = probe_ids.shape
    wpc = words_per_cluster(max_size)
    mask = torch.empty((b, n_probe * wpc), dtype=torch.int64, device=query_bits.device)
    _load_ext().bloom_partial_mask(
        query_bits.contiguous(),
        bloom_sigs_t.contiguous(),
        probe_ids.contiguous(),
        mask,
        wpc,
    )
    return mask


def _clause_partial_mask_cuda_impl(
    flat_items: Tensor,
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
    max_size: int,
) -> Tensor:
    """Phase 2 alone (eager, for tests): 1-bit-per-item masks over the probed clusters
    from the exact AND-of-OR clause predicate.

    Same output layout as ``_bloom_partial_mask_cuda_impl`` — ``[B, n_probe * wpc]`` int64,
    bit ``s % 64`` of word ``pi * wpc + s // 64`` is the verdict for slot ``s`` of the
    ``pi``-th probed cluster — so phase 3 consumes either one unchanged. Ids are read from
    ``flat_items`` (``P = n_probe * max_size``), which is why exact mode needs no
    cluster-major attribute copy and no extra registered buffer."""
    b, p = flat_items.shape
    n_probe = p // max_size
    wpc = words_per_cluster(max_size)
    mask = torch.empty((b, n_probe * wpc), dtype=torch.int64, device=flat_items.device)
    _load_ext().clause_partial_mask(
        flat_items.contiguous(),
        item_clause_attrs.contiguous(),
        # The kernel reads torch.bool storage as bytes; .to() is a no-op when already bool.
        clause_is_reverse.contiguous().to(torch.bool),
        query_clause_attrs.contiguous(),
        mask,
        max_size,
    )
    return mask


def _cps_cuda_score_topk(
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
    cfg: CodesignedProbeScoreCudaConfig,
) -> tuple[Tensor, Tensor]:
    """Phase 3 + phase 4, shared by every eager ``_impl`` here: launch the masked dp4a
    scorer over the prepared buffers, then the host top-K epilogue.

    The scoring kernel is filter-agnostic — bloom, exact and no-filter differ only in what
    (if anything) filled ``mask`` — so this tail is written once."""
    _load_ext().cps_scores(
        q_codes,
        q_scales,
        mask,
        flat_probed_items,
        item_codes.contiguous(),
        all_scores,
        float(global_scale),
        has_mask,
        max_size,
        cfg.block_p,
        cfg.num_warps,
        cfg.unroll,
    )
    # Same epilogue as the Triton _cps_finish: P >= k is guaranteed by the layer's
    # index-build assert, so topk(k) needs no min/pad path.
    topk_scores, topk_local = torch.topk(all_scores, k, dim=1)
    topk_ids = flat_probed_items.gather(1, topk_local)
    return topk_ids, topk_scores


def _codesigned_probe_score_cuda_impl(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    k: int,
    *,
    query_bits: Tensor | None = None,
    bloom_sigs_t: Tensor | None = None,
    probe_ids: Tensor | None = None,
    config: CodesignedProbeScoreCudaConfig | None = None,
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
        mask = _bloom_partial_mask_cuda_impl(query_bits, bloom_sigs_t, probe_ids, max_size)
    else:
        # 1x1 int64 dummy: has_mask=False gates every mask load, so it is never
        # dereferenced (same convention as the Triton wrapper's qb dummies).
        mask = torch.empty(1, 1, dtype=torch.int64, device=query.device)
    return _cps_cuda_score_topk(
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


def _cpse_cuda_prep(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    *,
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
    max_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Validation + contiguity + buffers for the exact path — the CUDA mirror of the Triton
    ``_cpse_prep``, which is why the messages match it word for word.

    Every check here has a ``TORCH_CHECK`` twin in the ``.cu`` launchers; those are for
    direct extension calls, these are the ones users see. ``max_size`` is explicit because
    the clause mask kernel reads ids from ``flat_probed_items`` and never sees cluster ids,
    so it cannot derive the cluster-span width itself."""
    if query.dim() != 2 or flat_probed_items.dim() != 2:
        raise ValueError("query must be [B, D] and flat_probed_items [B, P]")
    if item_codes.dtype != torch.int8:
        raise TypeError(f"item_codes must be int8, got {item_codes.dtype}")
    if item_clause_attrs.dim() != 3:
        raise ValueError("item_clause_attrs must be [N, C, A_max]")
    if query_clause_attrs.dim() != 2:
        raise ValueError("query_clause_attrs must be [B, C]")

    b = query.shape[0]
    p = flat_probed_items.shape[1]
    c = item_clause_attrs.shape[1]
    b_q, c_q = query_clause_attrs.shape
    if c != c_q:
        raise ValueError(f"clause-count mismatch: items C={c}, query C={c_q}")
    if b_q != b:
        raise ValueError(f"batch mismatch: query B={b}, query_clause_attrs B={b_q}")
    if clause_is_reverse.shape != (c,):
        raise ValueError(f"clause_is_reverse must be [{c}], got {tuple(clause_is_reverse.shape)}")
    if max_size <= 0 or p % max_size != 0:
        raise ValueError(f"P ({p}) must be a positive multiple of max_size ({max_size})")

    q_codes, q_scales = quantize_int8(query)
    q_codes, q_scales = q_codes.contiguous(), q_scales.contiguous()
    flat_probed_items = flat_probed_items.contiguous()

    # torch.empty is safe: phase 3 writes every slot in [0, P) (real dot or -inf).
    all_scores = torch.empty((b, p), dtype=torch.float32, device=query.device)
    return q_codes, q_scales, flat_probed_items, all_scores


def _codesigned_probe_score_exact_cuda_impl(
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
    config: CodesignedProbeScoreCudaConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Eager entry point for tune scripts / parity tests; the compiled path goes through the
    custom op. Mirrors the Triton ``_codesigned_probe_score_exact_impl`` argument for
    argument plus ``max_size`` (the padded cluster width, so the mask kernel can lay its
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
    mask = _clause_partial_mask_cuda_impl(
        flat_probed_items,
        item_clause_attrs,
        clause_is_reverse,
        query_clause_attrs,
        max_size,
    )
    return _cps_cuda_score_topk(
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
    "retrieve::codesigned_probe_score_cuda", mutates_args=(), device_types="cuda"
)
def codesigned_probe_score_cuda(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Plain int8 ANN scoring (no attribute filter), CUDA C++ backend — sibling of the
    Triton ``retrieve::codesigned_probe_score`` op. Requires P >= k."""
    return _codesigned_probe_score_cuda_impl(query, flat_probed_items, item_codes, global_scale, k)


@codesigned_probe_score_cuda.register_fake
def _(query, flat_probed_items, item_codes, global_scale, k):
    b = query.shape[0]
    return (
        query.new_empty((b, k), dtype=torch.int64),
        query.new_empty((b, k), dtype=torch.float32),
    )


@torch.library.custom_op(
    "retrieve::codesigned_probe_score_bloom_cuda", mutates_args=(), device_types="cuda"
)
def codesigned_probe_score_bloom_cuda(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    query_bits: Tensor,
    bloom_sigs_t: Tensor,
    probe_ids: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Int8 ANN scoring behind the partial bloom filter, CUDA C++ backend — counterpart
    of the Triton ``retrieve::codesigned_probe_score_bloom`` op. A separate op rather
    than one op with an Optional, so the layer just routes."""
    return _codesigned_probe_score_cuda_impl(
        query,
        flat_probed_items,
        item_codes,
        global_scale,
        k,
        query_bits=query_bits,
        bloom_sigs_t=bloom_sigs_t,
        probe_ids=probe_ids,
    )


@codesigned_probe_score_bloom_cuda.register_fake
def _(query, flat_probed_items, item_codes, query_bits, bloom_sigs_t, probe_ids, global_scale, k):
    b = query.shape[0]
    return (
        query.new_empty((b, k), dtype=torch.int64),
        query.new_empty((b, k), dtype=torch.float32),
    )


@torch.library.custom_op(
    "retrieve::codesigned_probe_score_exact_cuda", mutates_args=(), device_types="cuda"
)
def codesigned_probe_score_exact_cuda(
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
    """Int8 ANN scoring behind the exact AND-of-OR clause filter, CUDA C++ backend —
    counterpart of the Triton ``retrieve::codesigned_probe_score_exact``, and bit-identical
    to it (same predicate, same int32 dot, same fp32 epilogue, same host top-K).

    ``max_size`` is the padded cluster width, i.e. ``P // n_probe``: the phase-2 kernel
    reads ids from ``flat_probed_items`` rather than from cluster ids, so it needs the span
    width to write the cluster-major mask layout phase 3 decodes. It is the layer's
    ``padded_cluster_items.shape[1]`` — a Python int, never a device read. Requires
    P >= k."""
    return _codesigned_probe_score_exact_cuda_impl(
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


@codesigned_probe_score_exact_cuda.register_fake
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
