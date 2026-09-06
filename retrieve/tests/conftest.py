"""Global test fixtures and CUDA gate.

`retrieve` is a GPU-only library, so the entire suite is skipped without CUDA.
Tests build tensors explicitly with ``device="cuda"`` (via the helpers below
or directly) — using ``torch.set_default_device("cuda")`` would clash with
library code that intentionally creates CPU-side generators.
"""

from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="retrieve is GPU-only; CUDA required")
        for it in items:
            it.add_marker(skip)


def require_cps_cuda() -> None:
    """Gate the calling test on the CUDA C++ extension, distinguishing the two ways it
    can be absent.

    - **No toolchain** (no CUDA device, or no ``nvcc`` on PATH / under ``$CUDA_HOME``)
      → ``skip``. The suite is expected to run on boxes that cannot build it.
    - **A toolchain that failed to build** → ``fail``, with the nvcc/ninja output.
      This is the case worth being loud about: a compile error silently skipping is
      exactly what a broken first GPU run would look like, and it would look green.

    The first call pays the one-time JIT compile; later calls hit the memoized
    outcome in the wrapper (and the ninja cache)."""
    # Full-module-path import: the package __init__ re-exports the *op* under the
    # same name as the module, so `from retrieve.kernels.silvertorch import ...`
    # would grab the op and shadow the module.
    from retrieve.kernels.silvertorch.codesigned_probe_score_cuda import (
        ToolchainMissing,
        ensure_built,
    )

    try:
        ensure_built()
    except ToolchainMissing as e:
        pytest.skip(f"retrieve CUDA extension unavailable: {e}")
    except ImportError as e:
        pytest.fail(
            f"the CUDA toolchain is present but the retrieve CUDA extension failed "
            f"to build — this is a real failure, not a skip:\n{e}",
            pytrace=False,
        )


def require_cps_cute() -> None:
    """Gate the calling test on the CuTe DSL backend, with the same split as
    ``require_cps_cuda``.

    - **No DSL** (``nvidia-cutlass-dsl`` — the ``cute`` extra — not installed, or no
      CUDA device) → ``skip``. The suite is expected to run on boxes without the extra.
    - **A DSL that is present and failed to compile a kernel** → ``fail``, with the DSL
      error text. Skipping here would let a broken port look green.

    The first call pays the one-time import + compile of the ``D=128`` scorer; later
    calls hit the memoized outcome in the host module."""
    # Full-module-path import: the package __init__ re-exports the *op* under the
    # same name as the module, so `from retrieve.kernels.silvertorch import ...`
    # would grab the op and shadow the module.
    from retrieve.kernels.silvertorch.codesigned_probe_score_cute import (
        CuteMissing,
        ensure_built,
    )

    try:
        ensure_built()
    except CuteMissing as e:
        pytest.skip(f"retrieve CuTe DSL backend unavailable: {e}")
    except ImportError as e:
        pytest.fail(
            f"the CuTe DSL is present but the retrieve cute kernels failed to import or "
            f"compile — this is a real failure, not a skip:\n{e}",
            pytrace=False,
        )


def require_official() -> None:
    """Gate the calling test on Meta's official SilverTorch ops (``torch.ops.st.*``,
    the ``official`` extra), with the same split as ``require_cps_cuda``:

    - **Not runnable here** (``silvertorch`` not installed, or no CUDA device) →
      ``skip``. The suite is expected to run on boxes without the extra — the Mac
      collects and skips it.
    - **Installed but broken** (``silvertorch`` imports but ``silvertorch._C`` failed
      to build / load, or the pinned sha lacks an op the adapter calls) → ``fail``
      with the loader's message. A build failure that silently skipped would look
      green on the first GPU run.

    The first call pays the extension load; later calls hit the memo in the adapter."""
    # Full-module-path import, as for the cuda / cute gates: the package __init__
    # re-exports ops under module names, so import the module explicitly.
    from retrieve.kernels.silvertorch.official import OfficialMissing, ensure_loaded

    try:
        ensure_loaded()
    except OfficialMissing as e:
        pytest.skip(f"official SilverTorch backend unavailable: {e}")
    except ImportError as e:
        pytest.fail(
            f"silvertorch is installed but the official ops failed to load — this is a "
            f"real failure, not a skip:\n{e}",
            pytrace=False,
        )


def make_index(
    n: int,
    d: int,
    *,
    normalized: bool = True,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Random unit-norm item embeddings, ``[N, D]``, on CUDA."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    embs = torch.randn(n, d, generator=g, device="cuda", dtype=dtype)
    if normalized:
        embs = embs / embs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return embs


def make_query(
    b: int,
    d: int,
    *,
    normalized: bool = True,
    seed: int = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Random unit-norm query embeddings, ``[B, D]``, on CUDA."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, d, generator=g, device="cuda", dtype=dtype)
    if normalized:
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return q


def make_mask(b: int, n: int, *, pass_rate: float, seed: int = 2) -> torch.Tensor:
    """Random boolean mask ``[B, N]`` with the given expected pass rate."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(b, n, generator=g, device="cuda") < pass_rate


def make_attrs(
    n: int,
    c: int,
    a_max: int,
    *,
    n_vocab: int = 50,
    pad_rate: float = 0.3,
    seed: int = 3,
) -> torch.Tensor:
    """Random per-item clause attributes ``[N, C, A_max]`` with -1 padding."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    attrs = torch.randint(0, n_vocab, (n, c, a_max), generator=g, dtype=torch.long, device="cuda")
    pad = torch.rand(n, c, a_max, generator=g, device="cuda") < pad_rate
    attrs[pad] = -1
    return attrs


def make_query_attrs(
    b: int,
    c: int,
    *,
    n_vocab: int = 50,
    inactive_rate: float = 0.2,
    seed: int = 4,
) -> torch.Tensor:
    """Random query clause attributes ``[B, C]`` with some clauses inactive (-1)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(0, n_vocab, (b, c), generator=g, dtype=torch.long, device="cuda")
    inactive = torch.rand(b, c, generator=g, device="cuda") < inactive_rate
    q[inactive] = -1
    return q


def valid_id_set(ids: torch.Tensor, scores: torch.Tensor, b: int) -> set[int]:
    """Set of ids in row ``b`` with a finite score and a non-padded id."""
    valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
    return set(ids[b][valid].tolist())


def assert_topk_id_sets_match(
    out_ids: torch.Tensor,
    out_scores: torch.Tensor,
    ref_ids: torch.Tensor,
    ref_scores: torch.Tensor,
    b: int,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    """Per-row id-set comparison with tie tolerance at the K-th boundary.

    Tensor-core matmul (`tl.dot`) and torch `@` differ in accumulator order
    enough to flip the K-th-place tiebreak when two items have ~atol score
    spread. Symmetric-difference ids must lie within `atol` of their side's
    finite-score minimum.
    """
    k = out_ids.shape[1]
    out_pairs = [
        (out_ids[b, j].item(), out_scores[b, j].item())
        for j in range(k)
        if torch.isfinite(out_scores[b, j]) and out_ids[b, j].item() >= 0
    ]
    ref_pairs = [
        (ref_ids[b, j].item(), ref_scores[b, j].item())
        for j in range(k)
        if torch.isfinite(ref_scores[b, j]) and ref_ids[b, j].item() >= 0
    ]
    out_set = {p[0] for p in out_pairs}
    ref_set = {p[0] for p in ref_pairs}
    if out_set == ref_set:
        return
    out_min = min(s for _, s in out_pairs) if out_pairs else float("-inf")
    ref_min = min(s for _, s in ref_pairs) if ref_pairs else float("-inf")
    for i in ref_set - out_set:
        s = next(sc for idx, sc in ref_pairs if idx == i)
        assert s <= ref_min + atol + rtol * abs(ref_min), (
            f"row {b}: ref-only id {i} score={s:.6f} not at boundary {ref_min:.6f}"
        )
    for i in out_set - ref_set:
        s = next(sc for idx, sc in out_pairs if idx == i)
        assert s <= out_min + atol + rtol * abs(out_min), (
            f"row {b}: out-only id {i} score={s:.6f} not at boundary {out_min:.6f}"
        )


def recall_at_k(approx_ids: torch.Tensor, exact_ids: torch.Tensor) -> float:
    """Mean per-row set overlap of ``approx`` vs ``exact``, divided by K."""
    b, k = approx_ids.shape
    overlap = sum(len(set(approx_ids[i].tolist()) & set(exact_ids[i].tolist())) for i in range(b))
    return overlap / (b * k)


def assert_recall_monotone(recalls: list[float], *, slack: float = 0.02) -> None:
    """Assert ``recalls`` is non-decreasing within ``slack`` between adjacent entries."""
    for prev, nxt in zip(recalls, recalls[1:]):
        assert nxt >= prev - slack, f"recall regressed: {recalls}"
