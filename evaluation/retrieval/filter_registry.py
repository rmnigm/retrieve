"""Filter-aware wrapper over `retrieval.registry`.

For each ``(algo, filter_kind, sweep)`` cell, builds the underlying
retrieval module via the unmodified yambda
[build_algorithm](registry.py) and threads `FilterModule` masks /
candidate ids into its forward. The resulting forward signature is

    forward(q, qa_narrow=None, qa_wide=None) -> (ids, scores)

The `_bench_primitives` quality / perf passes always pass kwargs only
when the caller supplied attribute pools — yambda's bare-`q` lambdas
keep working unmodified.

Routing per algorithm:

* ``torch_fullscan`` — `FullScanKNN` already accepts a ``mask`` kwarg,
  but applies it as a **post-filter** (mask the top-K after the matmul).
  Recall can drop below the unfiltered baseline if the filter is very
  selective. The plan documents this; perf rows tag it as ``post_filter``.
* ``linr_v3_then_v2`` — V3 with the filter mask collapses to the
  positives via ``compact_mask`` inside the kernel; V3's top-N pool
  feeds V2 as ``candidate_ids``. Both V3 and V2 accept these kwargs as
  shipped.
* ``silvertorch`` — wide / combined sweeps use the bloom-fused
  `SilverTorch` (m_bits + k_hash supplied) and pass
  ``query_clause_attrs`` directly. Narrow-only sweeps skip silvertorch
  per the plan's bench thesis.

Composition: narrow + wide masks AND'd via
[combine_masks](../../retrieve/src/retrieve/layers/filters/__init__.py).
For 2-shelf wide queries the bloom is evaluated twice (once per shelf)
and the two masks AND'd, since the registered ``item_attrs_wide`` has
``C = 1`` and bloom queries are ``[B, C]`` int64 single-attribute-per-
clause (multi-value query is out of scope per filtering-api.md).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from retrieval.config import FilterSweepCfg
from retrieval.registry import build_algorithm
from retrieve.layers.filters import BloomFilter, ClauseIndex, combine_masks
from retrieve.layers.linr.v2_triton import LiNR_V2_Triton
from retrieve.layers.linr.v3_triton import LiNR_V3_Triton
from retrieve.layers.silvertorch.main import SilverTorch
from retrieve.layers.utils.retrieval import FullScanKNN

FilterKind = str  # "none" | "clause" | "bloom" | "combined"


# Forward = (q, qa_narrow=..., qa_wide=...) -> (ids, scores)
FilteredForwardFn = Callable[..., tuple[Tensor, Tensor]]


def synthesize_query_attrs_narrow(
    eval_split_qa_narrow: Tensor,  # [N_users, C_narrow] int64 from eval_split.parquet
    sweep: FilterSweepCfg,
    n_clauses: int,
) -> Tensor:
    """Mask the per-user narrow attrs to only the clauses this sweep activates.

    Inactive clauses are coded as ``-1`` — ClauseIndex.evaluate_mask
    treats that value as "match anything for this clause". Reverse
    clauses (e.g. C1 lang_reverse) keep the target's value as-is; the
    reverse semantics are baked into the index, not into the query.
    """
    qa = eval_split_qa_narrow.clone()
    active = sweep.active_clauses or list(range(n_clauses))
    inactive_mask = torch.ones(n_clauses, dtype=torch.bool)
    for c in active:
        if not (0 <= c < n_clauses):
            raise ValueError(f"sweep {sweep.name!r}: active clause {c} out of range")
        inactive_mask[c] = False
    if inactive_mask.any():
        qa[:, inactive_mask] = -1
    return qa


def _check_bloom_no_reverse(
    filter_kind: FilterKind,
    sweep: FilterSweepCfg,
    clause_is_reverse: Tensor | None,
) -> None:
    """Paper-strict guard: BloomFilter does not implement reverse clauses."""
    if filter_kind != "bloom":
        return
    if sweep.active_clauses and clause_is_reverse is not None:
        for c in sweep.active_clauses:
            if 0 <= c < clause_is_reverse.shape[0] and bool(clause_is_reverse[c].item()):
                raise ValueError(
                    f"filter_kind=bloom + sweep {sweep.name!r}: clause {c} is "
                    "a reverse clause — BloomFilter is paper-strict (no NOT). "
                    "Use filter_kind=clause or filter_kind=combined instead."
                )


def build_filter_modules(
    filter_kind: FilterKind,
    *,
    item_attrs_narrow: Tensor | None = None,
    item_attrs_wide: Tensor | None = None,
    clause_is_reverse: Tensor | None = None,
    bloom_m_bits: int = 1024,
    bloom_k_hash: int = 5,
    device: torch.device,
) -> tuple[ClauseIndex | None, BloomFilter | None]:
    """Build the ClauseIndex / BloomFilter once per filter_kind.

    Bloom signature construction is O(N × C × A_max × k_hash) and
    consumes a transient ~m_bits-sized scratch tensor; reuse the same
    instance across all sweeps + (algo, k) cells of a kind to avoid
    paying that cost per cell.
    """
    ci: ClauseIndex | None = None
    bf: BloomFilter | None = None
    if filter_kind in ("clause", "combined"):
        if item_attrs_narrow is None:
            raise ValueError(f"{filter_kind} requires item_attrs_narrow")
        ci = ClauseIndex().to(device)
        ci.register_index(item_attrs_narrow, clause_is_reverse=clause_is_reverse)
    if filter_kind in ("bloom", "combined"):
        if item_attrs_wide is None:
            raise ValueError(f"{filter_kind} requires item_attrs_wide")
        bf = BloomFilter(m_bits=bloom_m_bits, k_hash=bloom_k_hash).to(device)
        bf.register_index(item_attrs_wide)
    return ci, bf


def build_filtered_algorithm(
    algo_name: str,
    item_embs: Tensor,
    *,
    k: int,
    filter_kind: FilterKind,
    sweep: FilterSweepCfg,
    ci: ClauseIndex | None = None,
    bf: BloomFilter | None = None,
    clause_is_reverse: Tensor | None = None,
    item_attrs_wide: Tensor | None = None,
    bloom_m_bits: int = 1024,
    bloom_k_hash: int = 5,
    algo_params: dict[str, Any] | None = None,
) -> tuple[FilteredForwardFn, list[nn.Module], bool]:
    """Build a forward + module list for one (algo, filter_kind, sweep) cell.

    `ci` / `bf` are the pre-built filter modules (use
    `build_filter_modules`) — they're shared across all (algo, k) cells
    of the same filter_kind so we don't rebuild bloom signatures per
    cell. ``item_attrs_wide`` and ``clause_is_reverse`` are still passed
    in case the underlying algo needs them at build time (silvertorch
    bloom-fused path needs item_attrs_wide).

    Returns ``(forward, modules, is_cpu)`` where ``forward(q, qa_narrow=None,
    qa_wide=None)`` runs the underlying algo with the filter applied.

    Skips silvertorch on narrow-only sweeps (per the plan's decision —
    SilverTorch's bloom path is not engaged on narrow attrs, leaving an
    IVF + INT8 ANN that doesn't add signal vs ClauseIndex + LiNR).
    """
    _check_bloom_no_reverse(filter_kind, sweep, clause_is_reverse)
    algo_params = algo_params or {}
    device = item_embs.device

    if filter_kind in ("clause", "combined") and ci is None:
        raise ValueError(f"{filter_kind} requires a pre-built ClauseIndex (ci=)")
    if filter_kind in ("bloom", "combined") and bf is None:
        raise ValueError(f"{filter_kind} requires a pre-built BloomFilter (bf=)")
    filter_modules: list[nn.Module] = [m for m in (ci, bf) if m is not None]

    def _bloom_mask(qa_wide: Tensor) -> Tensor:
        """Run M bloom passes (one per active wide-shelf field) and AND the
        results. Always returns a [B, N] bool mask.
        """
        if bf is None:
            raise RuntimeError("bloom mask requested but BloomFilter not built")
        if qa_wide.dim() == 1:
            qa_wide = qa_wide.unsqueeze(-1)
        # qa_wide: [B, M]. Bloom expects [B, C=1]. Run M passes, AND the masks.
        masks: list[Tensor] = []
        for j in range(qa_wide.shape[1]):
            masks.append(bf.evaluate_mask(qa_wide[:, j : j + 1]))
        out = masks[0]
        for m in masks[1:]:
            out = out & m
        return out

    def _combined_mask(qa_narrow: Tensor | None, qa_wide: Tensor | None) -> Tensor | None:
        """Compose the optional narrow + wide masks via element-wise AND."""
        masks: list[Tensor | None] = []
        if ci is not None and qa_narrow is not None:
            masks.append(ci.evaluate_mask(qa_narrow))
        if bf is not None and qa_wide is not None:
            masks.append(_bloom_mask(qa_wide))
        return combine_masks(*masks)

    # ----- algo-specific routing -----
    is_cpu = False  # filter_registry only handles GPU algos for now

    if algo_name == "torch_fullscan":
        idx = FullScanKNN(k=k).to(device)
        idx.register_index(item_embs)

        def fs_forward(
            q: Tensor, qa_narrow: Tensor | None = None, qa_wide: Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
            mask = _combined_mask(qa_narrow, qa_wide)
            return idx(q, mask=mask)

        modules: list[nn.Module] = [idx, *filter_modules]
        return fs_forward, modules, is_cpu

    if algo_name == "linr_v3_then_v2":
        # Cascade: V3 with filter-mask → top candidate_pool → V2 reranks those.
        candidate_pool = int(algo_params.get("candidate_pool", 5000))
        v3_seed = int(algo_params.get("v3_seed", 0))
        v3 = LiNR_V3_Triton(k=candidate_pool, seed=v3_seed).to(device)
        v3.register_index(item_embs)
        stage2 = LiNR_V2_Triton(k=k).to(device)
        stage2.register_index(item_embs)

        def linr_forward(
            q: Tensor, qa_narrow: Tensor | None = None, qa_wide: Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
            mask = _combined_mask(qa_narrow, qa_wide)
            cand_ids, _ = v3(q, mask=mask)
            return stage2(q, candidate_ids=cand_ids)

        modules = [v3, stage2, *filter_modules]
        return linr_forward, modules, is_cpu

    if algo_name == "silvertorch":
        if filter_kind == "clause":
            # Per the plan: skip silvertorch on narrow-only sweeps.
            raise _SilvertorchSkippedOnNarrow(
                f"silvertorch is skipped on filter_kind=clause (sweep {sweep.name!r}); "
                "the bench thesis is 'narrow → ClauseIndex; wide → bloom-fused SilverTorch'."
            )
        n_lists = int(algo_params.get("n_lists", 1024))
        n_probe = int(algo_params.get("n_probe", 24))
        n_iter = int(algo_params.get("n_iter", 10))
        seed = int(algo_params.get("seed", 0))

        if filter_kind in ("bloom", "combined"):
            # Bloom-fused SilverTorch — pass query_clause_attrs directly into
            # codesigned_probe_score.  External narrow mask gets AND'd in via
            # the ``mask`` kwarg.
            st_m_bits = int(algo_params.get("m_bits", bloom_m_bits))
            st_k_hash = int(algo_params.get("k_hash", bloom_k_hash))
            idx = SilverTorch(
                k=k,
                n_lists=n_lists,
                n_probe=n_probe,
                m_bits=st_m_bits,
                k_hash=st_k_hash,
                n_iter=n_iter,
                seed=seed,
            ).to(device)
            # Build bloom signatures from the same item_attrs_wide our
            # standalone BloomFilter sees, so cross-checks line up.
            idx.register_index(item_embs, item_clause_attrs=item_attrs_wide)
        else:  # filter_kind == "none"
            idx = SilverTorch(
                k=k, n_lists=n_lists, n_probe=n_probe, n_iter=n_iter, seed=seed
            ).to(device)
            idx.register_index(item_embs)

        def st_forward(
            q: Tensor, qa_narrow: Tensor | None = None, qa_wide: Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
            narrow_mask = (
                ci.evaluate_mask(qa_narrow) if (ci is not None and qa_narrow is not None) else None
            )
            # For 2-shelf, fall back to external mask combine — the fused
            # bloom path in SilverTorch only consumes a single per-clause
            # value. Detect via qa_wide shape.
            if qa_wide is not None and qa_wide.dim() == 2 and qa_wide.shape[1] > 1:
                bloom_mask_external = _bloom_mask(qa_wide)
                external = combine_masks(narrow_mask, bloom_mask_external)
                return idx(q, query_clause_attrs=None, mask=external)
            qaw = qa_wide if (qa_wide is not None and bf is not None) else None
            if qaw is not None and qaw.dim() == 1:
                qaw = qaw.unsqueeze(-1)
            return idx(q, query_clause_attrs=qaw, mask=narrow_mask)

        modules = [idx, *filter_modules]
        return st_forward, modules, is_cpu

    if algo_name == "triton_knn":
        # LiNR_V1_Triton with mask kwarg — pre-filter inside the fused matmul.
        # Plan doesn't enumerate this in the goodreads YAMLs but it's free to
        # support, and the cross-check sweep (filter_kind=none) reuses it.
        from retrieve.layers.linr.v1_triton import LiNR_V1_Triton

        idx = LiNR_V1_Triton(k=k).to(device)
        idx.register_index(item_embs)

        def kt_forward(
            q: Tensor, qa_narrow: Tensor | None = None, qa_wide: Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
            mask = _combined_mask(qa_narrow, qa_wide)
            return idx(q, mask=mask)

        modules = [idx, *filter_modules]
        return kt_forward, modules, is_cpu

    if algo_name in ("voyager_hnsw",):
        # CPU baselines don't have a filter-aware contract; the goodreads
        # configs don't list voyager. Fall through to build_algorithm with
        # filtering disabled (we apply the filter as a post-mask).
        forward_unfiltered, modules, is_cpu = build_algorithm(
            algo_name, item_embs.cpu() if not is_cpu else item_embs, k=k, params=algo_params
        )

        def voy_forward(
            q: Tensor, qa_narrow: Tensor | None = None, qa_wide: Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
            ids, scores = forward_unfiltered(q)
            mask = _combined_mask(
                qa_narrow.to(item_embs.device) if qa_narrow is not None else None,
                qa_wide.to(item_embs.device) if qa_wide is not None else None,
            )
            if mask is not None:
                # Post-filter: mask out ids whose mask row is False.
                keep = mask.gather(1, ids.to(mask.device))
                ids = ids.masked_fill(~keep.cpu(), -1)
            return ids, scores

        modules = list(modules) + filter_modules
        return voy_forward, modules, is_cpu

    raise ValueError(f"unknown algorithm: {algo_name!r}")


class _SilvertorchSkippedOnNarrow(RuntimeError):
    """Sentinel raised when silvertorch is requested on a narrow-only sweep.

    The harness catches this and records the cell as skipped (rather than
    failed) in the per-row JSON output.
    """


SilvertorchSkippedOnNarrow = _SilvertorchSkippedOnNarrow
