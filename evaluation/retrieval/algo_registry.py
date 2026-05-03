"""Algorithm registry for the retrieval benchmark — unfiltered + filtered.

Each entry returns a ``(forward, modules, is_cpu)`` triple:

* ``forward(query) -> (ids, scores)`` for the unfiltered path
  (`build_algorithm`).
* ``forward(query, qa_narrow=..., qa_wide=...) -> (ids, scores)`` for the
  filtered path (`build_filtered_algorithm`); the optional kwargs are
  threaded into the underlying retrieval module's filter contract.

`modules` is the list of ``nn.Module``\\s holding device buffers so the
driver can ``del`` them between cells. `is_cpu` is True for the lone CPU
baseline (`voyager_hnsw`); the driver uses it to pick the right perf
primitive and to zero out CUDA-only memory columns.

Cascade rationale (LinR §4.3, Fig knn-v3): the quantized 1-bit V3 produces a
top-T pre-filter list; V2 then reranks at full precision. V1 is intentionally
absent as a stage-2 — V1's dense matmul does the same work as ``triton_knn``
alone, so a V3→V1 cascade is strictly slower than the unfiltered baseline
(measured ~1.22 ms vs 0.89 ms at 500M scale).

``silvertorch`` here is ``SilverTorch`` with ``m_bits``/``k_hash`` left unset
(bloom disabled) for the unfiltered Yambda path — bloom-fused only kicks in on
the filtered path when the filter sweep registers item attribute signatures.

``voyager_hnsw`` is the lone CPU baseline — Spotify's HNSW (multi-threaded by
default, what production deployments actually run). The PyPI ``faiss-cpu``
wheel was tried and dropped: its bundled libgomp does not parallelize
correctly in this environment (>10× slowdown at 32+ threads vs 1), so its
single-thread numbers were not meaningful as a "realistic CPU" baseline.

Filtered routing per algorithm:

* ``torch_fullscan`` — `FullScanKNN` accepts a ``mask`` kwarg and applies it
  as a **post-filter** (mask the top-K after the matmul). Recall can drop
  below the unfiltered baseline if the filter is very selective; the perf
  rows tag this as ``post_filter``. Goodreads driver skips this on
  filter sweeps (it equals the filtered-FullScan oracle by construction);
  ``linr_v2_filter_compact`` covers the same exact-filter cell with a more
  relevant perf profile.
* ``linr_v2_filter_compact`` — pure exact path. The filter primitive's
  native compact kernel (``clause_compact`` / ``bloom_compact`` / sparse
  ``combine_indices``) emits ``(candidate_ids, counts)`` of items that
  pass the predicate; V2 reranks those exactly via bmm. Recall vs the
  filtered-FullScan oracle is 1.0; this cell is the exact-LiNR speed /
  memory baseline for the filter suite.
* ``linr_v3_then_v2`` — V3 with the filter mask collapses to the
  positives via ``compact_mask`` inside the kernel; V3's top-N pool
  feeds V2 as ``candidate_ids``. Both V3 and V2 accept these kwargs as
  shipped. Approximate (V3 1-bit hash); recall measures hash quality.
* ``silvertorch`` — wide / combined sweeps use the bloom-fused
  `SilverTorch` (m_bits + k_hash supplied) and pass
  ``query_clause_attrs`` directly. Narrow-only sweeps skip silvertorch
  (bench thesis: narrow → ClauseIndex; wide → bloom-fused SilverTorch).
  Approximate (IVF probe + INT8 + bloom FPs).

Composition: narrow + wide masks AND'd via ``combine_masks``. For 2-shelf
wide queries the bloom is evaluated twice (once per shelf) and the two
masks AND'd, since the registered ``item_attrs_wide`` has ``C = 1`` and
bloom queries are ``[B, C]`` int64 single-attribute-per-clause.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from retrieval.config import FilterSweepCfg
from retrieval.voyager_baseline import VoyagerHNSW
from retrieve import (
    FullScanKNN,
    LiNR_V1_Triton,
    LiNR_V2_Triton,
    LiNR_V3_Triton,
    SilverTorch,
)
from retrieve.layers.filters import BloomFilter, ClauseIndex, combine_indices, combine_masks

ALGORITHMS = (
    "torch_fullscan",
    "triton_knn",
    "linr_v3_then_v2",
    "linr_v2_filter_compact",
    "silvertorch",
    "voyager_hnsw",
)

CPU_ALGOS = frozenset({"voyager_hnsw"})

ForwardFn = Callable[[Tensor], tuple[Tensor, Tensor]]
FilteredForwardFn = Callable[..., tuple[Tensor, Tensor]]
FilterKind = str  # "none" | "clause" | "bloom" | "combined"


# ----- unfiltered (yambda) ----------------------------------------------------


def build_algorithm(
    name: str,
    item_embs: Tensor,
    k: int,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[ForwardFn, list[nn.Module], bool]:
    """Build an unfiltered algorithm by name.

    Returns ``(forward, modules, is_cpu)``.
    """
    p = params or {}
    device = item_embs.device
    is_cpu = name in CPU_ALGOS

    if name == "torch_fullscan":
        idx = FullScanKNN(k=k).to(device)
        idx.register_index(item_embs)
        return (lambda q: idx(q)), [idx], is_cpu

    if name == "triton_knn":
        idx = LiNR_V1_Triton(k=k).to(device)
        idx.register_index(item_embs)
        return (lambda q: idx(q)), [idx], is_cpu

    if name == "linr_v3_then_v2":
        candidate_pool = int(p.get("candidate_pool", 5000))
        v3_seed = int(p.get("v3_seed", 0))
        v3 = LiNR_V3_Triton(k=candidate_pool, seed=v3_seed).to(device)
        v3.register_index(item_embs)
        stage2 = LiNR_V2_Triton(k=k).to(device)
        stage2.register_index(item_embs)

        # V2 takes candidate_ids directly — no mask round-trip, no scatter,
        # no [B, N+1] alloc, no compact_mask argsort.
        def forward(q: Tensor) -> tuple[Tensor, Tensor]:
            cand_ids, _ = v3(q)
            return stage2(q, candidate_ids=cand_ids)

        return forward, [v3, stage2], is_cpu

    if name == "silvertorch":
        n_lists = int(p.get("n_lists", 1024))
        n_probe = int(p.get("n_probe", 16))
        n_iter = int(p.get("n_iter", 10))
        seed = int(p.get("seed", 0))
        idx = SilverTorch(
            k=k,
            n_lists=n_lists,
            n_probe=n_probe,
            n_iter=n_iter,
            seed=seed,
        ).to(device)
        idx.register_index(item_embs)
        return (lambda q: idx(q)), [idx], is_cpu

    if name == "voyager_hnsw":
        m = int(p.get("m", 16))
        ef_construction = int(p.get("ef_construction", 200))
        ef_query = p.get("ef_query")
        ef_query = int(ef_query) if ef_query is not None else None
        num_threads = int(p.get("num_threads", -1))
        seed = int(p.get("seed", 0))
        idx = VoyagerHNSW(
            k=k,
            m=m,
            ef_construction=ef_construction,
            ef_query=ef_query,
            num_threads=num_threads,
            seed=seed,
        )
        idx.register_index(item_embs)
        return (lambda q: idx(q)), [idx], is_cpu

    raise ValueError(f"unknown algorithm: {name!r}")


# ----- filter modules ---------------------------------------------------------


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


def check_bloom_no_reverse(
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


# ----- filtered (goodreads / arxiv) -------------------------------------------


class SilvertorchSkippedOnNarrow(RuntimeError):
    """Sentinel raised when silvertorch is requested on a narrow-only sweep.

    The driver catches this and records the cell as skipped (rather than
    failed) in the per-row JSON output.
    """


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

    Skips silvertorch on narrow-only sweeps via `SilvertorchSkippedOnNarrow`.
    """
    check_bloom_no_reverse(filter_kind, sweep, clause_is_reverse)
    algo_params = algo_params or {}
    device = item_embs.device

    if filter_kind in ("clause", "combined") and ci is None:
        raise ValueError(f"{filter_kind} requires a pre-built ClauseIndex (ci=)")
    if filter_kind in ("bloom", "combined") and bf is None:
        raise ValueError(f"{filter_kind} requires a pre-built BloomFilter (bf=)")
    filter_modules: list[nn.Module] = [m for m in (ci, bf) if m is not None]

    def bloom_mask(qa_wide: Tensor) -> Tensor:
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

    def combined_mask(qa_narrow: Tensor | None, qa_wide: Tensor | None) -> Tensor | None:
        """Compose the optional narrow + wide masks via element-wise AND."""
        masks: list[Tensor | None] = []
        if ci is not None and qa_narrow is not None:
            masks.append(ci.evaluate_mask(qa_narrow))
        if bf is not None and qa_wide is not None:
            masks.append(bloom_mask(qa_wide))
        return combine_masks(*masks)

    is_cpu = False  # filtered path only handles GPU algos for now

    if algo_name == "torch_fullscan":
        idx = FullScanKNN(k=k).to(device)
        idx.register_index(item_embs)

        def fs_forward(
            q: Tensor, qa_narrow: Tensor | None = None, qa_wide: Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
            mask = combined_mask(qa_narrow, qa_wide)
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
            mask = combined_mask(qa_narrow, qa_wide)
            cand_ids, _ = v3(q, mask=mask)
            # V3 returns top-K=candidate_pool sorted DESC by Hamming score; if
            # the mask admits < candidate_pool items the trailing slots are -1.
            # V2's fused_masked_knn_topk does indirect loads via raw pointer
            # arithmetic, so item_embs_ptr + (-1)*stride walks off the buffer
            # (illegal memory access). Pass per-row counts so V2 only scores
            # the valid prefix.
            if mask is not None:
                counts = (cand_ids >= 0).sum(dim=1)
                return stage2(q, candidate_ids=cand_ids, counts=counts)
            return stage2(q, candidate_ids=cand_ids)

        modules = [v3, stage2, *filter_modules]
        return linr_forward, modules, is_cpu

    if algo_name == "linr_v2_filter_compact":
        # Exact filtered top-K via the filter primitive's native compact path.
        if filter_kind == "none":
            raise ValueError(
                "linr_v2_filter_compact requires a filter (clause/bloom/combined); "
                "use linr_v3_then_v2 for the no-filter quality suite."
            )
        v2 = LiNR_V2_Triton(k=k).to(device)
        v2.register_index(item_embs)

        def lv2_forward(
            q: Tensor, qa_narrow: Tensor | None = None, qa_wide: Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
            if filter_kind == "clause":
                assert ci is not None and qa_narrow is not None
                cand, counts = ci.evaluate_indices(qa_narrow)
            elif filter_kind == "bloom":
                assert bf is not None and qa_wide is not None
                qaw = qa_wide if qa_wide.dim() == 2 else qa_wide.unsqueeze(-1)
                if qaw.shape[1] == 1:
                    cand, counts = bf.evaluate_indices(qaw)
                else:
                    cand, counts = combine_indices(
                        [bf] * qaw.shape[1],
                        [qaw[:, j : j + 1] for j in range(qaw.shape[1])],
                    )
            else:  # combined
                assert ci is not None and bf is not None
                assert qa_narrow is not None and qa_wide is not None
                qaw = qa_wide if qa_wide.dim() == 2 else qa_wide.unsqueeze(-1)
                cand, counts = combine_indices(
                    [ci] + [bf] * qaw.shape[1],
                    [qa_narrow] + [qaw[:, j : j + 1] for j in range(qaw.shape[1])],
                )
            return v2(q, candidate_ids=cand, counts=counts)

        modules = [v2, *filter_modules]
        return lv2_forward, modules, is_cpu

    if algo_name == "silvertorch":
        if filter_kind == "clause":
            raise SilvertorchSkippedOnNarrow(
                f"silvertorch is skipped on filter_kind=clause (sweep {sweep.name!r}); "
                "the bench thesis is 'narrow → ClauseIndex; wide → bloom-fused SilverTorch'."
            )
        n_lists = int(algo_params.get("n_lists", 1024))
        n_probe = int(algo_params.get("n_probe", 24))
        n_iter = int(algo_params.get("n_iter", 10))
        seed = int(algo_params.get("seed", 0))

        if filter_kind in ("bloom", "combined"):
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
            # 2-shelf falls back to external mask combine — the fused bloom
            # path in SilverTorch only consumes a single per-clause value.
            if qa_wide is not None and qa_wide.dim() == 2 and qa_wide.shape[1] > 1:
                bloom_mask_external = bloom_mask(qa_wide)
                external = combine_masks(narrow_mask, bloom_mask_external)
                return idx(q, query_clause_attrs=None, mask=external)
            qaw = qa_wide if (qa_wide is not None and bf is not None) else None
            if qaw is not None and qaw.dim() == 1:
                qaw = qaw.unsqueeze(-1)
            return idx(q, query_clause_attrs=qaw, mask=narrow_mask)

        modules = [idx, *filter_modules]
        return st_forward, modules, is_cpu

    if algo_name == "triton_knn":
        idx = LiNR_V1_Triton(k=k).to(device)
        idx.register_index(item_embs)

        def kt_forward(
            q: Tensor, qa_narrow: Tensor | None = None, qa_wide: Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
            mask = combined_mask(qa_narrow, qa_wide)
            return idx(q, mask=mask)

        modules = [idx, *filter_modules]
        return kt_forward, modules, is_cpu

    if algo_name == "voyager_hnsw":
        # CPU baseline post-filter: build_algorithm's voyager forward, mask
        # the returned ids against the filter mask after-the-fact.
        forward_unfiltered, modules, is_cpu = build_algorithm(
            algo_name, item_embs, k=k, params=algo_params
        )

        def voy_forward(
            q: Tensor, qa_narrow: Tensor | None = None, qa_wide: Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
            ids, scores = forward_unfiltered(q)
            mask = combined_mask(
                qa_narrow.to(item_embs.device) if qa_narrow is not None else None,
                qa_wide.to(item_embs.device) if qa_wide is not None else None,
            )
            if mask is not None:
                keep = mask.gather(1, ids.to(mask.device))
                ids = ids.masked_fill(~keep.cpu(), -1)
            return ids, scores

        modules = list(modules) + filter_modules
        return voy_forward, modules, is_cpu

    raise ValueError(f"unknown algorithm: {algo_name!r}")


__all__ = [
    "ALGORITHMS",
    "FilterKind",
    "ForwardFn",
    "FilteredForwardFn",
    "SilvertorchSkippedOnNarrow",
    "build_algorithm",
    "build_filter_modules",
    "build_filtered_algorithm",
    "check_bloom_no_reverse",
    "synthesize_query_attrs_narrow",
]
