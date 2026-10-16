from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal, NamedTuple

import torch
from torch import Tensor

from retrieve.functional import masked_topk
from retrieve.indexing.ivf import csr_layout, probe_width
from retrieve.indexing.kmeans import KMeans, KMeansInit
from retrieve.indexing.quantize import quantize_int8, quantize_int8_global
from retrieve.interfaces import (
    RetrievalModule,
    SilverTorchBackend,
    check_backend,
    load_prebuilt,
    ops_for,
)
from retrieve.ops import official as official_mod
from retrieve.ops.official import DEFAULT_CONFIG as OFFICIAL_DEFAULT, OfficialConfig

FilterMode = Literal["none", "bloom", "exact"]


class PreparedFilter(NamedTuple):
    """A batch's filter, prepared outside ``forward`` by ``SilverTorch.prepare_queries``.

    Exactly one form is set: ``query_attrs`` ``[B, C]`` (exact: int64 on ``official``, int32 on
    ``official-fork``, where it is also the bloom's recheck), Meta's CPU ``plans_data`` /
    ``plans_offsets`` (bloom on ``official``; one plan per distinct expression with
    ``plan_index`` ``[B]`` on the partial path), or the adapted code's device ``plan_blob`` with
    its ``plan_slots`` / ``plan_stack`` (bloom on ``official-fork``; kernels.md § official-fork)."""

    query_attrs: Tensor | None = None
    plans_data: Tensor | None = None
    plans_offsets: Tensor | None = None
    plan_index: Tensor | None = None
    plan_blob: Tensor | None = None
    plan_slots: int = 0
    plan_stack: int = 0


__all__ = ["FilterMode", "OfficialConfig", "SilverTorch", "SilverTorchBuilder"]


class SilverTorch(RetrievalModule):
    """IVF + INT8 ANN with an optional attribute filter (SilverTorch Algorithm 1): an ``[N, D]``
    int8 index with one global scale, per-row int8 queries, int8×int8 → int32 dots. Phase 1
    (k-means, quantization, probe selection) is ours; phases 2+3 run on Meta's ops.
    ``filter_mode`` picks the predicate (``"none"``, ``"bloom"``, ``"exact"``), ``backend`` the
    ops: ``"official-fork"`` (default; Meta's code adapted, capturable) or ``"official"`` (Meta's
    code as published, eager-only). Buffers, state-dict portability and ``build_timings``:
    architecture.md § SilverTorch; the two backends: kernels.md § official / § official-fork."""

    centroids: Tensor
    item_codes: Tensor
    global_scale: Tensor
    cluster_offsets: Tensor
    cluster_sizes: Tensor
    sort_perm: Tensor
    inv_perm: Tensor
    bloom_index: Tensor
    bundle_b_offsets: Tensor
    item_clause_attrs: Tensor
    clause_slots: Tensor
    clause_is_reverse: Tensor

    def __init__(
        self,
        k: int,
        n_lists: int,
        n_probe: int,
        filter_mode: FilterMode = "none",
        m_bits: int | None = None,
        k_hash: int | None = None,
        n_iter: int = 10,
        seed: int = 0,
        kmeans_init: KMeansInit = "random",
        backend: SilverTorchBackend = "official-fork",
        official: OfficialConfig | None = None,
    ) -> None:
        super().__init__()
        if filter_mode not in ("none", "bloom", "exact"):
            raise ValueError(
                f"filter_mode must be 'none', 'bloom', or 'exact', got {filter_mode!r}"
            )
        check_backend(backend, SilverTorchBackend)
        self.official: OfficialConfig = official if official is not None else OFFICIAL_DEFAULT
        if backend == "official-fork" and self.official.bloom_path == "full":
            raise ValueError(
                "OfficialConfig(bloom_path='full') does not apply to backend='official-fork': "
                "the adapted code evaluates the bloom over the probed columns only "
                "(kernels.md § official-fork); the full-N ablation is backend='official'"
            )

        if filter_mode == "bloom":
            configured = self.official.width_for(backend) == "configured"
            if k_hash is None or (m_bits is None and configured):
                raise ValueError(
                    "filter_mode='bloom' requires both m_bits and k_hash (m_bits is optional "
                    "where the official bloom width is upstream's, OfficialConfig.b_multiplier)"
                )
            _check_bloom_params(m_bits, k_hash)
            if k_hash > official_mod.MAX_SEARCH_K:
                raise ValueError(
                    f"k_hash must be <= {official_mod.MAX_SEARCH_K} on backend={backend!r} "
                    f"(MAX_K_V2 in the official search kernel, unchecked upstream), got {k_hash}"
                )
            self.m_bits = m_bits if m_bits is not None else 0
            self.k_hash = k_hash
        else:
            if m_bits is not None or k_hash is not None:
                raise ValueError(
                    f"m_bits/k_hash only apply to filter_mode='bloom', got "
                    f"filter_mode={filter_mode!r}"
                )
            self.m_bits = 0
            self.k_hash = 0

        official_mod.ensure_loaded(backend)
        if backend == "official":
            ops_for("triton")  # exact mode packs our clause_mask into Meta's bit mask
        self.filter_mode: FilterMode = filter_mode
        self.k = k
        self.n_lists = n_lists
        self.n_probe = n_probe
        self.n_iter = n_iter
        self.seed = seed
        self.kmeans_init: KMeansInit = kmeans_init
        self.backend = backend
        self.build_timings: dict[str, float] = {}
        self.register_load_state_dict_post_hook(_rederive_cached_scalars)

    @property
    def capturable(self) -> bool:
        return self.backend != "official"

    def register_index(
        self,
        item_embs: Tensor,
        item_clause_attrs: Tensor | None = None,
        clause_is_reverse: Tensor | None = None,
    ) -> None:
        self._validate_register_args(item_embs, item_clause_attrs, clause_is_reverse)
        lap = _lap(item_embs)
        t0 = lap()
        assignments = self._build_ivf(item_embs)
        t1 = lap()
        perm, inv_perm, cluster_offsets, cluster_sizes = csr_layout(assignments, self.n_lists)
        ivf = {
            "cluster_offsets": cluster_offsets,
            "cluster_sizes": cluster_sizes,
            "sort_perm": perm,
            "inv_perm": inv_perm,
        }
        # A Python int: the scorers take the width as a scalar op argument, and deriving it
        # inside forward would sync.
        self._probe_width = probe_width(cluster_sizes, self.n_probe)
        self._check_probe_pool(self.n_probe, self._probe_width, self.k)
        t2 = lap()
        self._quantize_items(item_embs, perm)
        t3 = lap()
        # The state-dict key order (architecture.md § SilverTorch): quant buffers, then the IVF.
        for name, buf in ivf.items():
            self.register_buffer(name, buf)
        self._register_filter_buffers(item_clause_attrs, clause_is_reverse, perm)
        t4 = lap()
        self.build_timings = {
            "kmeans_s": t1 - t0,
            "assemble_s": t2 - t1,
            "quantize_s": t3 - t2,
            "filter_s": t4 - t3,
        }

    def set_query_params(self, *, n_probe: int) -> None:
        """Change ``n_probe`` after ``register_index`` with the same two validations."""
        if n_probe > self.n_lists:
            raise ValueError(f"n_probe ({n_probe}) cannot exceed n_lists ({self.n_lists}).")
        width = probe_width(self.cluster_sizes, n_probe)
        self._check_probe_pool(n_probe, width, self.k)
        self.n_probe = n_probe
        self._probe_width = width

    def _validate_register_args(
        self,
        item_embs: Tensor,
        item_clause_attrs: Tensor | None,
        clause_is_reverse: Tensor | None,
    ) -> None:
        if self.filter_mode == "none" and item_clause_attrs is not None:
            raise ValueError(
                "item_clause_attrs requires filter_mode='bloom' or filter_mode='exact'"
            )
        if self.filter_mode == "none" and clause_is_reverse is not None:
            raise ValueError("clause_is_reverse requires filter_mode='exact'")
        if self.filter_mode == "exact" and item_clause_attrs is None:
            raise ValueError("filter_mode='exact' requires item_clause_attrs at register_index")
        if self.filter_mode == "bloom" and clause_is_reverse is not None:
            raise ValueError("clause_is_reverse is only used with filter_mode='exact'")

        n = item_embs.shape[0]
        if self.n_lists > n:
            raise ValueError(f"n_lists ({self.n_lists}) cannot exceed N ({n}).")
        if self.n_probe > self.n_lists:
            raise ValueError(f"n_probe ({self.n_probe}) cannot exceed n_lists ({self.n_lists}).")

    def _build_ivf(self, item_embs: Tensor) -> Tensor:
        """K-means; registers ``centroids`` and returns the ``[N]`` assignment."""
        centroids, assignments = KMeans(
            n_lists=self.n_lists, n_iter=self.n_iter, seed=self.seed, init=self.kmeans_init
        ).fit(item_embs)
        self.register_buffer("centroids", centroids)
        return assignments

    def _check_probe_pool(self, n_probe: int, width: int, k: int) -> None:
        if width < k:
            raise ValueError(
                f"k={k} exceeds the probe pool: the {n_probe} largest clusters hold {width}"
            )

    def _quantize_items(self, item_embs: Tensor, perm: Tensor) -> None:
        codes, global_scale = quantize_int8_global(item_embs, rows=perm)
        self.register_buffer("item_codes", codes)
        self.register_buffer(
            "global_scale",
            torch.tensor(global_scale, dtype=torch.float32, device=item_embs.device),
        )
        # A Python float: ``.item()`` per call would sync and break graph capture.
        self._global_scale_f = float(global_scale)

    def _register_filter_buffers(
        self,
        item_clause_attrs: Tensor | None,
        clause_is_reverse: Tensor | None,
        perm: Tensor,
    ) -> None:
        """Filter buffers in the cluster-sorted doc space (``perm`` is ``sort_perm``): the official
        ``bloom_index`` / ``bundle_b_offsets`` (Meta's hash; ``k_hash`` is the search ``k``,
        ``OfficialConfig.build_k`` the build ``k``), plus the exact attrs where the bloom's
        survivors are rechecked; or the exact attrs and ``clause_is_reverse``."""
        device = self.item_codes.device
        attrs = None if item_clause_attrs is None else item_clause_attrs.long()[perm].contiguous()
        if self.filter_mode == "bloom":
            cfg = self.official
            if attrs is None:
                # No attributes at build time: an empty index; a later attribute query raises.
                bloom_index = torch.empty(0, dtype=torch.int64, device=device)
                bundle_b_offsets = torch.zeros(1, dtype=torch.int64, device=device)
            else:
                configured = cfg.width_for(self.backend) == "configured"
                bloom_index, bundle_b_offsets = official_mod.build_bloom_index(
                    attrs,
                    b_multiplier=cfg.b_multiplier,
                    build_k=cfg.build_k if cfg.build_k is not None else self.k_hash,
                    fast_build=cfg.fast_build,
                    bits_per_doc=self.m_bits if configured else None,
                    backend=self.backend,
                )
            self.register_buffer("bloom_index", bloom_index)
            self.register_buffer("bundle_b_offsets", bundle_b_offsets)
            if attrs is not None and cfg.recheck_for(self.backend):
                self._register_exact_buffers(
                    attrs, torch.zeros(attrs.shape[1], dtype=torch.bool, device=device)
                )
        elif self.filter_mode == "exact":
            assert attrs is not None  # narrowed by _validate_register_args
            if clause_is_reverse is None:
                clause_is_reverse = torch.zeros(attrs.shape[1], dtype=torch.bool, device=device)
            self._register_exact_buffers(attrs, clause_is_reverse)

    def _register_exact_buffers(self, attrs: Tensor, clause_is_reverse: Tensor) -> None:
        """``item_clause_attrs`` (+ ``clause_slots`` on official-fork, whose exact kernels read
        the clause-major int32 layout of ``fork_clause_attrs``) and ``clause_is_reverse``."""
        if self.backend == "official-fork":
            attrs, clause_slots = official_mod.fork_clause_attrs(attrs)
            self.register_buffer("clause_slots", clause_slots)
        self.register_buffer("item_clause_attrs", attrs)
        self.register_buffer("clause_is_reverse", clause_is_reverse)

    def compile(self, *args, **kwargs):
        """``nn.Module.compile``, refused on the eager-only ``official`` backend."""
        if self.backend == "official":
            raise RuntimeError(
                "SilverTorch(backend='official') is eager-only: every official op syncs the "
                "host and re-uploads its plans per call, so there is no torch.compile / "
                "CUDA-graph path. Use backend='official-fork' for compiled runs."
            )
        return super().compile(*args, **kwargs)

    def prepare_queries(self, query_clause_attrs: Tensor) -> PreparedFilter:
        """The query side of the filter for one batch, done once outside ``forward``: Meta's
        parsed plans (bloom; CPU on ``official``, uploaded once on ``official-fork``) or the
        attrs (exact). Host work and syncs belong here; ``forward`` has none of ours."""
        if self.filter_mode == "none":
            raise ValueError("prepare_queries needs filter_mode='bloom' or filter_mode='exact'")
        if self.filter_mode == "exact":
            return PreparedFilter(query_attrs=self._exact_query_attrs(query_clause_attrs))
        if self.bloom_index.numel() == 0:
            raise RuntimeError(
                "this official bloom index was registered without item_clause_attrs, so there "
                "is nothing to search; register_index with attributes or forward unfiltered"
            )
        cfg = self.official
        expressions = official_mod.queries_to_expressions(query_clause_attrs)
        if self.backend == "official-fork":
            return self._prepare_device_plans(expressions, query_clause_attrs)
        if cfg.bloom_path == "full":  # Meta's full search takes one plan per row
            data, offsets = official_mod.parse_plans(
                expressions, cfg.n_stored_hashes, cfg.max_sub_queries, backend=self.backend
            )
            return PreparedFilter(plans_data=data, plans_offsets=offsets)
        # One plan per distinct expression, as a server batching shared filters would send.
        unique, index = _unique_plans(expressions, query_clause_attrs.device)
        data, offsets = official_mod.parse_plans(
            unique, cfg.n_stored_hashes, cfg.max_sub_queries, backend=self.backend
        )
        return PreparedFilter(plans_data=data, plans_offsets=offsets, plan_index=index)

    def _exact_query_attrs(self, query_clause_attrs: Tensor) -> Tensor:
        if self.backend == "official-fork":
            return official_mod.fork_query_attrs(query_clause_attrs)
        return query_clause_attrs.long().contiguous()

    def _prepare_device_plans(
        self, expressions: list[str], query_clause_attrs: Tensor
    ) -> PreparedFilter:
        """The adapted code's plans, uploaded once into a blob of one shape per batch shape: one
        plan per distinct expression with the row → plan index."""
        cfg = self.official
        rows, n_clauses = query_clause_attrs.shape
        unique, index = _unique_plans(expressions, query_clause_attrs.device)
        plans = official_mod.prepare_plans(
            unique,
            cfg.n_stored_hashes,
            cfg.max_sub_queries,
            n_clauses,
            query_clause_attrs.device,
            slots=rows,
            backend=self.backend,
        )
        recheck = cfg.recheck_for(self.backend)
        return PreparedFilter(
            query_attrs=self._exact_query_attrs(query_clause_attrs) if recheck else None,
            plan_index=index,
            plan_blob=plans.blob,
            plan_slots=plans.slots,
            plan_stack=plans.max_stack_size,
        )

    def forward(
        self,
        query: Tensor,
        prepared: PreparedFilter | None = None,
        candidate_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """IVF + INT8 retrieval with the filter from ``prepared`` (``prepare_queries``; ``None``
        skips it). ``candidate_ids [B, P]`` (``-1`` = padding) switches to a pure re-rank of the
        given original ids with no filter."""
        if candidate_ids is not None:
            if prepared is not None:
                raise ValueError(
                    "candidate_ids path scores the given candidates without the fused "
                    "attribute filter; pass prepared OR candidate_ids, not both"
                )
            return self._forward_candidates(query, candidate_ids)
        if self.filter_mode == "none" and prepared is not None:
            raise ValueError(
                "a prepared filter requires filter_mode='bloom' or filter_mode='exact'"
            )
        if self.backend == "official":
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    "SilverTorch(backend='official') is eager-only and cannot be traced by "
                    "torch.compile / torch.export: Meta's scorers sync the host (kernels.md § "
                    "official). Call the module eagerly, or use backend='official-fork'."
                )
            return self._forward_official(query, prepared)
        return self._forward_fork(query, prepared)

    def _phase1_probe_ids(self, query: Tensor) -> Tensor:
        """Centroid scores → top-``n_probe`` cluster ids ``[B, n_probe]``."""
        cent_scores = query @ self.centroids.t()
        _, probe_ids = torch.topk(cent_scores, self.n_probe, dim=1)
        return probe_ids

    def _forward_official(
        self,
        query: Tensor,
        prepared: PreparedFilter | None,
    ) -> tuple[Tensor, Tensor]:
        """Phases 2+3 on Meta's ops as published, eager only (kernels.md § official): ``none`` →
        ``fused_kmean_ann``; ``bloom`` → the prepared plans into the partial-response search over
        the probed clusters + ``fused_kmean_ann_with_partial_masks`` (``bloom_path="partial"``),
        or the full-``N`` packed mask into ``fused_kmean_ann(filtering_bit_mask=…)``
        (``"full"``); ``exact`` → our ``clause_mask_packed`` over the sorted attrs as the
        ``filtering_bit_mask``. ``max_tensor_size_per_row`` is the compact probe width."""
        probe_ids = self._phase1_probe_ids(query)
        cfg = self.official
        filtering_bit_mask: Tensor | None = None
        partial: tuple[Tensor, Tensor, Tensor] | None = None

        if self.filter_mode == "exact" and prepared is not None:
            filtering_bit_mask = ops_for("triton").clause_mask_packed(
                self.item_clause_attrs, self.clause_is_reverse, prepared.query_attrs
            )
        elif self.filter_mode == "bloom" and prepared is not None:
            plans = (prepared.plans_data, prepared.plans_offsets)
            if cfg.bloom_path == "partial":
                partial = official_mod.bloom_partial_masks(
                    self.bloom_index,
                    self.bundle_b_offsets,
                    plans,
                    self.cluster_offsets[probe_ids],
                    self.cluster_sizes[probe_ids],
                    self.k_hash,
                    cfg.n_stored_hashes,
                    prepared.plan_index,
                    backend=self.backend,
                )
            else:
                filtering_bit_mask = official_mod.bloom_filtering_mask(
                    self.bloom_index,
                    self.bundle_b_offsets,
                    plans,
                    self.k_hash,
                    cfg.n_stored_hashes,
                    backend=self.backend,
                )

        return official_mod.official_probe_score(
            query,
            probe_ids,
            self.cluster_offsets,
            self.cluster_sizes,
            self.item_codes,
            self.sort_perm,
            self.global_scale,
            self.k,
            self._probe_width,
            score_path=cfg.score_path,
            divisor=cfg.divisor,
            filtering_bit_mask=filtering_bit_mask,
            partial=partial,
            backend=self.backend,
        )

    def _forward_fork(
        self,
        query: Tensor,
        prepared: PreparedFilter | None,
    ) -> tuple[Tensor, Tensor]:
        """Phases 2+3 on the adapted code in one ``torch.ops.stfork.search`` call (kernels.md §
        official-fork): ``bloom`` → the device plans evaluated over the probed clusters, the
        survivors rechecked exactly unless ``OfficialConfig.recheck`` is off; ``exact`` → the
        clauses tested over the probed docs inside the op."""
        probe_ids = self._phase1_probe_ids(query)
        cfg = self.official
        exact = None
        bloom = None
        if self.filter_mode == "exact" and prepared is not None:
            exact = self._fork_exact(prepared.query_attrs)
        elif self.filter_mode == "bloom" and prepared is not None:
            plans = official_mod.DevicePlans(
                prepared.plan_blob, prepared.plan_slots, prepared.plan_stack
            )
            bloom = official_mod.ForkBloom(
                self.bloom_index,
                self.bundle_b_offsets,
                plans,
                prepared.plan_index,
                self.k_hash,
                cfg.n_stored_hashes,
            )
            if prepared.query_attrs is not None:
                exact = self._fork_exact(prepared.query_attrs)
        return official_mod.fork_search(
            query,
            probe_ids,
            self.cluster_offsets,
            self.cluster_sizes,
            self.item_codes,
            self.sort_perm,
            self.global_scale,
            self.k,
            self._probe_width,
            score_path=cfg.score_path,
            divisor=cfg.divisor,
            bloom=bloom,
            exact=exact,
        )

    def _fork_exact(self, query_attrs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (self.item_clause_attrs, self.clause_slots, self.clause_is_reverse, query_attrs)

    def _forward_candidates(
        self,
        query: Tensor,
        candidate_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Re-rank ``candidate_ids [B, P]`` (original ids; ``-1`` = padding, never gathered,
        scored or returned) with the int8 dot. Rows with fewer than ``min(k, P)`` real
        candidates carry ``-1`` / ``-inf`` in the tail; ``P < k`` keeps ``min(k, P)`` columns."""
        valid = candidate_ids >= 0
        safe = self.inv_perm[candidate_ids.clamp_min(0)]
        cand_codes = self.item_codes[safe].to(torch.float32)
        q_codes, q_scales = quantize_int8(query)
        scores = torch.bmm(
            q_codes.to(torch.float32).unsqueeze(1), cand_codes.transpose(1, 2)
        ).squeeze(1)
        scores = scores * q_scales.unsqueeze(1) * self.global_scale
        return masked_topk(scores, self.k, valid=valid, gather_ids=candidate_ids, pad_to_k=False)


def _check_bloom_params(m_bits: int | None, k_hash: int) -> None:
    """``m_bits=None``: the published backend sizes its own index."""
    if m_bits is not None:
        if m_bits <= 0 or (m_bits & (m_bits - 1)) != 0:
            raise ValueError(f"m_bits must be a positive power of 2, got {m_bits}")
        if m_bits % 64 != 0:
            raise ValueError(f"m_bits must be a multiple of 64, got {m_bits}")
    if k_hash <= 0:
        raise ValueError(f"k_hash must be positive, got {k_hash}")


def _unique_plans(expressions: list[str], device: torch.device) -> tuple[list[str], Tensor]:
    """The distinct expressions in first-seen order and the ``[B]`` int64 row → plan index."""
    unique = list(dict.fromkeys(expressions))
    row_plan = {e: i for i, e in enumerate(unique)}
    index = torch.tensor([row_plan[e] for e in expressions], dtype=torch.int64)
    return unique, index.to(device, non_blocking=True)


def _rederive_cached_scalars(module: SilverTorch, incompatible_keys) -> None:
    """``load_state_dict`` post-hook: the two Python scalars the forwards read instead of the
    buffers a load has just replaced."""
    module._global_scale_f = float(module.global_scale.item())
    module._probe_width = probe_width(module.cluster_sizes, module.n_probe)


def _lap(item_embs: Tensor) -> Callable[[], float]:
    """Phase clock for ``build_timings``: device-synchronised wall seconds."""

    device = item_embs.device

    def clock() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    return clock


class SilverTorchBuilder:
    """Fluent construction of a registered ``SilverTorch`` (Meta's ``*ModuleBuilder`` shape):
    the constructor keywords are ``SilverTorch``'s, then ``set_item_embeddings`` (+
    ``set_item_attributes`` for a filter mode) to build the index, or ``set_state_dict`` to
    load a prebuilt one without k-means. ``build()`` is construct → register (or load) →
    ``.to(device)``."""

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
        self._embs: Tensor | None = None
        self._attrs: Tensor | None = None
        self._reverse: Tensor | None = None
        self._state_dict: dict[str, Tensor] | None = None
        self._device: torch.device | str | None = None

    def set_item_embeddings(self, item_embs: Tensor) -> SilverTorchBuilder:
        self._embs = item_embs
        return self

    def set_item_attributes(
        self, item_clause_attrs: Tensor, clause_is_reverse: Tensor | None = None
    ) -> SilverTorchBuilder:
        self._attrs = item_clause_attrs
        self._reverse = clause_is_reverse
        return self

    def set_backend(
        self, backend: SilverTorchBackend, official: OfficialConfig | None = None
    ) -> SilverTorchBuilder:
        self._kwargs.update(backend=backend, official=official)
        return self

    def set_device(self, device: torch.device | str) -> SilverTorchBuilder:
        self._device = device
        return self

    def set_state_dict(self, state_dict: dict[str, Tensor]) -> SilverTorchBuilder:
        self._state_dict = state_dict
        return self

    def build(self) -> SilverTorch:
        if (self._embs is None) == (self._state_dict is None):
            raise ValueError("set exactly one of set_item_embeddings / set_state_dict")
        module = SilverTorch(**self._kwargs)
        if self._state_dict is not None:
            load_prebuilt(module, self._state_dict)
        else:
            module.register_index(self._embs, self._attrs, self._reverse)
        return module if self._device is None else module.to(self._device)
