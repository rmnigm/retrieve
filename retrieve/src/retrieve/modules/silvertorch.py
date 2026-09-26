from __future__ import annotations

import time
from typing import Literal

import torch
from torch import Tensor

from retrieve.functional import masked_topk
from retrieve.indexing.bloom_hash import (
    build_query_bit_positions,
    build_signatures,
    build_transposed_sigs,
    generate_clause_salt,
    generate_seeds,
)
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

__all__ = ["FilterMode", "OfficialConfig", "SilverTorch", "SilverTorchBuilder"]


class SilverTorch(RetrievalModule):
    """Co-designed IVF + INT8 ANN + optional attribute filter (paper Algorithm 1, §4.2): an ``[N,
    D]`` int8 index with one global scale, per-row int8-quantized queries, and an int8×int8 →
    int32 dot dequantized once. ``filter_mode`` fuses a predicate into the probe+score kernel —
    ``"none"`` (plain ANN), ``"bloom"`` (subset test, needs ``m_bits``/``k_hash``), or
    ``"exact"`` (exact-clause, no false positives).

    ``backend="triton"`` (default) keeps all probe intermediates off HBM; ``backend="torch"``
    runs the same semantics eager but materializes ``[B, P, D]``, so large ``P·B·D`` needs the
    Triton backend. ``backend="official"``
    routes phases 2+3 to Meta's own ``torch.ops.st.*`` kernels (``meta-recsys/silvertorch``,
    the ``official`` extra) — the reference the Triton kernels are checked against: same
    k-means, same int8 codes and probes, the official scorer over the same cluster-sorted table
    (the CSR every backend registers: ``cluster_offsets`` / ``cluster_sizes`` / ``sort_perm`` /
    ``inv_perm``), the official bloom index + expression parser for
    ``filter_mode="bloom"`` (their hash, sized by ``OfficialConfig.b_multiplier``; ``m_bits`` is
    optional there) and our ``clause_mask`` packed into the scorer's bit mask for
    ``filter_mode="exact"``. It is **eager-only** (every official op syncs the host):
    ``torch.compile`` of an official module raises. ``triton`` and ``torch`` call the same
    op names in ``retrieve.ops.triton`` / ``retrieve.ops.reference`` (resolved through
    ``interfaces.ops_for``). See docs/system/kernels.md for every backend's design and
    constraints. ``triton`` and ``torch`` register the same buffers
    (a checkpoint is portable between them in every ``filter_mode``); an official state_dict
    is portable to them only without bloom (``bloom_index`` / ``bundle_b_offsets`` instead of
    ``bloom_transposed``).
    ``load_state_dict`` into a
    module of the same shape (one whose ``register_index`` already ran) re-derives the two
    Python-scalar caches the forwards read (``_global_scale_f``, ``_probe_width``) from
    the loaded buffers, so a loaded index scores like the one that was saved.

    After ``register_index``, ``build_timings`` holds the wall seconds of the four build
    phases (``kmeans_s``, ``assemble_s``, ``quantize_s``, ``filter_s``; device-synchronised)
    and ``set_query_params(n_probe=...)`` changes the probe width without a rebuild. ``k`` is a
    plain attribute, settable at any time. ``capturable`` says whether a forward can be
    CUDA-graph captured / compiled: every backend but ``official``."""

    centroids: Tensor
    item_codes: Tensor
    global_scale: Tensor
    cluster_offsets: Tensor  # [n_lists + 1] int64 CSR
    cluster_sizes: Tensor
    sort_perm: Tensor  # [N] sorted position -> original id
    inv_perm: Tensor  # [N] original id -> sorted position
    bloom_index: Tensor  # official + bloom: [W] int64
    bundle_b_offsets: Tensor  # official + bloom: [n_bundles + 1] int64
    bloom_transposed: Tensor  # triton / torch + bloom: [m_bits, ceil(N / 64)] int64
    hash_seeds: Tensor
    clause_salt: Tensor
    item_clause_attrs: Tensor
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
        backend: SilverTorchBackend = "triton",
        official: OfficialConfig | None = None,
    ) -> None:
        super().__init__()
        if filter_mode not in ("none", "bloom", "exact"):
            raise ValueError(
                f"filter_mode must be 'none', 'bloom', or 'exact', got {filter_mode!r}"
            )
        check_backend(backend, SilverTorchBackend)
        if official is not None and backend != "official":
            raise ValueError("official=OfficialConfig(...) only applies to backend='official'")
        # Import the backend's op namespace now (a Triton JIT registration, or the official
        # extension: OfficialMissing when the package cannot run here, ImportError when it is
        # present but broken) so failures surface at construction, not at the first forward.
        ops_for(backend)
        if backend == "official":
            official_mod.ensure_loaded()
            ops_for("triton")  # exact mode packs our clause_mask into the official bit mask
        self.official: OfficialConfig = official if official is not None else OFFICIAL_DEFAULT

        if filter_mode == "bloom":
            # On the official backend the bloom width comes from OfficialConfig.b_multiplier,
            # so m_bits is optional there; k_hash is the search-time `k` of the official
            # index (≤ MAX_K_V2, an unchecked upstream limit).
            if k_hash is None or (m_bits is None and backend != "official"):
                raise ValueError(
                    "filter_mode='bloom' requires both m_bits and k_hash (m_bits is "
                    "optional on backend='official', whose width is OfficialConfig.b_multiplier)"
                )
            if m_bits is not None:
                if m_bits <= 0 or (m_bits & (m_bits - 1)) != 0:
                    raise ValueError(f"m_bits must be a positive power of 2, got {m_bits}")
                if m_bits % 64 != 0:
                    raise ValueError(f"m_bits must be a multiple of 64, got {m_bits}")
            if k_hash <= 0:
                raise ValueError(f"k_hash must be positive, got {k_hash}")
            if backend == "official" and k_hash > official_mod.MAX_SEARCH_K:
                raise ValueError(
                    f"k_hash must be <= {official_mod.MAX_SEARCH_K} on backend='official' "
                    f"(MAX_K_V2 in the official search kernel, unchecked upstream), got {k_hash}"
                )
            self.m_bits = m_bits if m_bits is not None else 0
            self.k_hash = k_hash
            self.word_count = self.m_bits // 64
        else:
            if m_bits is not None or k_hash is not None:
                raise ValueError(
                    f"m_bits/k_hash only apply to filter_mode='bloom', got "
                    f"filter_mode={filter_mode!r}"
                )
            self.m_bits = 0
            self.k_hash = 0
            self.word_count = 0

        self.filter_mode: FilterMode = filter_mode
        self.k = k
        self.n_lists = n_lists
        self.n_probe = n_probe
        self.n_iter = n_iter
        self.seed = seed
        self.kmeans_init: KMeansInit = kmeans_init
        self.backend = backend
        self.build_timings: dict[str, float] = {}
        # Dispatch table built once; forward calls the bound method for this backend.
        self._forward_impl = {
            "triton": self._forward_ops,
            "torch": self._forward_ops,
            "official": self._forward_official,
        }[backend]
        # The two Python-scalar caches below (_global_scale_f, _probe_width) are set by
        # register_index; a state-dict load replaces the buffers they were derived from, so
        # they are re-derived after every load_state_dict.
        self.register_load_state_dict_post_hook(_rederive_cached_scalars)

    @property
    def capturable(self) -> bool:
        return self.backend != "official"

    @property
    def has_bloom(self) -> bool:
        return self.filter_mode == "bloom"

    @property
    def has_exact(self) -> bool:
        return self.filter_mode == "exact"

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
        # One layout on every backend (kernels.md § SilverTorch kernels): the int8 table in
        # cluster-sorted (CSR) order plus the permutation both ways. Frozen order: centroids,
        # item_codes, global_scale, cluster_offsets, cluster_sizes, sort_perm, inv_perm, then
        # the filter buffers.
        perm, inv_perm, cluster_offsets, cluster_sizes = csr_layout(assignments, self.n_lists)
        ivf = {
            "cluster_offsets": cluster_offsets,
            "cluster_sizes": cluster_sizes,
            "sort_perm": perm,
            "inv_perm": inv_perm,
        }
        # Plain Python int, cached: the scorers take the width as a scalar op argument, and
        # deriving it inside forward would sync (and hand dynamo a data-dependent value).
        self._probe_width = probe_width(cluster_sizes, self.n_probe)
        self._check_probe_pool(self.n_probe, self._probe_width, self.k)
        t2 = lap()
        self._quantize_items(item_embs, perm)
        t3 = lap()
        # Buffer registration order is frozen (state-dict key order): centroids, item_codes,
        # global_scale, the IVF buffers, then the filter buffers — hence the IVF buffers are
        # registered here, after the quant buffers.
        for name, buf in ivf.items():
            self.register_buffer(name, buf)
        self._register_filter_buffers(
            item_embs.shape[0], item_clause_attrs, clause_is_reverse, perm
        )
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
        """K-means clustering; registers ``centroids`` and returns the ``[N]`` assignment the
        layout (``indexing.csr_layout``) is derived from."""
        centroids, assignments = KMeans(
            n_lists=self.n_lists, n_iter=self.n_iter, seed=self.seed, init=self.kmeans_init
        ).fit(item_embs)
        self.register_buffer("centroids", centroids)
        return assignments

    def _check_probe_pool(self, n_probe: int, width: int, k: int) -> None:
        # The scorers' top-k runs over the compact probe width (the n_probe largest clusters)
        # with no pad tail, so it must hold >= k slots.
        if width < k:
            raise ValueError(
                f"k={k} exceeds the probe pool: the {n_probe} largest clusters hold {width}"
            )

    def _quantize_items(self, item_embs: Tensor, perm: Tensor) -> None:
        # Cluster-sorted, so a probed cluster is one contiguous run of code rows.
        codes, global_scale = quantize_int8_global(item_embs, rows=perm)
        self.register_buffer("item_codes", codes)
        # 0-d fp32 buffer: moves with .to(device) and parameterizes the kernel epilogue without a
        # per-index recompile.
        self.register_buffer(
            "global_scale",
            torch.tensor(global_scale, dtype=torch.float32, device=item_embs.device),
        )
        # Plain Python float: passing self.global_scale.item() per call forces a device→host sync
        # that breaks cudagraph capture. Cached once at index build.
        self._global_scale_f = float(global_scale)

    def _register_filter_buffers(
        self,
        n: int,
        item_clause_attrs: Tensor | None,
        clause_is_reverse: Tensor | None,
        perm: Tensor,
    ) -> None:
        """Filter buffers, in the cluster-sorted doc space (``perm`` is ``sort_perm``) so the
        scorers' positions address the same items. ``bloom`` → the transposed index
        ``bloom_transposed`` + ``hash_seeds`` + ``clause_salt`` on triton / torch, or the
        official ``bloom_index`` / ``bundle_b_offsets`` built by
        ``torch.ops.st.bloom_index_build`` (their hash; ``k_hash`` is the search ``k``,
        ``OfficialConfig.build_k`` the build ``k``); ``exact`` → the sorted narrow attrs +
        ``clause_is_reverse``, read by our kernels or, on official, by ``clause_mask``."""
        device = self.item_codes.device  # same device as item_embs
        attrs = None if item_clause_attrs is None else item_clause_attrs.long()[perm].contiguous()
        if self.filter_mode == "bloom" and self.backend == "official":
            cfg = self.official
            if attrs is None:
                # No attributes at build time: an empty index. A later query with
                # attributes has nothing to search and raises in _forward_official.
                bloom_index = torch.empty(0, dtype=torch.int64, device=device)
                bundle_b_offsets = torch.zeros(1, dtype=torch.int64, device=device)
            else:
                bloom_index, bundle_b_offsets = official_mod.build_bloom_index(
                    attrs,
                    b_multiplier=cfg.b_multiplier,
                    build_k=cfg.build_k if cfg.build_k is not None else self.k_hash,
                    fast_build=cfg.fast_build,
                )
            self.register_buffer("bloom_index", bloom_index)
            self.register_buffer("bundle_b_offsets", bundle_b_offsets)
        elif self.filter_mode == "bloom":
            seeds = generate_seeds(self.k_hash, device=device)
            if attrs is None:
                # No attributes at build time: all-zero signatures and an empty salt
                # (the clause count is unknown); a later query build derives its salt
                # device-side per call, see _query_bit_positions.
                sigs = torch.zeros(n, self.word_count, dtype=torch.int64, device=device)
                salt = torch.empty(0, dtype=torch.int64, device=device)
            else:
                salt = generate_clause_salt(attrs.shape[1], device=device)
                sigs = build_signatures(
                    attrs, seeds, self.m_bits, self.k_hash, self.word_count, clause_salt=salt
                )
            self.register_buffer("bloom_transposed", build_transposed_sigs(sigs))
            self.register_buffer("hash_seeds", seeds)
            # Registered (not rebuilt per call) so the bloom forward issues no
            # host→device copy — see bloom_hash.generate_clause_salt.
            self.register_buffer("clause_salt", salt)
        elif self.filter_mode == "exact":
            assert attrs is not None  # narrowed by _validate_register_args
            if clause_is_reverse is None:
                clause_is_reverse = torch.zeros(attrs.shape[1], dtype=torch.bool, device=device)
            self.register_buffer("item_clause_attrs", attrs)
            self.register_buffer("clause_is_reverse", clause_is_reverse)

    def compile(self, *args, **kwargs):
        """``nn.Module.compile`` — refused on the official backend (eager-only)."""
        if self.backend == "official":
            raise RuntimeError(
                "SilverTorch(backend='official') is eager-only: every official op syncs the "
                "host and re-uploads its plans per call, so there is no torch.compile / "
                "CUDA-graph path. Use backend='triton' or 'torch' for compiled runs."
            )
        return super().compile(*args, **kwargs)

    def forward(
        self,
        query: Tensor,
        query_clause_attrs: Tensor | None = None,
        candidate_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """IVF + (optional) attribute-filter-fused retrieval; ``query_clause_attrs`` is valid only
        for ``filter_mode="bloom"|"exact"`` and when ``None`` the filter branch is skipped (plain
        IVF + INT8 ANN). ``candidate_ids [B, P]`` (``-1`` = padding) switches to a pure
        re-rank of the given original ids with no filter."""
        if candidate_ids is not None:
            if query_clause_attrs is not None:
                raise ValueError(
                    "candidate_ids path scores the given candidates without the fused "
                    "attribute filter; pass query_clause_attrs OR candidate_ids, not both"
                )
            return self._forward_candidates(query, candidate_ids)
        if self.filter_mode == "none" and query_clause_attrs is not None:
            raise ValueError(
                "query_clause_attrs requires filter_mode='bloom' or filter_mode='exact'"
            )
        if self.backend == "official" and torch.compiler.is_compiling():
            raise RuntimeError(
                "SilverTorch(backend='official') is eager-only and cannot be traced by "
                "torch.compile / torch.export: every official op syncs the host. "
                "Call the module eagerly, or use backend='triton' / 'torch'."
            )
        return self._forward_impl(query, query_clause_attrs)

    def _phase1_probe_ids(self, query: Tensor) -> Tensor:
        """Phase 1 proper: centroid scores → top-``n_probe`` cluster ids ``[B, n_probe]``."""
        cent_scores = query @ self.centroids.t()
        _, probe_ids = torch.topk(cent_scores, self.n_probe, dim=1)
        return probe_ids

    def _query_bit_positions(self, query_clause_attrs: Tensor) -> Tensor:
        """``[B, C]`` query attrs → ``[B, C·k_hash]`` set-bit positions of the query signature
        (``-1`` = an inactive clause's slot), using the registered ``clause_salt`` buffer (no
        per-call host→device copy). The buffer is empty when the index was registered without
        attributes; then ``generate_clause_salt`` derives the ``[C]`` salt per call — a few tiny
        device-side kernels from ``arange`` and Python-int constants, still no host→device
        copy."""
        salt = self.clause_salt if self.clause_salt.numel() > 0 else None
        return build_query_bit_positions(
            query_clause_attrs.long(), self.hash_seeds, self.m_bits, self.k_hash, clause_salt=salt
        )

    def _forward_ops(
        self,
        query: Tensor,
        query_clause_attrs: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Phases 2+3 on the backend's op namespace: ``retrieve.ops.triton`` (one fused launch)
        or ``retrieve.ops.reference`` (the same semantics eager, materializing ``[B, width,
        D]``)."""
        ops = ops_for(self.backend)
        layout = (self._phase1_probe_ids(query), self.cluster_offsets, self.item_codes)
        tail = (self._global_scale_f, self.k, self._probe_width)

        if self.has_exact and query_clause_attrs is not None:
            return ops.codesigned_probe_score_exact(
                query,
                *layout,
                self.sort_perm,
                self.item_clause_attrs,
                self.clause_is_reverse,
                query_clause_attrs.long(),
                *tail,
            )
        if self.has_bloom and query_clause_attrs is not None:
            return ops.codesigned_probe_score_bloom(
                query,
                *layout,
                self.sort_perm,
                self._query_bit_positions(query_clause_attrs),
                self.bloom_transposed,
                *tail,
            )
        return ops.codesigned_probe_score(query, *layout, self.sort_perm, *tail)

    def _forward_official(
        self,
        query: Tensor,
        query_clause_attrs: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Algorithm 1 phases 2+3 on Meta's official ops, eager only.

        ``none`` → ``fused_kmean_ann``; ``bloom`` → the official expression parser (plans
        on CPU, memoised per expression tuple unless ``OfficialConfig.cache_plans=False``,
        the setting a timing run needs) + ``bloom_index_search_batch_return_partial_response``
        over the probed clusters + ``fused_kmean_ann_with_partial_masks``
        (``OfficialConfig.bloom_path="partial"``, the paper's co-design) or the full-``N``
        packed mask into
        ``fused_kmean_ann(filtering_bit_mask=…)`` (``"full"``, the S9 ablation); ``exact`` →
        our Triton ``clause_mask`` over the sorted attrs, packed into the same
        ``filtering_bit_mask`` (phase 2 ours, full ``N`` — labelled so in every table).
        ``max_tensor_size_per_row`` is the compact probe width the Triton scorer writes, so the
        official output has the same ``[B, width]`` (rounded to 32) and the ``masked_topk``
        epilogue costs the same in every arm."""
        probe_ids = self._phase1_probe_ids(query)
        cfg = self.official
        filtering_bit_mask: Tensor | None = None
        partial: tuple[Tensor, Tensor, Tensor] | None = None

        if self.has_exact and query_clause_attrs is not None:
            mask = ops_for("triton").clause_mask(
                self.item_clause_attrs, self.clause_is_reverse, query_clause_attrs.long()
            )  # [B, N] bool over cluster-sorted ids
            filtering_bit_mask = official_mod.pack_mask(mask, official_mod.MASK_BIT_ORDER)
        elif self.has_bloom and query_clause_attrs is not None:
            if self.bloom_index.numel() == 0:
                raise RuntimeError(
                    "this official bloom index was registered without item_clause_attrs, "
                    "so there is nothing to search; register_index with attributes or call "
                    "forward without query_clause_attrs"
                )
            expressions = official_mod.queries_to_expressions(query_clause_attrs)
            plans = official_mod.parse_plans(
                expressions, cfg.n_stored_hashes, cfg.max_sub_queries, cache=cfg.cache_plans
            )
            if cfg.bloom_path == "partial":
                partial = official_mod.bloom_partial_masks(
                    self.bloom_index,
                    self.bundle_b_offsets,
                    plans,
                    self.cluster_offsets[probe_ids],
                    self.cluster_sizes[probe_ids],
                    self.k_hash,
                    cfg.n_stored_hashes,
                )
            else:
                filtering_bit_mask = official_mod.bloom_filtering_mask(
                    self.bloom_index, self.bundle_b_offsets, plans, self.k_hash, cfg.n_stored_hashes
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
        )

    def _forward_candidates(
        self,
        query: Tensor,
        candidate_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Re-rank ``candidate_ids [B, P]`` (original item ids; ``-1`` = padding, as every
        compact producer in the library emits) with the int8 dot: pads are never gathered
        (``clamp_min(0)``), never scored (``-inf``) and never returned (``-1`` sentinel).
        Rows with fewer than ``min(k, P)`` real candidates carry ``-1`` / ``-inf`` in the
        tail; ``pad_to_k=False`` keeps ``min(k, P)`` columns when ``P < k``. Pure tensor
        flow, no host sync."""
        valid = candidate_ids >= 0
        # item_codes is cluster-sorted; candidate ids are original ids.
        safe = self.inv_perm[candidate_ids.clamp_min(0)]
        cand_codes = self.item_codes[safe].to(torch.float32)
        q_codes, q_scales = quantize_int8(query)
        scores = torch.bmm(
            q_codes.to(torch.float32).unsqueeze(1), cand_codes.transpose(1, 2)
        ).squeeze(1)
        scores = scores * q_scales.unsqueeze(1) * self.global_scale
        return masked_topk(scores, self.k, valid=valid, gather_ids=candidate_ids, pad_to_k=False)


def _rederive_cached_scalars(module: SilverTorch, incompatible_keys) -> None:
    """``load_state_dict`` post-hook: re-derive ``_global_scale_f`` and ``_probe_width`` from
    the loaded buffers. Both are plain Python scalars cached at ``register_index`` so the
    forwards issue no per-call ``.item()`` sync (cudagraph capture) and no shape read (SymInt
    under ``torch.compile(dynamic=True)``); a load replaces the buffers underneath them. The
    ``.item()`` syncs here run once, at load time. Buffers absent from the module (a load
    before ``register_index``, or a ``strict=False`` partial load) leave the corresponding
    cache untouched."""
    if hasattr(module, "global_scale"):
        module._global_scale_f = float(module.global_scale.item())
    if hasattr(module, "cluster_sizes"):
        module._probe_width = probe_width(module.cluster_sizes, module.n_probe)


def _lap(item_embs: Tensor):
    """Phase clock for ``build_timings``: device-synchronised wall seconds."""
    if not item_embs.is_cuda:
        return time.perf_counter

    def lap() -> float:
        torch.cuda.synchronize(item_embs.device)
        return time.perf_counter()

    return lap


class SilverTorchBuilder:
    """Fluent construction of a registered ``SilverTorch`` (Meta's ``*ModuleBuilder`` shape):
    the constructor keywords are ``SilverTorch``'s, then ``set_item_embeddings`` (+
    ``set_item_attributes`` for a filter mode) to build the index, or ``set_state_dict`` to
    load a prebuilt one without k-means — the load hook re-derives the cached scalars.
    ``build()`` is construct → register (or load) → ``.to(device)``."""

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
