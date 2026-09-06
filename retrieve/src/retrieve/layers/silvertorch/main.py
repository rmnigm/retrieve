from __future__ import annotations

import warnings
from typing import Literal

import torch
from torch import Tensor

from retrieve.interfaces import Backend, RetrievalModule
from retrieve.kernels.filters.clause_mask import clause_mask
from retrieve.kernels.silvertorch import official as official_mod
from retrieve.kernels.silvertorch.codesigned_probe_score import (
    codesigned_probe_score,
    codesigned_probe_score_bloom,
)
from retrieve.kernels.silvertorch.codesigned_probe_score_cuda import (
    build_transposed_sigs,
    codesigned_probe_score_bloom_cuda,
    codesigned_probe_score_cuda,
    codesigned_probe_score_exact_cuda,
)
from retrieve.kernels.silvertorch.codesigned_probe_score_cute import (
    codesigned_probe_score_bloom_cute,
    codesigned_probe_score_cute,
    codesigned_probe_score_exact_cute,
)
from retrieve.kernels.silvertorch.codesigned_probe_score_exact import (
    codesigned_probe_score_exact,
)
from retrieve.kernels.silvertorch.official import DEFAULT_CONFIG as OFFICIAL_DEFAULT
from retrieve.kernels.silvertorch.official import OfficialConfig
from retrieve.layers.filters.bloom_hash import (
    bloom_subset_match,
    build_query_signatures,
    build_signatures,
    generate_clause_salt,
    generate_seeds,
)
from retrieve.layers.filters.exact_attribute import clause_subset_match
from retrieve.layers.utils.kmeans import KMeansTorch
from retrieve.layers.utils.quantize import quantize_int8, quantize_int8_global
from retrieve.layers.utils.topk import masked_topk

FilterMode = Literal["none", "bloom", "exact"]

# Backends that run the paper's two-kernel design (phase-2 partial masks over the probed
# clusters -> masked dp4a scoring) and therefore read the *transposed* bloom index. The
# CuTe DSL backend is a port of the CUDA C++ one: same ops, same buffers, same layout.
_TWO_KERNEL_BACKENDS = ("cuda", "cute")


class SilverTorch(RetrievalModule):
    """Co-designed IVF + INT8 ANN + optional attribute filter (paper Algorithm 1, §4.2): an ``[N,
    D]`` int8 index with one global scale, per-row int8-quantized queries, and an int8×int8 →
    int32 dot dequantized once. ``filter_mode`` fuses a predicate into the probe+score kernel —
    ``"none"`` (plain ANN), ``"bloom"`` (subset test, needs ``m_bits``/``k_hash``), or
    ``"exact"`` (exact-clause, no false positives).

    ``backend="triton"`` (default) keeps all probe intermediates off HBM; ``backend="torch"``
    runs the same semantics eager but materializes ``[B, P, D]``, so large ``P·B·D`` needs the
    Triton backend. ``backend="cuda"`` runs the CUDA C++ implementation of all three filter
    modes, JIT-compiling on first forward and returning bit-identical results to the Triton
    backend; ``backend="cute"`` is the CuTe DSL port of that backend (same ops, same buffers,
    same mask layout, bit-identical to cuda; needs the ``cute`` extra). ``backend="official"``
    routes phases 2+3 to Meta's own ``torch.ops.st.*`` kernels (``meta-recsys/silvertorch``,
    the ``official`` extra) — the reference the Triton kernels are checked against: same
    k-means, same int8 codes and probes, the official scorer over a cluster-sorted table
    (buffers ``cluster_offsets`` / ``cluster_sizes`` / ``sort_perm`` / ``inv_perm`` instead of
    ``padded_cluster_items``), the official bloom index + expression parser for
    ``filter_mode="bloom"`` (their hash, sized by ``OfficialConfig.b_multiplier``; ``m_bits`` is
    optional there) and our ``clause_mask`` packed into the scorer's bit mask for
    ``filter_mode="exact"``. It is **eager-only** (every official op syncs the host):
    ``torch.compile`` of an official module raises. See docs/system/kernels.md for every
    backend's design and constraints. Only the ``"bloom"`` index differs across backends
    (cuda and cute register the transposed ``bloom_sigs_t`` instead of ``bloom_sigs``;
    official registers ``bloom_index`` / ``bundle_b_offsets``) — a cuda or cute ``"exact"``
    module's state_dict is identical to a triton one's, and a cute checkpoint is
    byte-identical to a cuda one; an official state_dict is portable to no other backend
    (cluster-sorted ``item_codes``, no ``padded_cluster_items``)."""

    centroids: Tensor
    item_codes: Tensor
    global_scale: Tensor
    padded_cluster_items: Tensor
    cluster_sizes: Tensor
    cluster_offsets: Tensor  # official only: [n_lists + 1] int64 CSR
    sort_perm: Tensor  # official only: [N] sorted position -> original id
    inv_perm: Tensor  # official only: [N] original id -> sorted position
    bloom_index: Tensor  # official + bloom: [W] int64
    bundle_b_offsets: Tensor  # official + bloom: [n_bundles + 1] int64
    bloom_sigs: Tensor
    bloom_sigs_t: Tensor
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
        backend: Backend = "triton",
        official: OfficialConfig | None = None,
    ) -> None:
        super().__init__()
        if filter_mode not in ("none", "bloom", "exact"):
            raise ValueError(
                f"filter_mode must be 'none', 'bloom', or 'exact', got {filter_mode!r}"
            )
        if backend not in ("triton", "torch", "cuda", "cute", "official"):
            raise ValueError(f"unknown backend {backend!r}")
        if official is not None and backend != "official":
            raise ValueError("official=OfficialConfig(...) only applies to backend='official'")
        if backend == "official":
            # Fail at construction, not at the first forward: OfficialMissing when the
            # package cannot run here, ImportError when it is present but broken.
            official_mod.ensure_loaded()
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
        self.backend = backend

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
        padded, cluster_sizes, sort_perm, cluster_offsets = self._build_ivf(item_embs)
        if self.backend == "official":
            # Official layout (plan §4.1 / D4): the int8 table in cluster-sorted (CSR) order
            # plus the permutation both ways; no padded_cluster_items. Frozen order:
            # centroids, item_codes, global_scale, cluster_offsets, cluster_sizes,
            # sort_perm, inv_perm, then the filter buffers.
            self._quantize_items(item_embs, perm=sort_perm)
            inv_perm = torch.empty_like(sort_perm)
            inv_perm[sort_perm] = torch.arange(sort_perm.numel(), device=sort_perm.device)
            self.register_buffer("cluster_offsets", cluster_offsets)
            self.register_buffer("cluster_sizes", cluster_sizes)
            self.register_buffer("sort_perm", sort_perm)
            self.register_buffer("inv_perm", inv_perm)
            self._register_official_filter_buffers(item_clause_attrs, clause_is_reverse)
            return
        self._quantize_items(item_embs)
        # Buffer registration order is frozen (state-dict key order): centroids, item_codes,
        # global_scale, padded_cluster_items, cluster_sizes, then the filter buffers — hence the
        # two IVF buffers computed by _build_ivf are registered here, after the quant buffers.
        self.register_buffer("padded_cluster_items", padded)
        self.register_buffer("cluster_sizes", cluster_sizes)
        self._register_filter_buffers(item_embs.shape[0], item_clause_attrs, clause_is_reverse)

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

    def _build_ivf(self, item_embs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """K-means clustering; registers ``centroids`` and returns ``(padded_cluster_items,
        cluster_sizes, sort_perm, cluster_offsets)`` for registration after the quantization
        buffers (frozen order). ``sort_perm`` / ``cluster_offsets`` are the cluster-sorted
        CSR view of the same assignment — the official backend's layout."""
        n = item_embs.shape[0]
        centroids, assignments = KMeansTorch(
            n_lists=self.n_lists, n_iter=self.n_iter, seed=self.seed
        ).fit(item_embs)
        cluster_sizes = torch.bincount(assignments, minlength=self.n_lists)
        max_size = int(cluster_sizes.max().item())
        # Plain Python int, cached for the same reason as _global_scale_f below: the
        # cuda exact op takes the padded cluster width as a scalar argument, and
        # reading it back off padded_cluster_items.shape[1] inside forward would hand
        # dynamo a SymInt under `torch.compile(dynamic=True)` — which the custom op's
        # `int` schema slot would then have to specialize or graph-break on.
        self._max_cluster_size = max_size

        # P (probe pool width) = n_probe × max_cluster_size; topk runs with no pad tail, so the
        # index must supply >= k candidate slots per query.
        if self.n_probe * max_size < self.k:
            raise ValueError(
                f"k={self.k} exceeds probe pool n_probe * max_cluster_size = "
                f"{self.n_probe} * {max_size} = {self.n_probe * max_size}"
            )

        sort_idx = torch.argsort(assignments)
        sorted_clusters = assignments[sort_idx]
        offsets = torch.zeros(self.n_lists + 1, dtype=torch.long, device=item_embs.device)
        offsets[1:] = cluster_sizes.cumsum(0)

        padded = torch.full(
            (self.n_lists, max_size),
            -1,
            dtype=torch.long,
            device=item_embs.device,
        )
        within_slot = torch.arange(n, device=item_embs.device) - offsets[sorted_clusters]
        padded[sorted_clusters, within_slot] = sort_idx

        self.register_buffer("centroids", centroids)
        return padded, cluster_sizes, sort_idx, offsets

    def _quantize_items(self, item_embs: Tensor, perm: Tensor | None = None) -> None:
        codes, global_scale = quantize_int8_global(item_embs)
        if perm is not None:
            # Same codes, same global scale — permuted into cluster-sorted order for the
            # official CSR scorer (plan D3: both arms score identical int8 codes).
            codes = codes[perm].contiguous()
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
    ) -> None:
        device = self.item_codes.device  # same device as item_embs
        if self.filter_mode == "bloom":
            seeds = generate_seeds(self.k_hash, device=device)
            if item_clause_attrs is None:
                # No attributes at build time: all-zero signatures and an empty salt
                # (the clause count is unknown); a later query build derives its salt
                # on the fly, see _query_salt.
                sigs = torch.zeros(n, self.word_count, dtype=torch.int64, device=device)
                salt = torch.empty(0, dtype=torch.int64, device=device)
            else:
                salt = generate_clause_salt(item_clause_attrs.shape[1], device=device)
                sigs = build_signatures(
                    item_clause_attrs.long(),
                    seeds,
                    self.m_bits,
                    self.k_hash,
                    self.word_count,
                    clause_salt=salt,
                )
            if self.backend in _TWO_KERNEL_BACKENDS:
                # The cuda/cute path reads only the transposed index; row-wise sigs are
                # a build-time intermediate. Registering both would double the filter
                # index memory. Note this changes the state-dict key set — see
                # docs/system/architecture.md on cross-backend checkpoint portability.
                #
                # Size: bloom_sigs_t is [m_bits, n_lists * wpc] int64, i.e.
                #   W * 8 * n_lists * wpc * 64 bytes  =  bloom_sigs * (n_lists * wpc * 64 / N)
                # The factor is the IVF *padding* slack: n_lists * max_size / N rounded
                # up to whole 64-slot words. It is ~1.0-1.1x for a balanced index but
                # grows without bound as clusters get more uneven, because every cluster
                # is padded out to the largest one.
                slack = self.padded_cluster_items.numel() / max(n, 1)
                if slack > 2.0:
                    warnings.warn(
                        f"{self.backend}+bloom transposed index is {slack:.1f}x the row-wise "
                        f"bloom_sigs size: n_lists * max_cluster_size = "
                        f"{self.padded_cluster_items.numel()} padded slots for N={n} "
                        f"items. Cluster sizes are very uneven — re-cluster (more "
                        f"n_iter, a different n_lists/seed) to bring this down.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                self.register_buffer(
                    "bloom_sigs_t", build_transposed_sigs(sigs, self.padded_cluster_items)
                )
            else:
                self.register_buffer("bloom_sigs", sigs)
            self.register_buffer("hash_seeds", seeds)
            # Registered (not rebuilt per call) so the bloom forward issues no
            # host→device copy — see bloom_hash.generate_clause_salt.
            self.register_buffer("clause_salt", salt)
        elif self.filter_mode == "exact":
            assert item_clause_attrs is not None  # narrowed by _validate_register_args
            c = item_clause_attrs.shape[1]
            if clause_is_reverse is None:
                clause_is_reverse = torch.zeros(
                    c, dtype=torch.bool, device=item_clause_attrs.device
                )
            self.register_buffer("item_clause_attrs", item_clause_attrs.long())
            self.register_buffer("clause_is_reverse", clause_is_reverse)

    def _register_official_filter_buffers(
        self,
        item_clause_attrs: Tensor | None,
        clause_is_reverse: Tensor | None,
    ) -> None:
        """Filter buffers of the official backend, in the cluster-sorted doc space so the
        scorer's ``cluster_offsets`` address the same items: ``bloom`` → the official
        ``bloom_index`` / ``bundle_b_offsets`` built by ``torch.ops.st.bloom_index_build``
        (their hash; ``k_hash`` is the search ``k``, ``OfficialConfig.build_k`` the build
        ``k``); ``exact`` → the narrow attrs (sorted) + ``clause_is_reverse`` read by our
        ``clause_mask``."""
        device = self.item_codes.device
        cfg = self.official
        if self.filter_mode == "bloom":
            if item_clause_attrs is None:
                # No attributes at build time: an empty index. A later query with
                # attributes has nothing to search and raises in _forward_official.
                bloom_index = torch.empty(0, dtype=torch.int64, device=device)
                bundle_b_offsets = torch.zeros(1, dtype=torch.int64, device=device)
            else:
                attrs_sorted = item_clause_attrs.long()[self.sort_perm]
                bloom_index, bundle_b_offsets = official_mod.build_bloom_index(
                    attrs_sorted,
                    b_multiplier=cfg.b_multiplier,
                    build_k=cfg.build_k if cfg.build_k is not None else self.k_hash,
                    fast_build=cfg.fast_build,
                )
            self.register_buffer("bloom_index", bloom_index)
            self.register_buffer("bundle_b_offsets", bundle_b_offsets)
        elif self.filter_mode == "exact":
            assert item_clause_attrs is not None  # narrowed by _validate_register_args
            c = item_clause_attrs.shape[1]
            if clause_is_reverse is None:
                clause_is_reverse = torch.zeros(c, dtype=torch.bool, device=device)
            self.register_buffer(
                "item_clause_attrs", item_clause_attrs.long()[self.sort_perm].contiguous()
            )
            self.register_buffer("clause_is_reverse", clause_is_reverse)

    def compile(self, *args, **kwargs):
        """``nn.Module.compile`` — refused on the official backend (eager-only, plan D7)."""
        if self.backend == "official":
            raise RuntimeError(
                "SilverTorch(backend='official') is eager-only: every official op syncs the "
                "host and re-uploads its plans per call, so there is no torch.compile / "
                "CUDA-graph path (plan D7). Use backend='triton' or 'torch' for compiled runs."
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
        IVF + INT8 ANN)."""
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
        if self.backend == "triton":
            return self._forward_triton(query, query_clause_attrs)
        if self.backend == "cuda":
            return self._forward_cuda(query, query_clause_attrs)
        if self.backend == "cute":
            return self._forward_cute(query, query_clause_attrs)
        if self.backend == "official":
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    "SilverTorch(backend='official') is eager-only and cannot be traced by "
                    "torch.compile / torch.export (plan D7): every official op syncs the host. "
                    "Call the module eagerly, or use backend='triton' / 'torch'."
                )
            return self._forward_official(query, query_clause_attrs)
        return self._forward_torch_eager(query, query_clause_attrs)

    def _phase1_probe_ids(self, query: Tensor) -> Tensor:
        """Phase 1 proper: centroid scores → top-``n_probe`` cluster ids ``[B, n_probe]``."""
        cent_scores = query @ self.centroids.t()
        _, probe_ids = torch.topk(cent_scores, self.n_probe, dim=1)
        return probe_ids

    def _phase1_probe_with_ids(self, query: Tensor) -> tuple[Tensor, Tensor]:
        """Phase 1: centroid top-``n_probe`` then gather padded probed items; returns
        ``(probe_ids [B, n_probe], flat_items [B, P])`` (P = n_probe × max_cluster_size)
        with ``-1`` marking empty-cluster padding in ``flat_items``."""
        b = query.shape[0]
        probe_ids = self._phase1_probe_ids(query)
        probed = self.padded_cluster_items[probe_ids]
        return probe_ids, probed.reshape(b, -1)

    def _phase1_probe(self, query: Tensor) -> Tensor:
        return self._phase1_probe_with_ids(query)[1]

    def _query_bits(self, query_clause_attrs: Tensor) -> Tensor:
        """``[B, C]`` query attrs → ``[B, W]`` bloom query signature, using the registered
        ``clause_salt`` buffer (no per-call host→device copy). The buffer is empty when the
        index was registered without attributes; then the salt is derived on the fly."""
        salt = self.clause_salt if self.clause_salt.numel() > 0 else None
        return build_query_signatures(
            query_clause_attrs.long().unsqueeze(-1),
            self.hash_seeds,
            self.m_bits,
            self.k_hash,
            self.word_count,
            clause_salt=salt,
        )

    def _forward_triton(
        self,
        query: Tensor,
        query_clause_attrs: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        flat_items = self._phase1_probe(query)

        if self.has_exact and query_clause_attrs is not None:
            return codesigned_probe_score_exact(
                query,
                flat_items,
                self.item_codes,
                self.item_clause_attrs,
                self.clause_is_reverse,
                query_clause_attrs.long(),
                self._global_scale_f,
                self.k,
            )

        if self.has_bloom and query_clause_attrs is not None:
            qb = self._query_bits(query_clause_attrs)
            return codesigned_probe_score_bloom(
                query,
                flat_items,
                self.item_codes,
                qb,
                self.bloom_sigs,
                self._global_scale_f,
                self.k,
            )
        return codesigned_probe_score(
            query,
            flat_items,
            self.item_codes,
            self._global_scale_f,
            self.k,
        )

    def _forward_cuda(
        self,
        query: Tensor,
        query_clause_attrs: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """CUDA C++ backend — see ``_forward_two_kernel``."""
        return self._forward_two_kernel(
            query,
            query_clause_attrs,
            plain=codesigned_probe_score_cuda,
            bloom=codesigned_probe_score_bloom_cuda,
            exact=codesigned_probe_score_exact_cuda,
        )

    def _forward_cute(
        self,
        query: Tensor,
        query_clause_attrs: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """CuTe DSL backend — the cuda forward with the cute ops; see ``_forward_two_kernel``."""
        return self._forward_two_kernel(
            query,
            query_clause_attrs,
            plain=codesigned_probe_score_cute,
            bloom=codesigned_probe_score_bloom_cute,
            exact=codesigned_probe_score_exact_cute,
        )

    def _forward_two_kernel(
        self,
        query: Tensor,
        query_clause_attrs: Tensor | None,
        *,
        plain,
        bloom,
        exact,
    ) -> tuple[Tensor, Tensor]:
        """Paper Algorithm 1 as the cuda and cute backends run it: phase 1 host-side,
        then a two-kernel filtered path (partial 1-bit masks over the probed clusters →
        masked dp4a scoring) or plain mask-free scoring. Bloom and exact share the
        scoring kernel and differ only in which phase-2 kernel builds the mask; the
        exact path reads ids straight from ``flat_items``, so it needs the padded
        cluster width rather than the probed cluster ids. ``plain`` / ``bloom`` /
        ``exact`` are the backend's three custom ops (identical signatures)."""
        probe_ids, flat_items = self._phase1_probe_with_ids(query)

        if self.has_exact and query_clause_attrs is not None:
            return exact(
                query,
                flat_items,
                self.item_codes,
                self.item_clause_attrs,
                self.clause_is_reverse,
                query_clause_attrs.long(),
                self._global_scale_f,
                self.k,
                # Python int cached at index build (_build_ivf) — no device read and
                # no tensor-shape read, so cudagraph capture stays clean and dynamic
                # compilation cannot turn it into a SymInt.
                self._max_cluster_size,
            )

        if self.has_bloom and query_clause_attrs is not None:
            qb = self._query_bits(query_clause_attrs)
            return bloom(
                query,
                flat_items,
                self.item_codes,
                qb,
                self.bloom_sigs_t,
                probe_ids,
                self._global_scale_f,
                self.k,
            )
        return plain(
            query,
            flat_items,
            self.item_codes,
            self._global_scale_f,
            self.k,
        )

    def _forward_official(
        self,
        query: Tensor,
        query_clause_attrs: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Algorithm 1 phases 2+3 on Meta's official ops (plan §5.1), eager only.

        ``none`` → ``fused_kmean_ann``; ``bloom`` → the official expression parser (plans
        cached on CPU) + ``bloom_index_search_batch_return_partial_response`` over the
        probed clusters + ``fused_kmean_ann_with_partial_masks`` (``OfficialConfig.bloom_path
        ="partial"``, the paper's co-design) or the full-``N`` packed mask into
        ``fused_kmean_ann(filtering_bit_mask=…)`` (``"full"``, the S9 ablation); ``exact`` →
        our Triton ``clause_mask`` over the sorted attrs, packed into the same
        ``filtering_bit_mask`` (phase 2 ours, full ``N`` — labelled so in every table).
        ``max_tensor_size_per_row`` is the static ``n_probe · max_cluster_size`` of the
        padded layout, so the official output has our ``[B, P]`` width (rounded to 32) and
        the ``masked_topk`` epilogue costs the same in every arm."""
        probe_ids = self._phase1_probe_ids(query)
        cfg = self.official
        filtering_bit_mask: Tensor | None = None
        partial: tuple[Tensor, Tensor, Tensor] | None = None

        if self.has_exact and query_clause_attrs is not None:
            mask = clause_mask(
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
            plans = official_mod.parse_plans(expressions, cfg.hash_k, cfg.max_sub_queries)
            if cfg.bloom_path == "partial":
                partial = official_mod.bloom_partial_masks(
                    self.bloom_index,
                    self.bundle_b_offsets,
                    plans,
                    self.cluster_offsets[probe_ids],
                    self.cluster_sizes[probe_ids],
                    self.k_hash,
                    cfg.hash_k,
                )
            else:
                filtering_bit_mask = official_mod.bloom_filtering_mask(
                    self.bloom_index, self.bundle_b_offsets, plans, self.k_hash, cfg.hash_k
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
            self.n_probe * self._max_cluster_size,
            score_path=cfg.score_path,
            divisor=cfg.divisor,
            filtering_bit_mask=filtering_bit_mask,
            partial=partial,
        )

    def _forward_torch_eager(
        self,
        query: Tensor,
        query_clause_attrs: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Eager pure-torch forward mirroring the Triton kernel: IVF probe, optional
        bloom/exact filter, int8xint8 dot + rescale, mask + topk + ``-1``/``-inf`` pad.
        The dot runs in fp32 (not ``torch._int_mm``) — at D=128 the integer products fit the
        fp32 mantissa exactly, so it is bit-identical to an int32 accumulator."""
        flat_items = self._phase1_probe(query)

        valid = flat_items >= 0
        safe = flat_items.clamp_min(0)

        keep = valid
        if self.has_bloom and query_clause_attrs is not None:
            qb = self._query_bits(query_clause_attrs)  # [B, W]
            keep = keep & bloom_subset_match(qb, self.bloom_sigs[safe])
        elif self.has_exact and query_clause_attrs is not None:
            gathered = self.item_clause_attrs[safe]  # [B, P, C, A_max]
            keep = keep & clause_subset_match(
                gathered, query_clause_attrs.long(), self.clause_is_reverse
            )

        q_codes, q_scales = quantize_int8(query)  # [B, D] int8, [B] fp32
        codes = self.item_codes[safe].to(torch.float32)  # [B, P, D]
        scores = torch.einsum("bd,bpd->bp", q_codes.to(torch.float32), codes)
        scores = scores * q_scales.unsqueeze(1) * self.global_scale
        return masked_topk(scores, self.k, valid=keep, gather_ids=flat_items)

    def _forward_candidates(
        self,
        query: Tensor,
        candidate_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.backend == "official":
            # item_codes is cluster-sorted on this backend; candidate ids are original ids.
            candidate_ids = self.inv_perm[candidate_ids]
        cand_codes = self.item_codes[candidate_ids].to(torch.float32)
        q_codes, q_scales = quantize_int8(query)
        scores = torch.bmm(
            q_codes.to(torch.float32).unsqueeze(1), cand_codes.transpose(1, 2)
        ).squeeze(1)
        scores = scores * q_scales.unsqueeze(1) * self.global_scale

        # Scores here are always finite (int8 dot × finite scales), so masked_topk's
        # non-finite → -1 sentinel never fires; pad_to_k=False keeps the current
        # min(k, P)-column, no-pad, no-sentinel semantics when P < k.
        return masked_topk(scores, self.k, gather_ids=candidate_ids, pad_to_k=False)


def build_silvertorch(
    item_embs: Tensor,
    k: int,
    *,
    n_lists: int,
    n_probe: int,
    filter_mode: FilterMode = "none",
    m_bits: int | None = None,
    k_hash: int | None = None,
    n_iter: int = 10,
    seed: int = 0,
    item_clause_attrs: Tensor | None = None,
    clause_is_reverse: Tensor | None = None,
    backend: Backend = "triton",
    official: OfficialConfig | None = None,
) -> SilverTorch:
    """Construct a ``SilverTorch`` and run ``register_index(item_embs, ...)`` in one call."""
    module = SilverTorch(
        k=k,
        n_lists=n_lists,
        n_probe=n_probe,
        filter_mode=filter_mode,
        m_bits=m_bits,
        k_hash=k_hash,
        n_iter=n_iter,
        seed=seed,
        backend=backend,
        official=official,
    )
    module.register_index(item_embs, item_clause_attrs, clause_is_reverse)
    return module
