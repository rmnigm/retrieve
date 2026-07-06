from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from retrieve.interfaces import Backend, RetrievalModule
from retrieve.kernels.silvertorch.codesigned_probe_score import (
    codesigned_probe_score,
    codesigned_probe_score_bloom,
)
from retrieve.kernels.silvertorch.codesigned_probe_score_exact import (
    codesigned_probe_score_exact,
)
from retrieve.layers.filters.bloom_hash import (
    bloom_subset_match,
    build_query_signatures,
    build_signatures,
    generate_seeds,
)
from retrieve.layers.filters.exact_attribute import clause_subset_match
from retrieve.layers.utils.kmeans import KMeansTorch
from retrieve.layers.utils.quantize import quantize_int8, quantize_int8_global
from retrieve.layers.utils.topk import masked_topk

FilterMode = Literal["none", "bloom", "exact"]


class SilverTorch(RetrievalModule):
    """Co-designed IVF + INT8 ANN + optional attribute filter (paper Algorithm 1, §4.2): an ``[N,
    D]`` int8 index with one global scale, per-row int8-quantized queries, and an int8×int8 →
    int32 dot dequantized once. ``filter`` fuses a predicate into the probe+score kernel —
    ``"none"`` (plain ANN), ``"bloom"`` (subset test, needs ``m_bits``/``k_hash``), or
    ``"exact"`` (exact-clause, no false positives).

    ``backend="triton"`` (default) keeps all probe intermediates off HBM; ``backend="torch"``
    runs the same semantics eager but materializes ``[B, P, D]``, so large ``P·B·D`` needs the
    Triton backend."""

    centroids: Tensor
    item_codes: Tensor
    global_scale: Tensor
    padded_cluster_items: Tensor
    cluster_sizes: Tensor
    bloom_sigs: Tensor
    hash_seeds: Tensor
    item_clause_attrs: Tensor
    clause_is_reverse: Tensor

    def __init__(
        self,
        k: int,
        n_lists: int,
        n_probe: int,
        filter: FilterMode = "none",
        m_bits: int | None = None,
        k_hash: int | None = None,
        n_iter: int = 10,
        seed: int = 0,
        backend: Backend = "triton",
    ) -> None:
        super().__init__()
        if filter not in ("none", "bloom", "exact"):
            raise ValueError(f"filter must be 'none', 'bloom', or 'exact', got {filter!r}")

        if filter == "bloom":
            if m_bits is None or k_hash is None:
                raise ValueError("filter='bloom' requires both m_bits and k_hash")
            if m_bits <= 0 or (m_bits & (m_bits - 1)) != 0:
                raise ValueError(f"m_bits must be a positive power of 2, got {m_bits}")
            if m_bits % 64 != 0:
                raise ValueError(f"m_bits must be a multiple of 64, got {m_bits}")
            if k_hash <= 0:
                raise ValueError(f"k_hash must be positive, got {k_hash}")
            self.m_bits = m_bits
            self.k_hash = k_hash
            self.word_count = m_bits // 64
        else:
            if m_bits is not None or k_hash is not None:
                raise ValueError(
                    f"m_bits/k_hash only apply to filter='bloom', got filter={filter!r}"
                )
            self.m_bits = 0
            self.k_hash = 0
            self.word_count = 0

        self.filter: FilterMode = filter
        self.k = k
        self.n_lists = n_lists
        self.n_probe = n_probe
        self.n_iter = n_iter
        self.seed = seed
        self.backend = backend

    @property
    def has_bloom(self) -> bool:
        return self.filter == "bloom"

    @property
    def has_exact(self) -> bool:
        return self.filter == "exact"

    def register_index(
        self,
        item_embs: Tensor,
        item_clause_attrs: Tensor | None = None,
        clause_is_reverse: Tensor | None = None,
    ) -> None:
        self._validate_register_args(item_embs, item_clause_attrs, clause_is_reverse)
        padded, cluster_sizes = self._build_ivf(item_embs)
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
        if self.filter == "none" and item_clause_attrs is not None:
            raise ValueError("item_clause_attrs requires filter='bloom' or filter='exact'")
        if self.filter == "none" and clause_is_reverse is not None:
            raise ValueError("clause_is_reverse requires filter='exact'")
        if self.filter == "exact" and item_clause_attrs is None:
            raise ValueError("filter='exact' requires item_clause_attrs at register_index")
        if self.filter == "bloom" and clause_is_reverse is not None:
            raise ValueError("clause_is_reverse is only used with filter='exact'")

        n = item_embs.shape[0]
        if self.n_lists > n:
            raise ValueError(f"n_lists ({self.n_lists}) cannot exceed N ({n}).")
        if self.n_probe > self.n_lists:
            raise ValueError(f"n_probe ({self.n_probe}) cannot exceed n_lists ({self.n_lists}).")

    def _build_ivf(self, item_embs: Tensor) -> tuple[Tensor, Tensor]:
        """K-means clustering; registers ``centroids`` and returns ``(padded_cluster_items,
        cluster_sizes)`` for registration after the quantization buffers (frozen order)."""
        n = item_embs.shape[0]
        centroids, assignments = KMeansTorch(
            n_lists=self.n_lists, n_iter=self.n_iter, seed=self.seed
        ).fit(item_embs)
        cluster_sizes = torch.bincount(assignments, minlength=self.n_lists)
        max_size = int(cluster_sizes.max().item())

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
        return padded, cluster_sizes

    def _quantize_items(self, item_embs: Tensor) -> None:
        codes, global_scale = quantize_int8_global(item_embs)
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
        if self.filter == "bloom":
            seeds = generate_seeds(self.k_hash, device=device)
            if item_clause_attrs is None:
                sigs = torch.zeros(n, self.word_count, dtype=torch.int64, device=device)
            else:
                sigs = build_signatures(
                    item_clause_attrs.long(),
                    seeds,
                    self.m_bits,
                    self.k_hash,
                    self.word_count,
                )
            self.register_buffer("bloom_sigs", sigs)
            self.register_buffer("hash_seeds", seeds)
        elif self.filter == "exact":
            assert item_clause_attrs is not None  # narrowed by _validate_register_args
            c = item_clause_attrs.shape[1]
            if clause_is_reverse is None:
                clause_is_reverse = torch.zeros(
                    c, dtype=torch.bool, device=item_clause_attrs.device
                )
            self.register_buffer("item_clause_attrs", item_clause_attrs.long())
            self.register_buffer("clause_is_reverse", clause_is_reverse)

    def forward(
        self,
        query: Tensor,
        query_clause_attrs: Tensor | None = None,
        candidate_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """IVF + (optional) attribute-filter-fused retrieval; ``query_clause_attrs`` is valid only
        for ``filter="bloom"|"exact"`` and when ``None`` the filter branch is skipped (plain IVF
        + INT8 ANN)."""
        if candidate_ids is not None:
            if query_clause_attrs is not None:
                raise ValueError(
                    "candidate_ids path scores the given candidates without the fused "
                    "attribute filter; pass query_clause_attrs OR candidate_ids, not both"
                )
            return self._forward_candidates(query, candidate_ids)
        if self.filter == "none" and query_clause_attrs is not None:
            raise ValueError("query_clause_attrs requires filter='bloom' or filter='exact'")
        if self.backend == "triton":
            return self._forward_triton(query, query_clause_attrs)
        return self._forward_torch_eager(query, query_clause_attrs)

    def _phase1_probe(self, query: Tensor) -> Tensor:
        """Phase 1: centroid top-``n_probe`` then gather padded probed items; returns ``flat_items
        [B, P]`` (P = n_probe × max_cluster_size) with ``-1`` marking empty-cluster padding."""
        b = query.shape[0]
        cent_scores = query @ self.centroids.t()
        _, probe_ids = torch.topk(cent_scores, self.n_probe, dim=1)
        probed = self.padded_cluster_items[probe_ids]
        return probed.reshape(b, -1)

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
            qb = build_query_signatures(
                query_clause_attrs.long().unsqueeze(-1),
                self.hash_seeds,
                self.m_bits,
                self.k_hash,
                self.word_count,
            )
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

    def _forward_torch_eager(
        self,
        query: Tensor,
        query_clause_attrs: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Pure-torch forward (compiled in ``__init__``) mirroring the Triton kernel: IVF probe,
        optional bloom/exact filter, int8×int8 dot + rescale, mask + topk + ``-1``/``-inf`` pad.
        The dot runs in fp32 (not ``torch._int_mm``) — at D=128 the integer products fit the fp32
        mantissa exactly, so it's bit-identical to an int32 accumulator."""
        flat_items = self._phase1_probe(query)

        valid = flat_items >= 0
        safe = flat_items.clamp_min(0)

        keep = valid
        if self.has_bloom and query_clause_attrs is not None:
            qb = build_query_signatures(
                query_clause_attrs.long().unsqueeze(-1),
                self.hash_seeds,
                self.m_bits,
                self.k_hash,
                self.word_count,
            )  # [B, W]
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
    filter: FilterMode = "none",
    m_bits: int | None = None,
    k_hash: int | None = None,
    n_iter: int = 10,
    seed: int = 0,
    item_clause_attrs: Tensor | None = None,
    clause_is_reverse: Tensor | None = None,
    backend: Backend = "triton",
) -> SilverTorch:
    """Construct a ``SilverTorch`` and run ``register_index(item_embs, ...)`` in one call."""
    module = SilverTorch(
        k=k,
        n_lists=n_lists,
        n_probe=n_probe,
        filter=filter,
        m_bits=m_bits,
        k_hash=k_hash,
        n_iter=n_iter,
        seed=seed,
        backend=backend,
    )
    module.register_index(item_embs, item_clause_attrs, clause_is_reverse)
    return module
