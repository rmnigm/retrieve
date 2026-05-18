from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import Backend, RetrievalModule
from retrieve.kernels.triton.silvertorch.codesigned_probe_score import (
    codesigned_probe_score,
)
from retrieve.layers.filters.bloom import (
    _build_query_signatures,
    _build_signatures,
    _generate_seeds,
)
from retrieve.layers.utils.kmeans import KMeansTorch
from retrieve.layers.utils.quantize import quantize_int8


class SilverTorch(RetrievalModule):
    """Co-designed IVF + INT8 ANN + (optional) Bloom attribute filter (Algorithm 1).

    With ``m_bits`` and ``k_hash`` set, the bloom filter is fused into the
    ``codesigned_probe_score`` Triton kernel (or its torch equivalent —
    see ``backend=`` below). Leave both unset (or both ``None``) to build a
    bloom-free IVF + INT8 ANN — no signature buffers are allocated and the
    bloom branch is skipped at query time.

    ``backend="triton"`` (default) routes phase 2+3 through the fused
    ``codesigned_probe_score`` kernel — no ``[B, P, W]`` / ``[B, P, D]``
    intermediates touch HBM. ``backend="torch"`` runs the same semantics
    via pure torch ops, eager — callers that want Inductor fusion + cudagraph
    capture should wrap the module with ``torch.compile`` themselves (the
    evaluation harness already does this). The torch path materializes
    ``[B, P, D]`` fp32 inside the dot product, so configurations with
    large ``P × B × D`` must use the Triton backend.
    """

    centroids: Tensor
    item_codes: Tensor
    item_scales: Tensor
    padded_cluster_items: Tensor
    cluster_sizes: Tensor
    bloom_sigs: Tensor
    hash_seeds: Tensor

    def __init__(
        self,
        k: int,
        n_lists: int,
        n_probe: int,
        m_bits: int | None = None,
        k_hash: int | None = None,
        n_iter: int = 10,
        seed: int = 0,
        backend: Backend = "triton",
    ) -> None:
        super().__init__()
        if (m_bits is None) ^ (k_hash is None):
            raise ValueError("m_bits and k_hash must be set together (or both left unset)")
        self.has_bloom = m_bits is not None
        if self.has_bloom:
            assert m_bits is not None and k_hash is not None  # narrow for type-checkers
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
            self.m_bits = 0
            self.k_hash = 0
            self.word_count = 0
        self.k = k
        self.n_lists = n_lists
        self.n_probe = n_probe
        self.n_iter = n_iter
        self.seed = seed
        self.backend = backend

    def register_index(
        self,
        item_embs: Tensor,
        item_clause_attrs: Tensor | None = None,
    ) -> None:
        if not self.has_bloom and item_clause_attrs is not None:
            raise ValueError(
                "item_clause_attrs requires bloom config — pass m_bits and k_hash to __init__"
            )
        n, _ = item_embs.shape
        if self.n_lists > n:
            raise ValueError(f"n_lists ({self.n_lists}) cannot exceed N ({n}).")
        if self.n_probe > self.n_lists:
            raise ValueError(f"n_probe ({self.n_probe}) cannot exceed n_lists ({self.n_lists}).")

        centroids, assignments = KMeansTorch(
            n_lists=self.n_lists, n_iter=self.n_iter, seed=self.seed
        ).fit(item_embs)
        cluster_sizes = torch.bincount(assignments, minlength=self.n_lists)
        max_size = int(cluster_sizes.max().item())

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

        codes, scales = quantize_int8(item_embs)

        self.register_buffer("centroids", centroids)
        self.register_buffer("item_codes", codes)
        self.register_buffer("item_scales", scales)
        self.register_buffer("padded_cluster_items", padded)
        self.register_buffer("cluster_sizes", cluster_sizes)

        if self.has_bloom:
            seeds = _generate_seeds(self.k_hash, device=item_embs.device)
            if item_clause_attrs is None:
                sigs = torch.zeros(n, self.word_count, dtype=torch.int64, device=item_embs.device)
            else:
                sigs = _build_signatures(
                    item_clause_attrs.long(),
                    seeds,
                    self.m_bits,
                    self.k_hash,
                    self.word_count,
                )
            self.register_buffer("bloom_sigs", sigs)
            self.register_buffer("hash_seeds", seeds)

    def forward(
        self,
        query: Tensor,
        query_clause_attrs: Tensor | None = None,
        candidate_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """IVF + (optional) Bloom-fused retrieval.

        ``query_clause_attrs`` is only valid when the index was built with a
        bloom config. With it ``None``, the bloom branch is skipped entirely.
        """
        if candidate_ids is not None:
            return self._forward_candidates(query, candidate_ids)
        if not self.has_bloom and query_clause_attrs is not None:
            raise ValueError(
                "query_clause_attrs requires bloom config — pass m_bits and k_hash to __init__"
            )
        if self.backend == "triton":
            return self._forward_triton(query, query_clause_attrs)
        return self._forward_torch_eager(query, query_clause_attrs)

    def _phase1_probe(self, query: Tensor) -> Tensor:
        """Phase 1: centroid top-``n_probe``, gather padded probed items.

        Returns ``flat_items[B, P]`` (P = n_probe × max_cluster_size); ``-1``
        slots mark empty cluster padding.
        """
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
        if self.has_bloom and query_clause_attrs is not None:
            qb = _build_query_signatures(
                query_clause_attrs.long().unsqueeze(-1),
                self.hash_seeds,
                self.m_bits,
                self.k_hash,
                self.word_count,
            )
            sigs = self.bloom_sigs
        else:
            qb = None
            sigs = None
        return codesigned_probe_score(
            query,
            flat_items,
            self.item_codes,
            self.item_scales,
            self.k,
            query_bits=qb,
            bloom_sigs=sigs,
        )

    def _forward_torch_eager(
        self,
        query: Tensor,
        query_clause_attrs: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Pure-torch forward — compiled in ``__init__``.

        Mirrors the Triton kernel's semantics: phase 1 IVF probe, optional
        bloom subset test, INT8 dequant + dot, mask + topk + ``-1 / -inf``
        pad. Calls the eager bloom-sig body directly so the whole forward
        traces into one graph.
        """
        b = query.shape[0]
        flat_items = self._phase1_probe(query)
        p = flat_items.shape[1]

        valid = flat_items >= 0
        safe = flat_items.clamp_min(0)

        keep = valid
        if self.has_bloom and query_clause_attrs is not None:
            qb = _build_query_signatures(
                query_clause_attrs.long().unsqueeze(-1),
                self.hash_seeds,
                self.m_bits,
                self.k_hash,
                self.word_count,
            )  # [B, W]
            probed_sigs = self.bloom_sigs[safe]  # [B, P, W]
            match = (qb.unsqueeze(1) & probed_sigs) == qb.unsqueeze(1)
            keep = keep & match.all(dim=-1)

        codes = self.item_codes[safe].to(torch.float32)  # [B, P, D]
        scales = self.item_scales[safe]  # [B, P]
        scores = torch.einsum("bd,bpd->bp", query, codes) * scales
        scores = scores.masked_fill(~keep, float("-inf"))

        actual_k = min(self.k, p)
        topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
        topk_ids = flat_items.gather(1, topk_local)

        if actual_k == self.k:
            return topk_ids, topk_scores

        device = query.device
        out_ids = torch.full((b, self.k), -1, dtype=torch.long, device=device)
        out_scores = torch.full((b, self.k), float("-inf"), dtype=torch.float32, device=device)
        out_ids[:, :actual_k] = topk_ids
        out_scores[:, :actual_k] = topk_scores
        return out_ids, out_scores

    def _forward_candidates(
        self,
        query: Tensor,
        candidate_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cand_codes = self.item_codes[candidate_ids]
        cand_scales = self.item_scales[candidate_ids]
        cand_embs = cand_codes.float() * cand_scales.unsqueeze(-1)
        scores = torch.bmm(query.unsqueeze(1), cand_embs.transpose(1, 2)).squeeze(1)

        actual_k = min(self.k, scores.shape[1])
        topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
        topk_ids = candidate_ids.gather(1, topk_local)
        return topk_ids, topk_scores


def build_silvertorch(
    item_embs: Tensor,
    k: int,
    *,
    n_lists: int,
    n_probe: int,
    m_bits: int | None = None,
    k_hash: int | None = None,
    n_iter: int = 10,
    seed: int = 0,
    item_clause_attrs: Tensor | None = None,
    backend: Backend = "triton",
) -> SilverTorch:
    module = SilverTorch(
        k=k,
        n_lists=n_lists,
        n_probe=n_probe,
        m_bits=m_bits,
        k_hash=k_hash,
        n_iter=n_iter,
        seed=seed,
        backend=backend,
    )
    module.register_index(item_embs, item_clause_attrs)
    return module
