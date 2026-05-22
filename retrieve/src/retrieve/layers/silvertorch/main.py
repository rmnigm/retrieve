from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from retrieve.interfaces import Backend, RetrievalModule
from retrieve.kernels.triton.silvertorch.codesigned_probe_score import (
    codesigned_probe_score,
)
from retrieve.kernels.triton.silvertorch.codesigned_probe_score_exact import (
    codesigned_probe_score_exact,
)
from retrieve.layers.filters.bloom import (
    _build_query_signatures,
    _build_signatures,
    _generate_seeds,
)
from retrieve.layers.utils.kmeans import KMeansTorch
from retrieve.layers.utils.quantize import quantize_int8, quantize_int8_global

FilterMode = Literal["none", "bloom", "exact"]


class SilverTorch(RetrievalModule):
    """Co-designed IVF + INT8 ANN + (optional) attribute filter (Algorithm 1).

    Paper-faithful int8 ANN: the index stores ``[N, D]`` int8 codes with a
    single global per-tensor ``global_scale``; the query is int8-quantized
    per-row at forward time; the dot product runs int8×int8 → int32 (dp4a
    / IMMA-fallback path inside the codesigned kernel) and dequantizes
    once with ``q_scale * global_scale``. See SilverTorch paper §4.2.

    ``filter`` selects the predicate fused into the codesigned probe+score
    kernel:

    - ``"none"``: plain IVF + INT8 ANN, no attribute filter.
    - ``"bloom"``: paper's bloom subset test (``m_bits``/``k_hash`` required);
      one false-positive-tolerant filter shared across all clauses, fused
      into ``codesigned_probe_score``.
    - ``"exact"``: exact-clause predicate (no false positives), fused into
      ``codesigned_probe_score_exact``. Bandwidth-cheaper per item at small
      ``C × A_max``; trades the bloom hash flexibility for exact-value match.

    ``backend="triton"`` (default) routes phase 2+3 through the fused kernel
    — no ``[B, P, W]`` / ``[B, P, C, A_max]`` / ``[B, P, D]`` intermediates
    touch HBM. ``backend="torch"`` runs the same semantics via pure torch
    ops, eager — callers that want Inductor fusion + cudagraph capture
    should wrap the module with ``torch.compile`` themselves. The torch
    path materializes ``[B, P, D]`` fp32 inside the dot product, so
    configurations with large ``P × B × D`` must use the Triton backend.

    **Quality knob.** Global scale matches the paper; on indexes with a
    long-tailed row-norm distribution it can drop ~1–2% recall@K vs a
    per-item-scale variant (`quantize_int8`). To swap: store
    ``item_scales [N] fp32`` from ``quantize_int8`` instead of
    ``global_scale``, propagate it through the three forward paths, and
    have the kernel gather ``item_scales[safe_ids]`` in place of the scalar
    ``global_scale`` in the dequant epilogue. Bench numbers in
    ``_agent_scratch/bench_results/silvertorch_int8mm_quality.json``.
    """

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
        if self.filter == "none" and item_clause_attrs is not None:
            raise ValueError(
                "item_clause_attrs requires filter='bloom' or filter='exact'"
            )
        if self.filter == "none" and clause_is_reverse is not None:
            raise ValueError("clause_is_reverse requires filter='exact'")
        if self.filter == "exact" and item_clause_attrs is None:
            raise ValueError("filter='exact' requires item_clause_attrs at register_index")
        if self.filter == "bloom" and clause_is_reverse is not None:
            raise ValueError("clause_is_reverse is only used with filter='exact'")

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

        codes, global_scale = quantize_int8_global(item_embs)

        self.register_buffer("centroids", centroids)
        self.register_buffer("item_codes", codes)
        # 0-d fp32 buffer: moves with .to(device), parameterizes the kernel's
        # epilogue without forcing a recompile per index.
        self.register_buffer(
            "global_scale",
            torch.tensor(global_scale, dtype=torch.float32, device=item_embs.device),
        )
        self.register_buffer("padded_cluster_items", padded)
        self.register_buffer("cluster_sizes", cluster_sizes)

        if self.filter == "bloom":
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
        elif self.filter == "exact":
            assert item_clause_attrs is not None  # narrowed by the validation above
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
        """IVF + (optional) attribute-filter-fused retrieval.

        ``query_clause_attrs`` is only valid when ``filter`` is ``"bloom"`` or
        ``"exact"``. With it ``None``, the filter branch is skipped entirely
        and the kernel runs as plain IVF + INT8 ANN.
        """
        if candidate_ids is not None:
            return self._forward_candidates(query, candidate_ids)
        if self.filter == "none" and query_clause_attrs is not None:
            raise ValueError(
                "query_clause_attrs requires filter='bloom' or filter='exact'"
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

        if self.has_exact and query_clause_attrs is not None:
            return codesigned_probe_score_exact(
                query,
                flat_items,
                self.item_codes,
                float(self.global_scale.item()),
                self.k,
                item_clause_attrs=self.item_clause_attrs,
                clause_is_reverse=self.clause_is_reverse,
                query_clause_attrs=query_clause_attrs.long(),
            )

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
            float(self.global_scale.item()),
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

        Mirrors the Triton kernel semantics: phase 1 IVF probe, optional
        bloom subset test OR exact-clause predicate, int8×int8 → int32 dot +
        per-row × global rescale, mask + topk + ``-1 / -inf`` pad. Calls
        the eager bloom-sig body directly so the whole forward traces into
        one graph.

        The dot is computed in fp32 (cast from int8) rather than via
        ``torch._int_mm`` because the kernel reference uses fp32 here — at
        D=128 the integer products fit in fp32 mantissa exactly, so the
        result is bit-identical to an int32-then-cast accumulator.
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
        elif self.has_exact and query_clause_attrs is not None:
            qa = query_clause_attrs.long()
            gathered = self.item_clause_attrs[safe]  # [B, P, C, A_max]
            q = qa.unsqueeze(1).unsqueeze(-1)  # [B, 1, C, 1]
            clause_match = (gathered == q).any(dim=-1)  # [B, P, C]
            rev = self.clause_is_reverse.unsqueeze(0).unsqueeze(0)  # [1, 1, C]
            clause_match = torch.where(rev, ~clause_match, clause_match)
            inactive = (qa == -1).unsqueeze(1)  # [B, 1, C]
            keep = keep & (clause_match | inactive).all(dim=-1)

        q_codes, q_scales = quantize_int8(query)  # [B, D] int8, [B] fp32
        codes = self.item_codes[safe].to(torch.float32)  # [B, P, D]
        scores = torch.einsum("bd,bpd->bp", q_codes.to(torch.float32), codes)
        scores = scores * q_scales.unsqueeze(1) * self.global_scale
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
        cand_codes = self.item_codes[candidate_ids].to(torch.float32)
        q_codes, q_scales = quantize_int8(query)
        scores = torch.bmm(
            q_codes.to(torch.float32).unsqueeze(1), cand_codes.transpose(1, 2)
        ).squeeze(1)
        scores = scores * q_scales.unsqueeze(1) * self.global_scale

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
    filter: FilterMode = "none",
    m_bits: int | None = None,
    k_hash: int | None = None,
    n_iter: int = 10,
    seed: int = 0,
    item_clause_attrs: Tensor | None = None,
    clause_is_reverse: Tensor | None = None,
    backend: Backend = "triton",
) -> SilverTorch:
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
