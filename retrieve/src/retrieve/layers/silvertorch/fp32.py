from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import RetrievalModule
from retrieve.kernels.triton.silvertorch.codesigned_probe_score_fp32 import (
    codesigned_probe_score_fp32,
)
from retrieve.layers.filters.bloom import _build_signatures, _generate_seeds
from retrieve.layers.utils.kmeans import KMeansTorch


class SilverTorchFp32(RetrievalModule):
    """Co-designed IVF + FP32 ANN + (optional) Bloom attribute filter.

    Fp32 sibling of ``SilverTorch``: same IVF + bloom co-design, but scores
    against raw fp32 item embeddings instead of int8 codes + scales. With
    ``m_bits`` and ``k_hash`` set, the bloom filter is fused into the
    ``codesigned_probe_score_fp32`` Triton kernel. Leave both unset (or both
    ``None``) to build a bloom-free IVF + FP32 ANN — no signature buffers
    are allocated and the bloom branch is skipped at query time. ``mask`` is
    honored independently in either configuration.
    """

    centroids: Tensor
    item_embs: Tensor
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
    ) -> None:
        super().__init__()
        if (m_bits is None) ^ (k_hash is None):
            raise ValueError("m_bits and k_hash must be set together (or both left unset)")
        self.has_bloom = m_bits is not None
        if self.has_bloom:
            assert m_bits is not None and k_hash is not None
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

        self.register_buffer("centroids", centroids)
        self.register_buffer("item_embs", item_embs.contiguous().to(torch.float32))
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
        mask: Tensor | None = None,
        candidate_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if candidate_ids is not None:
            return self._forward_candidates(query, candidate_ids)

        if not self.has_bloom and query_clause_attrs is not None:
            raise ValueError(
                "query_clause_attrs requires bloom config — pass m_bits and k_hash to __init__"
            )

        b = query.shape[0]

        cent_scores = query @ self.centroids.t()
        _, probe_ids = torch.topk(cent_scores, self.n_probe, dim=1)

        probed = self.padded_cluster_items[probe_ids]
        flat_items = probed.reshape(b, -1)

        if self.has_bloom and query_clause_attrs is not None:
            qb = _build_signatures(
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

        return codesigned_probe_score_fp32(
            query,
            flat_items,
            self.item_embs,
            self.k,
            query_bits=qb,
            bloom_sigs=sigs,
            mask=mask,
        )

    def _forward_candidates(
        self,
        query: Tensor,
        candidate_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cand_embs = self.item_embs[candidate_ids]
        scores = torch.bmm(query.unsqueeze(1), cand_embs.transpose(1, 2)).squeeze(1)

        actual_k = min(self.k, scores.shape[1])
        topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
        topk_ids = candidate_ids.gather(1, topk_local)
        return topk_ids, topk_scores


def build_silvertorch_fp32(
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
) -> SilverTorchFp32:
    module = SilverTorchFp32(
        k=k,
        n_lists=n_lists,
        n_probe=n_probe,
        m_bits=m_bits,
        k_hash=k_hash,
        n_iter=n_iter,
        seed=seed,
    )
    module.register_index(item_embs, item_clause_attrs)
    return module
