"""Cudagraph-capture regression for the compiled LiNR paths on the Triton backend.

Roadmap C4 found that compiled ``linr_v2`` / ``linr_v3`` never actually ran under
cudagraph trees: inductor reported ``skipping cudagraphs due to mutated inputs`` and
silently fell back to compiled-eager, so A1's golden ``graph`` cells for those two algos
were not graph numbers at all. The cause was the ``@triton_op`` registration of
``clause_compact`` / ``bloom_compact`` — inductor's TTIR mutation analysis walks a store
address back through every argument of the ``tt.call`` that produced it, and
``compact_store``'s address is data-dependent on the pass mask, so the index buffers
(graph inputs) were reported as mutated. Both kernels are now opaque ``custom_op``s.

This test is the gate on that: it builds the three filtered LiNR variants the way
``evaluation/retrieval/algos.py`` does, compiles them with the same settings
``evaluation/retrieval/bench.py:graph_callable`` uses (``mode="reduce-overhead"``,
``dynamic=False``, ``fullgraph=True``, five warm-up calls) and reads the same counter
(``torch._dynamo.utils.counters["inductor"]["cudagraph_skips"]``). A non-zero skip count
is the exact failure the harness turns into ``NotCapturable``.

Correctness alongside capture: after the warm-up the graph replays on two queries it was not
captured on, each ``torch.equal`` to eager on its scores and equal up to ties on its ids
(``torch.topk``'s tie order is not guaranteed stable).
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch._dynamo.utils import counters

from retrieve.modules import (
    BloomFilter,
    ExactAttributeFilter,
    OneBitKNN,
    PostfilterKNN,
    PrefilterKNN,
)
from tests.conftest import make_attrs, make_index, make_query, make_query_attrs
from tests.parity.conftest import assert_topk_equal

N, D, B, C, A_MAX, K = 512, 64, 4, 2, 2, 8
CANDIDATE_POOL = 64


class _V1(nn.Module):
    """LiNR V1 — dense matmul + ``[B, N]`` bool mask from the filter."""

    def __init__(self, embs: Tensor, filter_mod):
        super().__init__()
        self.idx = PostfilterKNN(k=K, backend="triton")
        self.idx.register_index(embs)
        self.filter = filter_mod

    def forward(self, q: Tensor, qa: Tensor):
        return self.idx(q, mask=self.filter.evaluate_mask(qa))


class _V2(nn.Module):
    """LiNR V2 — the filter's compact candidate list rescored exactly."""

    def __init__(self, embs: Tensor, filter_mod):
        super().__init__()
        self.idx = PrefilterKNN(k=K, backend="triton")
        self.idx.register_index(embs)
        self.filter = filter_mod

    def forward(self, q: Tensor, qa: Tensor):
        cand, counts = self.filter.evaluate_indices(qa)
        return self.idx(q, candidate_ids=cand, counts=counts)


class _V3(nn.Module):
    """LiNR V3 — filter → 1-bit OPORP pool → exact rescoring."""

    def __init__(self, embs: Tensor, filter_mod):
        super().__init__()
        self.stage1 = OneBitKNN(k=CANDIDATE_POOL, seed=0, backend="triton")
        self.stage1.register_index(embs)
        self.stage2 = PrefilterKNN(k=K, backend="triton")
        self.stage2.register_index(embs)
        self.filter = filter_mod

    def forward(self, q: Tensor, qa: Tensor):
        pos, pcounts = self.filter.evaluate_indices(qa)
        cand, _ = self.stage1(q, candidate_ids=pos, counts=pcounts)
        return self.stage2(q, candidate_ids=cand, counts=(cand >= 0).sum(dim=1))


VARIANTS = {"linr_v1": _V1, "linr_v2": _V2, "linr_v3": _V3}


def _build(algo: str, filter_kind: str) -> nn.Module:
    embs = make_index(N, D)
    attrs = make_attrs(N, c=C, a_max=A_MAX)
    if filter_kind == "clause":
        f = ExactAttributeFilter(backend="triton")
        f.register_index(attrs)
    else:
        f = BloomFilter(m_bits=512, k_hash=4, backend="triton")
        f.register_index(attrs)
    return VARIANTS[algo](embs, f)


def _assert_replays_match_eager(module: nn.Module, b: int, label: str) -> None:
    """Compile as the harness does, warm up five times on one query, then replay the captured
    graph on two different queries: zero ``cudagraph_skips``, and each replay equals eager on
    the same inputs (scores ``torch.equal``, ids up to ties)."""

    torch._dynamo.reset()
    skips_before = int(counters["inductor"]["cudagraph_skips"])
    compiled = torch.compile(module, mode="reduce-overhead", dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for _ in range(5):
            compiled(make_query(b, D), make_query_attrs(b, c=C))
        for seed in (21, 22):
            query, q_attrs = make_query(b, D, seed=seed), make_query_attrs(b, c=C, seed=seed)
            # cudagraph-tree outputs are reclaimed on the next call — clone before comparing.
            out_ids, out_scores = (t.clone() for t in compiled(query, q_attrs))
            eager_ids, eager_scores = module(query, q_attrs)
            assert_topk_equal(out_ids, out_scores, eager_ids, eager_scores)
    skips = int(counters["inductor"]["cudagraph_skips"]) - skips_before
    assert skips == 0, (
        f"{label}: inductor skipped cudagraphs {skips}x — the compiled forward fell back to "
        f"compiled-eager, which is what the harness reports as NotCapturable"
    )


@pytest.mark.parametrize("algo", ["linr_v1", "linr_v2", "linr_v3"])
@pytest.mark.parametrize("filter_kind", ["clause", "bloom"])
def test_compiled_captures_without_cudagraph_skips(algo, filter_kind):
    """Zero ``cudagraph_skips``, and the captured forward replays to eager on new inputs."""
    torch.manual_seed(0)
    module = _build(algo, filter_kind)
    _assert_replays_match_eager(module, B, f"{algo} / {filter_kind}")


def test_compiled_batch_of_one_bloom_v2():
    """Pins L3's first defect: the two-phase ``bloom_compact`` returned ``counts`` as a view at
    element offset ``T - 1`` of its scan buffer when ``B == 1`` (``contiguous()`` is a no-op on a
    ``[1]`` view), and inductor's ``assert_alignment`` on custom-op outputs rejected the compiled
    forward. ``N = 512`` at ``block_n = 256`` is the smallest odd-offset shape."""
    torch.manual_seed(0)
    _assert_replays_match_eager(_build("linr_v2", "bloom"), 1, "linr_v2 / bloom, B=1")
