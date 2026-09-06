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

Correctness alongside capture: scores must be ``torch.equal`` to eager, and ids equal up
to ties among equal scores — the compaction kernels write their row in atomic order, so
two runs may order equal-scoring candidates differently even eager-vs-eager.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from retrieve.layers.filters import BloomFilter, ExactAttributeFilter
from retrieve.layers.linr import OneBitKNN, PostfilterKNN, PrefilterKNN
from tests.conftest import make_attrs, make_index, make_query, make_query_attrs

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


def _assert_ids_equal_up_to_ties(out_ids: Tensor, ref_ids: Tensor, scores: Tensor) -> None:
    """Ids must agree once slots holding an equal score are treated as interchangeable.

    Called only after the scores tensors compared ``torch.equal``, so grouping by the one
    scores tensor is well defined: for every row and every distinct score value, the two
    id multisets at that value must match.
    """
    assert out_ids.shape == ref_ids.shape
    for b in range(out_ids.shape[0]):
        row = scores[b]
        for value in torch.unique(row):
            slots = row == value
            got = sorted(out_ids[b][slots].tolist())
            want = sorted(ref_ids[b][slots].tolist())
            assert got == want, (
                f"row {b}, score {value.item()}: ids {got} != {want} — a difference that "
                f"ties among equal scores do not explain"
            )


@pytest.mark.parametrize("algo", ["linr_v1", "linr_v2", "linr_v3"])
@pytest.mark.parametrize("filter_kind", ["clause", "bloom"])
def test_compiled_captures_without_cudagraph_skips(algo, filter_kind):
    """Zero ``cudagraph_skips`` on the warm-up, and the captured forward matches eager."""
    from torch._dynamo.utils import counters

    torch.manual_seed(0)
    module = _build(algo, filter_kind)
    query = make_query(B, D)
    q_attrs = make_query_attrs(B, c=C)

    with torch.inference_mode():
        eager_ids, eager_scores = (t.clone() for t in module(query, q_attrs))

    torch._dynamo.reset()
    skips_before = int(counters["inductor"]["cudagraph_skips"])
    compiled = torch.compile(module, mode="reduce-overhead", dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for _ in range(5):
            compiled(query, q_attrs)
        torch.cuda.synchronize()
        skips = int(counters["inductor"]["cudagraph_skips"]) - skips_before
        # cudagraph-tree outputs are reclaimed on the next call — clone before comparing.
        out_ids, out_scores = (t.clone() for t in compiled(query, q_attrs))
        torch.cuda.synchronize()

    assert skips == 0, (
        f"{algo} / {filter_kind}: inductor skipped cudagraphs {skips}x — the compiled "
        f"forward fell back to compiled-eager, which is what the harness reports as "
        f"NotCapturable and what made A1's golden `graph` cells not graph numbers"
    )
    assert torch.equal(out_scores, eager_scores), (
        f"{algo} / {filter_kind}: compiled scores differ from eager"
    )
    _assert_ids_equal_up_to_ties(out_ids, eager_ids, eager_scores)
