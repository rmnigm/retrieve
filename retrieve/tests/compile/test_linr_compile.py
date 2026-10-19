"""Cudagraph-capture gate for the compiled LiNR paths on the Triton backend.

The compaction kernel is an opaque ``custom_op``: under ``triton_op`` inductor's TTIR mutation
analysis walked ``compact_store``'s data-dependent address back to the index buffers, reported
them as mutated, and silently fell back to compiled-eager (kernels.md, "Graph-break behavior").
This builds the three LiNR variants with the exact filter the way the harness does,
compiles them with the
harness's ``graph`` settings (``mode="reduce-overhead"``, ``dynamic=False``, ``fullgraph=True``,
five warm-up calls) and reads the same counter
(``torch._dynamo.utils.counters["inductor"]["cudagraph_skips"]``). A non-zero skip count is the
failure the harness turns into ``NotCapturable``.

Correctness alongside capture: after the warm-up the graph replays on two queries it was not
captured on, each ``torch.equal`` to eager on its scores and equal up to ties on its ids.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch._dynamo.utils import counters

from retrieve.modules import ExactAttributeFilter, OneBitKNN, PostfilterKNN, PrefilterKNN
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

    def prepare_queries(self, qa: Tensor) -> Tensor:
        return self.filter.prepare_queries(qa)

    def forward(self, q: Tensor, qf: Tensor):
        return self.idx(q, mask=self.filter.evaluate_mask(qf))


class _V2(nn.Module):
    """LiNR V2 — the filter's compact candidate list rescored exactly."""

    def __init__(self, embs: Tensor, filter_mod):
        super().__init__()
        self.idx = PrefilterKNN(k=K, backend="triton")
        self.idx.register_index(embs)
        self.filter = filter_mod

    def prepare_queries(self, qa: Tensor) -> Tensor:
        return self.filter.prepare_queries(qa)

    def forward(self, q: Tensor, qf: Tensor):
        cand, counts = self.filter.evaluate_indices(qf)
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

    def prepare_queries(self, qa: Tensor) -> Tensor:
        return self.filter.prepare_queries(qa)

    def forward(self, q: Tensor, qf: Tensor):
        pos, pcounts = self.filter.evaluate_indices(qf)
        cand, _ = self.stage1(q, candidate_ids=pos, counts=pcounts)
        return self.stage2(q, candidate_ids=cand, counts=(cand >= 0).sum(dim=1))


VARIANTS = {"linr_v1": _V1, "linr_v2": _V2, "linr_v3": _V3}


def _build(algo: str, d: int = D) -> nn.Module:
    embs = make_index(N, d)
    f = ExactAttributeFilter(backend="triton")
    f.register_index(make_attrs(N, c=C, a_max=A_MAX))
    return VARIANTS[algo](embs, f)


def _assert_replays_match_eager(module: nn.Module, b: int, label: str, d: int = D) -> None:
    """Compile as the harness does, warm up five times on one query, then replay the captured
    graph on two different queries: zero ``cudagraph_skips``, and each replay equals eager on
    the same inputs (scores ``torch.equal``, ids up to ties)."""

    torch._dynamo.reset()
    skips_before = int(counters["inductor"]["cudagraph_skips"])
    compiled = torch.compile(module, mode="reduce-overhead", dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for _ in range(5):
            compiled(make_query(b, d), module.prepare_queries(make_query_attrs(b, c=C)))
        for seed in (21, 22):
            query, q_attrs = make_query(b, d, seed=seed), make_query_attrs(b, c=C, seed=seed)
            # cudagraph-tree outputs are reclaimed on the next call: clone before comparing.
            out_ids, out_scores = (
                t.clone() for t in compiled(query, module.prepare_queries(q_attrs))
            )
            eager_ids, eager_scores = module(query, module.prepare_queries(q_attrs))
            assert_topk_equal(out_ids, out_scores, eager_ids, eager_scores)
    skips = int(counters["inductor"]["cudagraph_skips"]) - skips_before
    assert skips == 0, (
        f"{label}: inductor skipped cudagraphs {skips}x — the compiled forward fell back to "
        f"compiled-eager, which is what the harness reports as NotCapturable"
    )


@pytest.mark.parametrize("algo", ["linr_v1", "linr_v2", "linr_v3"])
def test_compiled_captures_without_cudagraph_skips(algo):
    """Zero ``cudagraph_skips``, and the captured forward replays to eager on new inputs."""
    torch.manual_seed(0)
    _assert_replays_match_eager(_build(algo), B, algo)


def test_compiled_batch_of_one_v2():
    """``B == 1``: ``clause_compact``'s ``counts`` is a fresh tensor, not a view at element offset
    ``T - 1`` of its scan buffer, which inductor's ``assert_alignment`` on custom-op outputs
    rejects (``_host.compact_finish``). ``N = 512`` at ``block_n = 512`` is one tile."""
    torch.manual_seed(0)
    _assert_replays_match_eager(_build("linr_v2"), 1, "linr_v2, B=1")


def test_compiled_v2_wide_d():
    """V2 past D_PAD 256 runs ``fused_masked_knn_topk``'s SPLIT body at ``WIDE_CONFIG``
    (kernels.md § fused_masked_knn_topk, "Wide D"): it captures and replays like D = 64."""
    torch.manual_seed(0)
    _assert_replays_match_eager(_build("linr_v2", d=768), B, "linr_v2, D=768", d=768)
