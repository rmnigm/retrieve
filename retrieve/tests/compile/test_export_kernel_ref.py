"""``torch.export`` preserves the ``@triton_op`` kernel reference (Invariant 2 regression).

``torch.library.triton_op`` requires the ``wrap_triton(_kernel)[grid](...)`` launch to appear
textually inline in the decorated body — the tracing machinery walks the decorated function's
source to capture the kernel. The K2 host-wrapper dedup (shared ``_cpse_prep``/``_cpse_finish``
around a single inline launch line) must not break that: if a refactor accidentally moves the
launch behind a helper or the dataclass/kwargs indirection confuses tracing, export either
raises or produces a graph with no path back to the Triton kernel.

This test exports a tiny module whose forward calls the public
``retrieve::codesigned_probe_score_exact`` op and asserts two things:

1. **Graph-level**: the exported graph still contains a node referencing the kernel — either
   the opaque custom-op call (``torch.ops.retrieve.codesigned_probe_score_exact.default``) or,
   if export decomposed the op, a ``higher_order.triton_kernel_wrapper_*`` node (those HOPs
   hold the kernel via a process-global side table, so their presence *is* a live kernel
   reference; the forward calls no other triton op, so any such node is ours).
2. **Executable**: re-running the exported module reproduces the eager output bit-for-bit —
   the strongest possible proof that the preserved reference actually reaches the same kernel
   with the same config (the kernel is deterministic: no atomics, fixed reduction order).

Lives under tests/compile/ until the export plan creates tests/export/.

The op-registry gates live here too, over every ``retrieve::`` op the Triton package registers:
each has a same-named twin in ``retrieve.ops.reference`` with the schema's argument names and
kinds, declares no mutable argument, and leaves every input bit-identical after a call (the op
and its twin); ``torch.library.opcheck`` runs on the two opaque ``custom_op``s with a
``register_fake``.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import retrieve.ops.triton  # noqa: F401  (registers torch.ops.retrieve.*)
from retrieve.indexing.quantize import quantize_int8_global, quantize_oporp_1bit
from retrieve.ops import reference
from retrieve.ops.triton.codesigned_probe_score_exact import (
    codesigned_probe_score_exact,
)
from tests.conftest import make_attrs, make_index, make_query, make_query_attrs
from tests.parity.conftest import make_bloom, make_exact, make_probe_family


class _ExactScorer(torch.nn.Module):
    """Minimal wrapper: index-side state as buffers, forward = one op call."""

    def __init__(self, lay, item_codes, item_clause_attrs, clause_is_reverse, global_scale, k):
        super().__init__()
        self.register_buffer("cluster_offsets", lay.cluster_offsets)
        self.register_buffer("item_codes", item_codes)
        self.register_buffer("sort_perm", lay.sort_perm)
        self.register_buffer("item_clause_attrs", item_clause_attrs)
        self.register_buffer("clause_is_reverse", clause_is_reverse)
        self.global_scale = global_scale
        self.k = k
        self.width = lay.width

    def forward(self, query, probe_ids, query_clause_attrs):
        return codesigned_probe_score_exact(
            query,
            probe_ids,
            self.cluster_offsets,
            self.item_codes,
            self.sort_perm,
            self.item_clause_attrs,
            self.clause_is_reverse,
            query_clause_attrs,
            self.global_scale,
            self.k,
            self.width,
        )


def _references_kernel(node) -> bool:
    """True if a graph node keeps the codesigned-exact kernel reachable."""
    if node.op != "call_function":
        return False
    if node.target is torch.ops.retrieve.codesigned_probe_score_exact.default:
        return True
    # Decomposed form: triton_kernel_wrapper_mutation / _functional HOPs reference the kernel
    # by index into torch._higher_order_ops.triton_kernel_wrap's kernel side table.
    return getattr(node.target, "__name__", "").startswith("triton_kernel_wrapper")


def test_export_preserves_kernel_reference():
    torch.manual_seed(0)
    d, b, c, a_max, k = 32, 2, 2, 2, 4
    lay = make_probe_family(b, 8, 12, 3)  # includes empty clusters and a tail past the items
    n = lay.n

    g = torch.Generator(device="cuda").manual_seed(0)
    item_codes = torch.randint(-127, 128, (n, d), generator=g, dtype=torch.int8, device="cuda")
    mod = _ExactScorer(
        lay,
        item_codes,
        make_attrs(n, c=c, a_max=a_max),
        torch.zeros(c, dtype=torch.bool, device="cuda"),
        global_scale=0.02,
        k=k,
    )
    query = make_query(b, d)
    q_attrs = make_query_attrs(b, c=c)
    args = (query, lay.probe_ids, q_attrs)

    eager_ids, eager_scores = mod(*args)

    ep = torch.export.export(mod, args)

    kernel_refs = [node for node in ep.graph.nodes if _references_kernel(node)]
    assert kernel_refs, (
        f"exported graph lost the codesigned_probe_score_exact kernel "
        f"reference — no custom-op or triton HOP node found in:\n{ep.graph}"
    )

    # Round-trip execution: the exported module must launch the same kernel with the same
    # DEFAULT_CONFIG, so ids and scores are bit-identical to eager.
    out_ids, out_scores = ep.module()(*args)
    torch.testing.assert_close(out_ids, eager_ids, rtol=0, atol=0)
    torch.testing.assert_close(out_scores, eager_scores, rtol=0, atol=0)


_SCHEMA_KIND = {"Tensor": "Tensor", "float": "float", "SymInt": "int", "int": "int"}


def _retrieve_schemas() -> dict[str, torch.FunctionSchema]:
    return {
        s.name.removeprefix("retrieve::"): s
        for s in torch._C._jit_get_all_schemas()
        if s.name.startswith("retrieve::")
    }


def _op_args() -> dict[str, tuple]:
    """One small valid call per registered op, keyed by op name."""
    b, d, k = 3, 64, 8
    lay = make_probe_family(b, 16, 64, 4)
    n = lay.n
    embs = make_index(n, d)
    query = make_query(b, d)
    codes, gs = quantize_int8_global(embs)
    qpos, bt, sigs, qb = make_bloom(n, b)
    attrs, rev, q_attrs = make_exact(n, b, reverse="mixed")
    probe = (lay.probe_ids, lay.cluster_offsets, codes, lay.sort_perm)
    g = torch.Generator(device="cuda").manual_seed(0)
    pos = torch.randint(0, n, (b, 64), generator=g, device="cuda")
    counts = torch.tensor([64, 10, 0], device="cuda")
    item_bits = quantize_oporp_1bit(embs, seed=0)[0]
    q_bits = quantize_oporp_1bit(query, seed=0)[0]
    return {
        "bloom_match": (qb, sigs),
        "bloom_compact": (qb, sigs),
        "clause_mask": (attrs, rev, q_attrs),
        "clause_compact": (attrs, rev, q_attrs),
        "codesigned_probe_score": (query, *probe, gs, k, lay.width),
        "codesigned_probe_score_bloom": (query, *probe, qpos, bt, gs, k, lay.width),
        "codesigned_probe_score_exact": (query, *probe, attrs, rev, q_attrs, gs, k, lay.width),
        "fused_masked_knn_topk": (query, embs, pos, counts, k),
        "oporp_1bit_match_topk_full": (q_bits, item_bits, k),
        "oporp_1bit_match_topk_indirect": (q_bits, item_bits, k, pos, counts),
    }


def test_every_op_has_a_same_signature_reference_twin():
    schemas = _retrieve_schemas()
    assert len(schemas) >= 10, sorted(schemas)
    assert sorted(schemas) == sorted(_op_args()), "an op has no input builder here"
    for name, schema in schemas.items():
        twin = inspect.signature(getattr(reference, name))
        got = [(p.name, p.annotation) for p in twin.parameters.values()]
        want = [(a.name, _SCHEMA_KIND[str(a.type)]) for a in schema.arguments]
        assert got == want, f"{name}: reference twin {got} vs op schema {want}"


def test_no_op_declares_a_mutable_argument():
    schemas = _retrieve_schemas()
    assert len(schemas) >= 10
    for name, schema in schemas.items():
        assert schema.is_mutable is False, name
        assert all(a.alias_info is None for a in schema.arguments), name


@pytest.mark.parametrize("name", sorted(_retrieve_schemas()))
def test_inputs_unchanged_after_the_op_and_its_twin(name):
    args = _op_args()[name]
    before = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
    for fn in (getattr(torch.ops.retrieve, name), getattr(reference, name)):
        fn(*args)
        torch.cuda.synchronize()
        for i, (a, b) in enumerate(zip(args, before, strict=True)):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b), f"{name} ({fn}): input {i} was written"


@pytest.mark.parametrize("name", ["clause_compact", "bloom_compact"])
def test_opcheck_on_the_fake_impls(name):
    torch.library.opcheck(getattr(torch.ops.retrieve, name).default, _op_args()[name])
