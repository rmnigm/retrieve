"""``torch.export`` preserves the ``@triton_op`` kernel reference, and the op-registry gates.

``torch.library.triton_op`` requires the ``wrap_triton(_kernel)[grid](...)`` launch to appear
textually inline in the decorated body: the tracing machinery walks the decorated function's
source to capture the kernel. The shared ``_prep`` / ``_finish`` helpers around that one launch
line must not break it: if a refactor moves the launch behind a helper, export either raises or
produces a graph with no path back to the kernel. The export test wraps ``fused_masked_knn_topk``
and asserts the exported graph still references the kernel (the custom-op node, or a
``triton_kernel_wrapper_*`` HOP if export decomposed the op) and that re-running the exported
module reproduces the eager output bit for bit (the kernel is deterministic: no atomics, a fixed
reduction order).

The op-registry gates run over every ``retrieve::`` op the Triton package registers: each has a
same-named twin in ``retrieve.ops.reference`` with the schema's argument names and kinds, declares
no mutable argument, and leaves every input bit-identical after a call (the op and its twin);
``torch.library.opcheck`` runs on the opaque ``custom_op`` with a ``register_fake``.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import retrieve.ops.triton  # noqa: F401  (registers torch.ops.retrieve.*)
from retrieve.functional import compact_mask
from retrieve.indexing.quantize import quantize_oporp_1bit
from retrieve.ops import reference
from retrieve.ops.triton.fused_masked_knn_topk import fused_masked_knn_topk
from tests.conftest import make_index, make_mask, make_query
from tests.parity.conftest import make_exact


class _Prefilter(torch.nn.Module):
    """Minimal wrapper: the item table as a buffer, forward = one op call."""

    def __init__(self, item_embs, k):
        super().__init__()
        self.register_buffer("item_embs", item_embs)
        self.k = k

    def forward(self, query, positive_indices, counts):
        return fused_masked_knn_topk(query, self.item_embs, positive_indices, counts, self.k)


def _references_kernel(node) -> bool:
    """True if a graph node keeps the kernel reachable."""
    if node.op != "call_function":
        return False
    if node.target is torch.ops.retrieve.fused_masked_knn_topk.default:
        return True
    # Decomposed form: triton_kernel_wrapper_mutation / _functional HOPs reference the kernel
    # by index into torch._higher_order_ops.triton_kernel_wrap's kernel side table.
    return getattr(node.target, "__name__", "").startswith("triton_kernel_wrapper")


def test_export_preserves_kernel_reference():
    n, d, b, k = 512, 32, 2, 4
    mod = _Prefilter(make_index(n, d).half(), k)
    query = make_query(b, d).half()
    pos, counts = compact_mask(make_mask(b, n, pass_rate=0.3))
    args = (query, pos, counts)

    eager_ids, eager_scores = mod(*args)

    ep = torch.export.export(mod, args)

    kernel_refs = [node for node in ep.graph.nodes if _references_kernel(node)]
    assert kernel_refs, (
        f"exported graph lost the fused_masked_knn_topk kernel reference: no custom-op or "
        f"triton HOP node found in:\n{ep.graph}"
    )

    out_ids, out_scores = ep.module()(*args)
    torch.testing.assert_close(out_ids, eager_ids, rtol=0, atol=0)
    torch.testing.assert_close(out_scores, eager_scores, rtol=0, atol=0)


_SCHEMA_KIND = {"Tensor": "Tensor", "float": "float", "SymInt": "int", "int": "int", "bool": "bool"}


def _retrieve_schemas() -> dict[str, torch.FunctionSchema]:
    return {
        s.name.removeprefix("retrieve::"): s
        for s in torch._C._jit_get_all_schemas()
        if s.name.startswith("retrieve::")
    }


def _op_args() -> dict[str, tuple]:
    """One small valid call per registered op, keyed by op name."""
    b, n, d, k = 3, 1024, 64, 8
    embs = make_index(n, d)
    query = make_query(b, d)

    attrs, rev, q_attrs = make_exact(n, b, reverse="mixed")
    g = torch.Generator(device="cuda").manual_seed(0)
    pos = torch.randint(0, n, (b, 64), generator=g, device="cuda")
    counts = torch.tensor([64, 10, 0], device="cuda")
    item_bits = quantize_oporp_1bit(embs, seed=0)[0]
    q_bits = quantize_oporp_1bit(query, seed=0)[0]
    scores = torch.randn(b, n, generator=g, device="cuda")
    return {
        "clause_mask": (attrs, rev, q_attrs),
        "clause_mask_packed": (attrs, rev, q_attrs),
        "clause_mask_scores": (scores, attrs, rev, q_attrs),
        "clause_compact": (attrs, rev, q_attrs),
        "fused_masked_knn_topk": (query, embs, pos, counts, k),
        "gemv_scores": (query[:1].half(), embs.half().t().contiguous()),
        "oporp_1bit_match_topk_full": (q_bits, item_bits, k),
        "oporp_1bit_match_topk_indirect": (q_bits, item_bits, k, pos, counts),
    }


def test_every_op_has_a_same_signature_reference_twin():
    schemas = _retrieve_schemas()
    assert len(schemas) >= 8, sorted(schemas)
    assert sorted(schemas) == sorted(_op_args()), "an op has no input builder here"
    for name, schema in schemas.items():
        twin = inspect.signature(getattr(reference, name))
        got = [(p.name, p.annotation) for p in twin.parameters.values()]
        want = [(a.name, _SCHEMA_KIND[str(a.type)]) for a in schema.arguments]
        assert got == want, f"{name}: reference twin {got} vs op schema {want}"


def test_no_op_declares_a_mutable_argument():
    schemas = _retrieve_schemas()
    assert len(schemas) >= 8
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


def test_opcheck_on_the_fake_impl():
    """Every opcheck utility on ``clause_compact``, ``test_aot_dispatch_dynamic``'s bitwise
    eager-vs-AOT comparison of the whole output included. The op writes nothing past
    ``counts[b]``, so the tail is whatever ``torch.empty`` held; PyTorch's deterministic mode
    fills uninitialised memory, which makes the full ``[B, N]`` buffer a function of the inputs
    and comparable bit for bit."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.library.opcheck(
            torch.ops.retrieve.clause_compact.default, _op_args()["clause_compact"]
        )
    finally:
        torch.use_deterministic_algorithms(prev)
