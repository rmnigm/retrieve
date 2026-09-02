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

The CUDA C++ backend's exact op, and its CuTe DSL twin, are parametrized in
alongside it. Nothing about ``@triton_op`` source-walking applies there — they are
ordinary ``@torch.library.custom_op``s — but the *other* half of this test does:
export must keep each as one opaque node and the exported module must reproduce
eager bit for bit. That closes the "``torch.export`` is not a gate for the cuda
ops" item the handoff §1 recorded as deferred.

Lives under tests/compile/ until the export plan creates tests/export/.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.silvertorch.codesigned_probe_score_cuda import (
    codesigned_probe_score_exact_cuda,
)
from retrieve.kernels.silvertorch.codesigned_probe_score_cute import (
    codesigned_probe_score_exact_cute,
)
from retrieve.kernels.silvertorch.codesigned_probe_score_exact import (
    codesigned_probe_score_exact,
)
from tests.conftest import (
    make_attrs,
    make_query,
    make_query_attrs,
    require_cps_cuda,
    require_cps_cute,
)


class _ExactScorer(torch.nn.Module):
    """Minimal wrapper: index-side state as buffers, forward = one op call.

    ``max_size`` is only meaningful on the cuda / cute backends (the clause-mask kernel
    needs the padded cluster width to lay its mask out cluster-major); the Triton op
    derives nothing from it, so it is ignored there."""

    def __init__(
        self,
        item_codes: torch.Tensor,
        item_clause_attrs: torch.Tensor,
        clause_is_reverse: torch.Tensor,
        global_scale: float,
        k: int,
        backend: str = "triton",
        max_size: int = 0,
    ):
        super().__init__()
        self.register_buffer("item_codes", item_codes)
        self.register_buffer("item_clause_attrs", item_clause_attrs)
        self.register_buffer("clause_is_reverse", clause_is_reverse)
        self.global_scale = global_scale
        self.k = k
        self.backend = backend
        self.max_size = max_size

    def forward(self, query, flat_probed_items, query_clause_attrs):
        if self.backend in ("cuda", "cute"):
            op = {
                "cuda": codesigned_probe_score_exact_cuda,
                "cute": codesigned_probe_score_exact_cute,
            }[self.backend]
            return op(
                query,
                flat_probed_items,
                self.item_codes,
                self.item_clause_attrs,
                self.clause_is_reverse,
                query_clause_attrs,
                self.global_scale,
                self.k,
                self.max_size,
            )
        return codesigned_probe_score_exact(
            query,
            flat_probed_items,
            self.item_codes,
            self.item_clause_attrs,
            self.clause_is_reverse,
            query_clause_attrs,
            self.global_scale,
            self.k,
        )


def _references_kernel(node, backend: str) -> bool:
    """True if a graph node keeps the codesigned-exact kernel reachable."""
    if node.op != "call_function":
        return False
    if backend == "cuda":
        # A CUDA C++ custom op has no decomposed form to fall back on: export either
        # preserves the opaque node or the reference is gone.
        return node.target is torch.ops.retrieve.codesigned_probe_score_exact_cuda.default
    if backend == "cute":
        return node.target is torch.ops.retrieve.codesigned_probe_score_exact_cute.default
    if node.target is torch.ops.retrieve.codesigned_probe_score_exact.default:
        return True
    # Decomposed form: triton_kernel_wrapper_mutation / _functional HOPs reference the kernel
    # by index into torch._higher_order_ops.triton_kernel_wrap's kernel side table.
    return getattr(node.target, "__name__", "").startswith("triton_kernel_wrapper")


@pytest.mark.parametrize("backend", ["triton", "cuda", "cute"])
def test_export_preserves_kernel_reference(backend):
    if backend == "cuda":
        require_cps_cuda()
    elif backend == "cute":
        require_cps_cute()
    torch.manual_seed(0)
    # P = n_probe * max_size: the cuda / cute op rebuilds the cluster-span mask layout
    # from max_size, so P must stay a whole multiple of it.
    n, d, b, p, c, a_max, k, max_size = 64, 32, 2, 16, 2, 2, 4, 8

    g = torch.Generator(device="cuda").manual_seed(0)
    item_codes = torch.randint(-127, 128, (n, d), generator=g, dtype=torch.int8, device="cuda")
    flat_probed_items = torch.randint(0, n, (b, p), generator=g, device="cuda")
    flat_probed_items[:, -2:] = -1  # exercise the -1-pad lanes
    mod = _ExactScorer(
        item_codes,
        make_attrs(n, c=c, a_max=a_max),
        torch.zeros(c, dtype=torch.bool, device="cuda"),
        global_scale=0.02,
        k=k,
        backend=backend,
        max_size=max_size,
    )
    query = make_query(b, d)
    q_attrs = make_query_attrs(b, c=c)
    args = (query, flat_probed_items, q_attrs)

    eager_ids, eager_scores = mod(*args)

    ep = torch.export.export(mod, args)

    kernel_refs = [node for node in ep.graph.nodes if _references_kernel(node, backend)]
    assert kernel_refs, (
        f"exported graph lost the codesigned_probe_score_exact ({backend}) kernel "
        f"reference — no custom-op or triton HOP node found in:\n{ep.graph}"
    )

    # Round-trip execution: the exported module must launch the same kernel with the same
    # DEFAULT_CONFIG, so ids and scores are bit-identical to eager.
    out_ids, out_scores = ep.module()(*args)
    torch.testing.assert_close(out_ids, eager_ids, rtol=0, atol=0)
    torch.testing.assert_close(out_scores, eager_scores, rtol=0, atol=0)
