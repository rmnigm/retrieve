"""One tiny invocation per tuner benchmark body, so schema drift between ``tune.py`` and the
kernel ``_impl`` signatures breaks CI instead of a tuning session (the
``codesigned-probe-score`` subcommand rotted exactly this way when Stage 2b moved kwargs off the
public op — see Phases K1/K7 in docs/plans/kernels-layers-design.md)."""

from __future__ import annotations

import pytest
import torch

from retrieve import tune

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_registry_covers_all_kernels():
    assert tuple(spec.name for spec in tune.KERNELS) == (
        "fused-masked-knn-topk",
        "oporp-1bit-match-topk",
        "codesigned-probe-score",
        "codesigned-probe-score-cuda",
        "codesigned-probe-score-exact",
        "codesigned-probe-score-exact-cuda",
        "clause-mask",
        "clause-compact",
        "bloom-compact",
    )


@pytest.mark.parametrize("spec", tune.KERNELS, ids=lambda spec: spec.name)
def test_tune_bodies_run_one_point(spec):
    # One real _impl call per spec at its tiny smoke regime with the cheapest grid point — exactly
    # the call _sweep's warmup loop and _bench lambda make. Checks call schemas, not timings.
    if spec.name in ("codesigned-probe-score-cuda", "codesigned-probe-score-exact-cuda"):
        from tests.conftest import require_cps_cuda

        require_cps_cuda()
    dev = torch.device("cuda:0")
    cfg = spec.config_cls(*spec.grid[0])
    out = spec.run(spec.make_inputs(dev, spec.smoke_regime), cfg)
    torch.cuda.synchronize()
    tensors = out if isinstance(out, tuple) else (out,)
    assert all(isinstance(t, torch.Tensor) for t in tensors)
    assert all(t.device.type == "cuda" for t in tensors)
