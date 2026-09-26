"""One tiny invocation per tuner benchmark body, so schema drift between ``tune.py`` and the
kernel ``_impl`` signatures breaks CI instead of a tuning session (the
``codesigned-probe-score`` subcommand rotted exactly this way when Stage 2b moved kwargs off the
public op — see Phases K1/K7 in docs/plans/kernels-layers-design.md)."""

from __future__ import annotations

import pytest
import torch

from retrieve.ops import tune

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_registry_covers_all_kernels():
    assert tuple(spec.name for spec in tune.KERNELS) == (
        "fused-masked-knn-topk",
        "oporp-1bit-match-topk",
        "codesigned-probe-score",
        "codesigned-probe-score-exact",
        "clause-mask",
        "clause-compact",
        "bloom-compact",
    )


@pytest.mark.parametrize("spec", tune.KERNELS, ids=lambda spec: spec.name)
def test_tune_bodies_run_one_point(spec):
    # One real _impl call per spec at its tiny smoke regime with the cheapest grid point — exactly
    # the call _sweep's warmup loop and _bench lambda make. Checks call schemas, not timings.
    dev = torch.device("cuda:0")
    cfg = spec.config_cls(*spec.grid[0])
    out = spec.run(spec.make_inputs(dev, spec.smoke_regime), cfg)
    torch.cuda.synchronize()
    tensors = out if isinstance(out, tuple) else (out,)
    assert all(isinstance(t, torch.Tensor) for t in tensors)
    assert all(t.device.type == "cuda" for t in tensors)


_CUR, _CAND = (256, 4), (128, 8)


@pytest.mark.parametrize(
    ("cand_ms", "expect"),
    [
        # 2 % faster everywhere: inside the noise band, the shipped default stays.
        ((0.98, 0.98), _CUR),
        # 10 % faster everywhere: taken.
        ((0.90, 0.90), _CAND),
        # 20 % faster on the geomean but 8 % slower in one regime: over the worst-regime cap.
        ((0.60, 1.08), _CUR),
        # 20 % faster, 4 % slower in one regime: inside the cap, taken.
        ((0.62, 1.04), _CAND),
    ],
)
def test_choose_keeps_current_inside_noise_and_caps_worst_regime(cand_ms, expect):
    timings = {f"B={i}": {_CUR: 1.0, _CAND: ms} for i, ms in enumerate(cand_ms)}
    pick, rule = tune._choose(_CUR, timings)
    assert pick == expect
    assert rule["worst_vs_current"]["128,8"] == pytest.approx(max(cand_ms))  # exact ratios, 1e-6
