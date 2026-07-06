"""One tiny invocation per tuner benchmark body, so schema drift between ``tune.py`` and the
kernel ``_impl`` signatures breaks CI instead of a tuning session (the
``codesigned-probe-score`` subcommand rotted exactly this way when Stage 2b moved kwargs off the
public op — see Phase K1 in docs/plans/kernels-layers-design.md)."""

from __future__ import annotations

import pytest
import torch

from retrieve import tune

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_tune_bodies_run_one_point(monkeypatch):
    # Shrink every sweep to a single cheap point: one (block, num_warps) per config grid and one
    # bucket / P value per regime axis. The regime-driven tuners take their shapes as arguments,
    # so those are passed small below instead of patched.
    monkeypatch.setattr(tune, "_FMKT_GRID", [(32, 4)])
    monkeypatch.setattr(tune, "_OPORP_GRID", [(64, 4)])
    monkeypatch.setattr(tune, "_CPS_GRID", [(32, 4)])
    monkeypatch.setattr(tune, "_CLAUSE_MASK_GRID", [(128, 2)])
    monkeypatch.setattr(tune, "_CLAUSE_COMPACT_GRID", [(128, 2)])
    monkeypatch.setattr(tune, "_BLOOM_COMPACT_GRID", [(128, 2)])
    # Bucket/P lists live in tune's namespace (_P_BUCKETS/_N_BUCKETS are imported from the kernel
    # modules; _DEFAULT_CPS_P_GRID is module-local).
    monkeypatch.setattr(tune, "_P_BUCKETS", (256,))
    monkeypatch.setattr(tune, "_N_BUCKETS", (4096,))
    monkeypatch.setattr(tune, "_DEFAULT_CPS_P_GRID", (1024,))

    # One real call instead of do_bench's rep/warmup loop: this test checks call schemas, not
    # timings, and both the warmup loop and the _bench lambda still exercise each _impl.
    def _bench_once(fn) -> float:
        fn()
        torch.cuda.synchronize()
        return 0.0

    monkeypatch.setattr(tune, "_bench", _bench_once)

    dev = torch.device("cuda:0")
    results = [
        tune._tune_fmkt(dev, d=64, b=2),
        tune._tune_oporp(dev, w=2, b=2),
        tune._tune_cps(dev, d=64, b=2, w=2),
        tune._tune_clause_mask(dev, regimes=((4096, 2, 2, 2),)),
        tune._tune_clause_compact(dev, regimes=((4096, 2, 2, 2),)),
        tune._tune_bloom_compact(dev, regimes=((4096, 2, 4),)),
    ]
    for result in results:
        default = result["default"]
        assert len(default) == 2
        assert all(isinstance(v, int) for v in default)
