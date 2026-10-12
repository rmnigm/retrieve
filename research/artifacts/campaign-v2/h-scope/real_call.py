"""H-SCOPE on a real eager call: goodreads `filter` c0_genre bloom, SilverTorch triton and official (n_probe 24, seed 0,
bs 16), built by the harness, one `measure.profile_once` each. Checks the scopes add up to kernels_us / kernels_calls and
that `scorer` is non-zero for both arms. Prints each arm's scopes. Exits 1 on a failed check.

    python real_call.py         (cwd evaluation/ of a checkout with H-SCOPE)"""

import sys
from pathlib import Path

import torch
from bench import inputs, measure, run
from bench.config import load_matrix

dev = torch.device("cuda")
jobs = load_matrix(
    Path("config/goodreads.yaml"),
    Path("config/suites.yaml"),
    "filter",
    algos=["silvertorch"],
)
picked = [
    j for j in jobs
    if j.filter_kind == "bloom" and j.sweep == "c0_genre" and j.seed == 0 and j.backend in ("triton", "official")
]  # fmt: skip
inp = inputs.load_inputs(picked[0].data, dev, with_filters=True)
bad = 0
for job in picked:
    params = {**job.build, "n_probe": 24}
    assets = run.sweep_assets(job, inp, max(job.ks), dev)
    module = run.build_module(job, inp, assets, max(job.ks), params)
    module.k = 100
    pool, qa = inputs.query_pool(
        inp, assets["qa_s"], assets["skip"], bs=16, seed=0, device=dev
    )
    with torch.inference_mode():
        out = measure.profile_once(lambda m=module: m(pool[0], qa[0]))
    sc = out["kernel_scopes"]
    total_us = sum(v["us"] for v in sc.values())
    total_calls = sum(v["calls"] for v in sc.values())
    ok = (
        abs(total_us - out["kernels_us"]) < 1e-6 * max(1.0, out["kernels_us"])
        and total_calls == out["kernels_calls"]
        and sc["scorer"]["us"] > 0
    )
    bad += not ok
    print(job.backend, "OK" if ok else "FAIL", f"kernels_us {out['kernels_us']:.1f} calls {out['kernels_calls']}",
          {k: (round(v["us"], 1), v["calls"]) for k, v in sc.items()})  # fmt: skip
sys.exit(1 if bad else 0)
