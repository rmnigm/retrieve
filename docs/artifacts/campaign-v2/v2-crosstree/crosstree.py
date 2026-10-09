"""V2 cross-tree check (surprise gate, pod b: V2 +45 % at p ~ 1 on PubMed 10 M at v2.1): V1 and V2 triton
built and timed by one tree's own harness and library, imported through PYTHONPATH, one process per
(tree, group, repeat). Run from <tree>/evaluation:

    PYTHONPATH=<tree>/evaluation:<tree>/retrieve/src python crosstree.py GROUP OUT_JSON [--profile]

GROUP gs: goodreads-synth clause p1 and p0001; pm: pubmed d768 clause c3_journal_reverse (pass 0.9993).
Per sweep, bs {1, 16} x mode {eager, graph}: V1 and V2 timed round-robin by `measure.latency_group`
(k 100, seed 0). --profile instead runs one torch.profiler session over 20 V2 graph replays at bs 16
per sweep and records the device kernels."""

import json
import sys
from pathlib import Path

import bench
import retrieve
import torch
from bench import inputs, measure, run
from bench.config import load_matrix
from torch.profiler import ProfilerActivity, profile

GROUPS = {
    "gs": ("goodreads-synth", "synth", ["p1", "p0001"], {}),
    "pm": ("pubmed", "filter", ["c3_journal_reverse"], {"n_min": 20, "target_s": 1.5}),
}
ALGOS = ["linr_v1_filter_mask", "linr_v2"]
K, SEED, BSS = 100, 0, (1, 16)
DEV = torch.device("cuda")


def jobs_for(ds: str, suite: str, sweep: str):
    jobs = load_matrix(
        Path(f"config/{ds}.yaml"), Path("config/suites.yaml"), suite, algos=ALGOS
    )
    out = {}
    for j in jobs:
        if (j.backend, j.filter_kind, j.sweep, j.seed) == (
            "triton",
            "clause",
            sweep,
            SEED,
        ) and not j.build:
            assert j.algo not in out, j.algo
            out[j.algo] = j
    assert set(out) == set(ALGOS), out
    return [out[a] for a in ALGOS]


@torch.inference_mode()
def main(group: str, out: Path, prof: bool) -> None:
    ds, suite, sweeps, kw = GROUPS[group]
    head = {
        "bench": bench.__file__,
        "retrieve": retrieve.__file__,
        "code_version": measure.code_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "group": group,
        "profile": prof,
    }
    print(json.dumps(head), flush=True)
    measure.warm_gpu_once()
    res, inp = {}, None
    for sweep in sweeps:
        jobs = jobs_for(ds, suite, sweep)
        inp = inp or inputs.load_inputs(jobs[0].data, DEV, with_filters=True)
        measure.setup(SEED)
        assets = run.sweep_assets(jobs[0], inp, max(jobs[0].ks), DEV)
        mods = []
        for j in jobs:
            m = run.build_module(j, inp, assets, max(j.ks), dict(j.build))
            m.k = K
            mods.append(m)
        for bs in (16,) if prof else BSS:
            pool, qa = inputs.query_pool(
                inp, assets["qa_s"], assets["skip"], bs=bs, seed=SEED, device=DEV
            )
            for mode in ("graph",) if prof else ("eager", "graph"):
                torch._dynamo.reset()
                callees = (
                    mods
                    if mode == "eager"
                    else [measure.graph_callable(m, pool[0], qa[0]) for m in mods]
                )
                fns = [run._rotate(c, pool, qa) for c in callees]
                if prof:
                    fn = fns[1]
                    for _ in range(10):
                        fn()
                    torch.cuda.synchronize()
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                    ) as p:
                        for _ in range(20):
                            fn()
                        torch.cuda.synchronize()
                    dev = [
                        e
                        for e in p.key_averages()
                        if e.device_type == torch.autograd.DeviceType.CUDA
                    ]
                    res[f"{sweep}/bs{bs}/{mode}/linr_v2"] = sorted(
                        (
                            {
                                "kernel": e.key,
                                "us_per_call": e.self_device_time_total / 20,
                                "calls": e.count / 20,
                            }
                            for e in dev
                        ),
                        key=lambda d: -d["us_per_call"],
                    )
                    continue
                timed = measure.latency_group(fns, bs=bs, mode=mode, **kw)
                for algo, (d, _) in zip(ALGOS, timed, strict=True):
                    res[f"{sweep}/bs{bs}/{mode}/{algo}"] = {
                        k: d.get(k)
                        for k in (
                            "median_ms",
                            "window_medians_ms",
                            "window_sm_mhz",
                            "spread",
                            "unstable",
                            "n",
                        )
                    }
                    print(
                        sweep,
                        bs,
                        mode,
                        algo,
                        round(d["median_ms"], 4),
                        d["window_sm_mhz"],
                        flush=True,
                    )
        del mods, assets
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    out.write_text(
        json.dumps({**head, "pass_rate_sweeps": sweeps, "results": res}, indent=1)
    )


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), "--profile" in sys.argv[3:])
