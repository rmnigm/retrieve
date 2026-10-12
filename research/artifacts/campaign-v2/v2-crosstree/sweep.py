"""fused_masked_knn_topk `programs` sweep inside V2 (triton, clause, k 100, seed 0), one tree per process
(PYTHONPATH as in crosstree.py). The v2.1 tree runs programs {864, ..., 13824} and `full` (programs >=
cdiv(P, BLOCK_N) * B: one tile per program, the old grid's shape) by replacing the op module's
DEFAULT_CONFIG, which the public op reads per call; the old tree runs its own kernel as `old`. Per (cell,
bs): eager (config set before each call) and a CUDA graph per variant, captured by hand with the config
in force (`graph_callable` would let dynamo reuse one capture across variants of one module), all graphs
in one shared pool; each mode timed round-robin over the variants by `measure.latency_group` (end to
end), then one sentinel-bracketed profiler session over 20 graph replays per variant (kernel level).

    python sweep.py CELL OUT_JSON      # CELL: pm | ax | axs"""

import dataclasses
import importlib
import json
import sys
import time
from pathlib import Path

import bench
import retrieve
import torch
from bench import inputs, measure, run
from bench.config import load_matrix
from torch.profiler import ProfilerActivity, profile

CELLS = {  # dataset, suite, sweep, latency_group overrides
    "pm": ("pubmed", "filter", "c3_journal_reverse", {"n_min": 20, "target_s": 1.5}),
    "ax": ("arxiv", "filter", "c3_nversions", {}),
    "axs": ("arxiv-synth", "synth", "p1", {}),
}
PROGRAMS = (864, 1728, 3456, 6912, 13824)
K, SEED, CALLS, PADS = 100, 0, 20, (0.0, 0.01, 0.1, 1.0, 5.0)
DEV = torch.device("cuda")


def kernel_us(fn) -> tuple[dict, float]:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    for pad in PADS:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            time.sleep(pad)
            torch.cuda._sleep(1)
            for _ in range(CALLS):
                fn()
            torch.cuda._sleep(1)
            torch.cuda.synchronize()
            time.sleep(pad)
        dev = [
            e
            for e in p.key_averages()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        if sum(e.count for e in dev if "spin_kernel" in e.key) == 2:
            return {
                e.key: e.self_device_time_total / CALLS
                for e in dev
                if "spin_kernel" not in e.key
            }, pad
    raise RuntimeError("sentinels missing at every pad")


@torch.inference_mode()
def main(cell: str, out: Path) -> None:
    # the module, not the same-named op the package re-exports: the op reads DEFAULT_CONFIG from it
    fm = importlib.import_module("retrieve.ops.triton.fused_masked_knn_topk")

    ds, suite, sweep, kw = CELLS[cell]
    new = hasattr(fm.DEFAULT_CONFIG, "programs")
    head = {"bench": bench.__file__, "retrieve": retrieve.__file__, "code_version": measure.code_version(),
            "cell": cell, "default_config": str(fm.DEFAULT_CONFIG)}  # fmt: skip
    print(json.dumps(head), flush=True)
    measure.warm_gpu_once()
    jobs = load_matrix(
        Path(f"config/{ds}.yaml"), Path("config/suites.yaml"), suite, algos=["linr_v2"]
    )
    (job,) = [
        j
        for j in jobs
        if (j.backend, j.filter_kind, j.sweep, j.seed)
        == ("triton", "clause", sweep, SEED)
    ]
    inp = inputs.load_inputs(job.data, DEV, with_filters=True)
    measure.setup(SEED)
    assets = run.sweep_assets(job, inp, max(job.ks), DEV)
    m = run.build_module(job, inp, assets, max(job.ks), dict(job.build))
    m.k = K
    default = fm.DEFAULT_CONFIG
    res = {"pass_rate": assets["pass_rate"], "n_items": inp["n_items"], "rows": []}
    for bs in (1, 16):
        pool, qa = inputs.query_pool(
            inp, assets["qa_s"], assets["skip"], bs=bs, seed=SEED, device=DEV
        )
        tiles = -(-inp["n_items"] // default.block_n) if new else None
        variants = (
            {str(p): p for p in PROGRAMS} | {"full": tiles * bs}
            if new
            else {"old": None}
        )
        cfgs = {
            n: dataclasses.replace(default, programs=p) if new else default
            for n, p in variants.items()
        }
        it = iter(range(1 << 62))

        def eager(cfg):
            def call():
                fm.DEFAULT_CONFIG = cfg
                i = next(it) % pool.shape[0]
                return m(pool[i], qa[i])

            return call

        mempool, sq, sqa = (
            torch.cuda.graph_pool_handle(),
            pool[0].clone(),
            qa[0].clone(),
        )
        graphs = {}
        for name, cfg in cfgs.items():
            fm.DEFAULT_CONFIG = cfg
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    m(sq, sqa)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=mempool):
                m(sq, sqa)
            graphs[name] = g

        def replay(g):
            def call():
                i = next(it) % pool.shape[0]
                sq.copy_(pool[i])
                sqa.copy_(qa[i])
                g.replay()

            return call

        e_t = measure.latency_group(
            [eager(c) for c in cfgs.values()], bs=bs, mode="eager", **kw
        )
        fm.DEFAULT_CONFIG = default
        g_t = measure.latency_group(
            [replay(g) for g in graphs.values()], bs=bs, mode="graph", **kw
        )
        for name, (de, _), (dg, _) in zip(cfgs, e_t, g_t, strict=True):
            kern, pad = kernel_us(replay(graphs[name]))
            fmkt = sum(v for k, v in kern.items() if "fused_masked_knn_topk" in k)
            row = {"bs": bs, "variant": name, "programs": variants[name],
                   "eager_ms": de["median_ms"], "eager_sm_mhz": de["window_sm_mhz"],
                   "graph_ms": dg["median_ms"], "graph_sm_mhz": dg["window_sm_mhz"],
                   "unstable": [de["unstable"], dg["unstable"]],
                   "fmkt_us": fmkt, "device_us": sum(kern.values()), "pad_s": pad,
                   "top": sorted(kern.items(), key=lambda kv: -kv[1])[:6]}  # fmt: skip
            res["rows"].append(row)
            print(cell, bs, name, f"eager {de['median_ms']:.3f} graph {dg['median_ms']:.3f} ms",
                  f"fmkt {fmkt:.1f} us", dg["window_sm_mhz"], flush=True)  # fmt: skip
        del graphs
        torch.cuda.empty_cache()
    out.write_text(json.dumps({**head, **res}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
