"""V-PROF3: one-cell torch.profiler comparisons, cells built exactly as `bench run` builds them.

    python prof3.py profile ITEM VARIANT OUT   # one profiler session, its own process
    python prof3.py time ITEM OUT              # both variants of ITEM in one process, interleaved

ITEM a: goodreads `codesign`, official bloom c0_genre, n_probe 32, bs 16, partial vs full.
ITEM b: goodreads-synth `synth`, V1 clause p01, bs 1, Triton vs torch.compile(max-autotune).
ITEM c0001 / c1: goodreads-synth `synth`, V1 Triton clause p0001 / p1, bs 16, eager vs graph.
k 100, seed 0 everywhere. `profile` writes OUT/<item>-<variant>/ (kernels.csv, api.csv, summary.json,
trace.json.gz); `time` writes OUT/<item>-timing.json."""

import csv
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from bench import inputs, measure, run
from bench.config import load_matrix
from torch.profiler import ProfilerActivity, profile

GR = ("goodreads", "codesign", "silvertorch")
SY = ("goodreads-synth", "synth", "linr_v1_filter_mask")
# item -> (dataset, suite, algo, filter_kind, sweep, bs, {variant: (backend, build match, query, mode)})
ITEMS = {
    "a": (
        *GR,
        "bloom",
        "c0_genre",
        16,
        {
            "partial": (
                "official",
                {"bloom_path": "partial"},
                {"n_probe": 32},
                "eager",
            ),
            "full": ("official", {"bloom_path": "full"}, {"n_probe": 32}, "eager"),
        },
    ),
    "b": (
        *SY,
        "clause",
        "p01",
        1,
        {
            "triton": ("triton", {}, {}, "eager"),
            "compile": ("torch", {"compile": "max-autotune"}, {}, "eager"),
        },
    ),
    **{
        f"c{p[1:]}": (
            *SY,
            "clause",
            p,
            16,
            {
                "eager": ("triton", {}, {}, "eager"),
                "graph": ("triton", {}, {}, "graph"),
            },
        )
        for p in ("p0001", "p1")
    },
}
K, SEED, CALLS = 100, 0, 20
DEV = torch.device("cuda")


def job_for(item: str, variant: str):
    ds, suite, algo, fk, sweep, _, variants = ITEMS[item]
    backend, build, query, _ = variants[variant]
    jobs = load_matrix(
        Path(f"config/{ds}.yaml"), Path("config/suites.yaml"), suite, algos=[algo]
    )
    hits = [
        j
        for j in jobs
        if j.backend == backend
        and j.filter_kind == fk
        and j.sweep == sweep
        and j.seed == SEED
        and K in j.ks
        and all(j.build.get(a) == v for a, v in build.items())
        and ("compile" in build or "compile" not in j.build)
        and query in j.query
    ]
    assert len(hits) == 1, (item, variant, len(hits))
    return hits[0], {**hits[0].build, **query}


def callee(item: str, variant: str, inp=None):
    """(zero-arg rotating call, mode, inp) for one variant, as `run.perf` builds it."""
    job, params = job_for(item, variant)
    bs, mode = ITEMS[item][5], ITEMS[item][6][variant][3]
    measure.setup(job.seed)
    inp = inp or inputs.load_inputs(job.data, DEV, with_filters=True)
    assets = run.sweep_assets(job, inp, max(job.ks), DEV)
    module = run.build_module(
        job, inp, assets, max(job.ks), run.resolve_pool(params, assets)
    )
    if "compile" in job.build:
        run.compile_warmup(module, inp, assets, DEV)
    if module.backend == "official":
        import dataclasses

        module.official = dataclasses.replace(module.official, cache_plans=False)
    module.k = K
    pool, qa_pool = inputs.query_pool(
        inp, assets["qa_s"], assets["skip"], bs=bs, seed=job.seed, device=DEV
    )
    c = module
    if mode == "graph":
        torch._dynamo.reset()
        c = measure.graph_callable(
            module, *((pool[0],) if qa_pool is None else (pool[0], qa_pool[0]))
        )
    return run._rotate(c, pool, qa_pool), mode, inp


@torch.inference_mode()
def do_profile(item: str, variant: str, out: Path) -> None:
    fn, mode, _ = callee(item, variant)
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak_mib = (torch.cuda.max_memory_allocated() - m0) / 2**20
    for pad in measure.PROFILE_PADS_S:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            time.sleep(pad)
            torch.cuda._sleep(1)
            for _ in range(CALLS):
                fn()
            torch.cuda._sleep(1)
            torch.cuda.synchronize()
            time.sleep(pad)
        ev = prof.key_averages()
        dev = [e for e in ev if e.device_type == torch.autograd.DeviceType.CUDA]
        if sum(e.count for e in dev if measure.SENTINEL in e.key) == 2:
            break
    else:
        raise RuntimeError(f"{item}-{variant}: sentinels missing at pad {pad} s")
    d = out / f"{item}-{variant}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    kern = sorted(
        (e for e in dev if measure.SENTINEL not in e.key),
        key=lambda e: -e.self_device_time_total,
    )
    with open(d / "kernels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "calls_per_call", "us_per_call", "share"])
        tot = sum(e.self_device_time_total for e in kern)
        for e in kern:
            w.writerow(
                [
                    e.key,
                    e.count / CALLS,
                    e.self_device_time_total / CALLS,
                    e.self_device_time_total / tot,
                ]
            )
    api = sorted(
        (
            e
            for e in ev
            if e.device_type == torch.autograd.DeviceType.CPU and e.key.startswith("cu")
        ),
        key=lambda e: -e.count,
    )
    with open(d / "api.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["api", "calls_per_call", "cpu_us_per_call"])
        for e in api:
            w.writerow([e.key, e.count / CALLS, e.self_cpu_time_total / CALLS])
    launches = sum(e.count for e in api if "LaunchKernel" in e.key) / CALLS
    summary = {
        "item": item,
        "variant": variant,
        "mode": mode,
        "calls": CALLS,
        "pad_s": pad,
        "kernels_per_call": sum(e.count for e in kern) / CALLS,
        "distinct_kernels": len(kern),
        "device_us_per_call": tot / CALLS,
        "kernel_launch_api_per_call": launches,
        "graph_launch_per_call": sum(e.count for e in api if "GraphLaunch" in e.key)
        / CALLS,
        "memcpy_per_call": sum(e.count for e in api if "Memcpy" in e.key) / CALLS,
        "malloc_per_call": sum(e.count for e in api if "Malloc" in e.key) / CALLS,
        "sync_per_call": sum(e.count for e in api if "Synchronize" in e.key) / CALLS,
        "peak_scratch_mib": peak_mib,
        "code_version": measure.code_version(),
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    prof.export_chrome_trace(str(d / "trace.json"))
    with (
        open(d / "trace.json", "rb") as src,
        gzip.open(d / "trace.json.gz", "wb") as dst,
    ):
        shutil.copyfileobj(src, dst)
    (d / "trace.json").unlink()
    print(json.dumps(summary))


@torch.inference_mode()
def do_time(item: str, out: Path) -> None:
    measure.warm_gpu_once()
    bs = ITEMS[item][5]
    fns, inp = {}, None
    for v in ITEMS[item][6]:
        fns[v], mode, inp = callee(item, v, inp)
        fns[v] = (fns[v], mode)
    res = {}
    if len({m for _, m in fns.values()}) == 1:
        timed = measure.latency_group([f for f, _ in fns.values()], bs=bs, mode=mode)
        res = {v: d for v, (d, _) in zip(fns, timed, strict=True)}
    else:  # eager vs graph: latency_group takes one mode, so alternate whole windows A, B, A, B, A, B
        for _ in range(3):
            for v, (f, m) in fns.items():
                ((d, _),) = measure.latency_group([f], bs=bs, mode=m, windows=1)
                res.setdefault(v, []).append(d)
    p = out / f"{item}-timing.json"
    p.write_text(
        json.dumps(
            {
                "item": item,
                "bs": bs,
                "k": K,
                "code_version": measure.code_version(),
                "arms": res,
            },
            indent=1,
            default=str,
        )
    )
    print(p)


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "profile":
        do_profile(a[0], a[1], Path(a[2]))
    else:
        do_time(a[0], Path(a[1]))
