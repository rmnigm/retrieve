"""OFFICIAL-REWORK M3: the official backend's forward per path (none / exact / bloom partial / bloom full), before
(staging 5618bb6 = pre-M1, package `retrieve_pre`, attrs into forward, plans parsed per call) and after (this tree: filter
prepared outside, M2's epilogue and plan dedup), on one shared index per path, fp16 score path (the timed default),
n_probe 32, k 100, bs {1, 16}. Per arm, one torch.profiler session of CALLS calls after a warm-up:
  device us per call, split into Meta's kernels (names in `st::`), our Triton kernels (`_..._kernel`) and torch / cub
  (our glue: gathers, top-k, casts); host wall ms per call (no profiler, synced); kernel launches, host syncs
  (cudaStreamSynchronize / cudaDeviceSynchronize) and memcpys per call; peak CUDA memory of one call above its inputs.
With `capture` (its own process: a failed capture can leave the context unusable): a CUDA-graph capture of the after
arm's forward per path, recording what refuses and the error.

    cd evaluation && PYTHONPATH=<pkgs>:.:../retrieve/src:../retrieve python m3_prof.py {goodreads|arxiv} out.json [capture]
"""

import importlib
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml
from bench import config, inputs
from torch.profiler import ProfilerActivity, profile

DEV = torch.device("cuda")
SWEEP = {"goodreads": "c0_genre", "arxiv": "c0_maincat"}
ST = {
    a: importlib.import_module(f"{p}.modules.silvertorch")
    for a, p in (("before", "retrieve_pre"), ("after", "retrieve"))
}
CALLS, N_POOL = 50, 16
LAUNCH = (
    "cudaLaunchKernel",
    "cuLaunchKernel",
    "cudaLaunchKernelExC",
    "cuLaunchKernelEx",
)
SYNC = ("cudaStreamSynchronize", "cudaDeviceSynchronize")
COPY = ("cudaMemcpy", "cudaMemcpyAsync")

ds, out = sys.argv[1], Path(sys.argv[2])
cfg_path = Path(f"config/{ds}.yaml")
inp = inputs.load_inputs(config.load_dataset(cfg_path, 128), DEV, with_filters=True)
clauses = tuple(yaml.safe_load(cfg_path.read_text())["filters"]["clause"][SWEEP[ds]])
qa_s, skip = inputs.sweep_qa(inp["qa"], clauses)
item_embs, attrs = inp["item_embs"].to(DEV), inp["item_attrs"].to(DEV)
PATHS = [("none", None), ("exact", None), ("bloom", "partial"), ("bloom", "full")]


def build(arm, mode, path):
    kw = {
        "k": 100,
        "n_lists": 1024,
        "n_probe": 32,
        "n_iter": 10,
        "seed": 0,
        "backend": "official",
    }
    oc = {"score_path": "fp16", **({"bloom_path": path} if path else {})}
    if arm == "before":
        oc["cache_plans"] = False  # the campaign's timed setting before M1
    kw["official"] = ST[arm].OfficialConfig(**oc)
    if mode != "none":
        kw["filter_mode"] = mode
    if mode == "bloom":
        kw["k_hash"] = 5
    m = ST[arm].SilverTorch(**kw)
    m.register_index(item_embs, *(() if mode == "none" else (attrs,)))
    return m


def kind(name):
    if "st::" in name:
        return "meta"
    if name.startswith("_") and name.endswith("_kernel"):
        return "ours_triton"
    return "torch"


def profile_arm(call):
    for i in range(10):
        call(i % N_POOL)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(CALLS):
        call(i % N_POOL)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / CALLS * 1e3
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(CALLS):
            call(i % N_POOL)
        torch.cuda.synchronize()
    dev = {"meta": 0.0, "ours_triton": 0.0, "torch": 0.0}
    api = {"launches": 0, "syncs": 0, "memcpys": 0}
    for e in prof.key_averages():
        if (
            e.device_type == torch.autograd.DeviceType.CUDA
            and e.self_device_time_total > 0
        ):
            dev[kind(e.key)] += e.self_device_time_total / CALLS
        elif e.key in LAUNCH:
            api["launches"] += e.count / CALLS
        elif e.key in SYNC:
            api["syncs"] += e.count / CALLS
        elif e.key in COPY:
            api["memcpys"] += e.count / CALLS
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    call(0)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    return {
        "wall_ms": wall,
        "device_us": dev,
        "device_us_total": sum(dev.values()),
        **api,
        "peak_mib": peak,
    }


rows = []
if len(sys.argv) > 3 and sys.argv[3] == "capture":
    pool, qa_pool = inputs.query_pool(
        inp, qa_s, skip, bs=16, seed=0, n_pool=2, device=DEV
    )
    only = (
        sys.argv[4] if len(sys.argv) > 4 else None
    )  # "mode-path": one capture per process
    with torch.inference_mode():
        for mode, path in [x for x in PATHS if only in (None, f"{x[0]}-{x[1]}")]:
            m = build("after", mode, path)
            prep = None if mode == "none" else m.prepare_queries(qa_pool[0])
            m(pool[0], prep)
            torch.cuda.synchronize()
            try:
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
                    m(pool[0], prep)
                result = "ok"
            except Exception as exc:  # noqa: BLE001 — what refuses capture is the record
                result = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
                frames = traceback.extract_tb(exc.__traceback__)
                result += " | at " + " < ".join(f.name for f in reversed(frames[-5:]))
            rows.append(
                {
                    "dataset": ds,
                    "mode": mode,
                    "bloom_path": path,
                    "graph_capture": result,
                }
            )
            print(rows[-1], flush=True)
    out.write_text(json.dumps(rows, indent=1))
    sys.exit(0)
with torch.inference_mode():
    for bs in (16, 1):
        pool, qa_pool = inputs.query_pool(
            inp, qa_s, skip, bs=bs, seed=0, n_pool=N_POOL, device=DEV
        )
        for mode, path in PATHS:
            after, before = build("after", mode, path), build("before", mode, path)
            before.load_state_dict(after.state_dict())
            prep = [
                None if mode == "none" else after.prepare_queries(qa_pool[i])
                for i in range(N_POOL)
            ]
            row = {"dataset": ds, "bs": bs, "mode": mode, "bloom_path": path,
                   "before": profile_arm(lambda i: before(pool[i], None if mode == "none" else qa_pool[i])),
                   "after": profile_arm(lambda i: after(pool[i], prep[i]))}  # fmt: skip
            torch.cuda.synchronize()
            rows.append(row)
            b, a = row["before"], row["after"]
            print(f"{ds} bs={bs} {mode:5} {path or '':7} wall {b['wall_ms']:.3f}->{a['wall_ms']:.3f} ms  device "
                  f"{b['device_us_total']:.0f}->{a['device_us_total']:.0f} us (meta {a['device_us']['meta']:.0f}, "
                  f"triton {a['device_us']['ours_triton']:.0f}, torch {a['device_us']['torch']:.0f})  launches "
                  f"{b['launches']:.0f}->{a['launches']:.0f}  syncs {b['syncs']:.0f}->{a['syncs']:.0f}  memcpy "
                  f"{b['memcpys']:.0f}->{a['memcpys']:.0f}  peak {b['peak_mib']:.1f}->{a['peak_mib']:.1f} MiB",
                  flush=True)  # fmt: skip
out.write_text(json.dumps(rows, indent=1))
