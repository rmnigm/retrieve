"""META-FORK instrument (plan §1.1, C2): host syncs, copies and launches of one SilverTorch forward per
backend. Per cell (filter none / exact / bloom partial / bloom full x bs x k), the filter prepared
beforehand (``prepare_queries``):

  error_mode   the forward under ``torch.cuda.set_sync_debug_mode("error")``: ok, or the error's first line
               (catches aten syncs, ``.item()`` included, also inside a C++ op);
  per forward  over 10 eager forwards under ``torch.profiler`` (CPU + CUDA activity), divided by 10:
               ``syncs`` (cudaStreamSynchronize / cudaDeviceSynchronize / cudaEventSynchronize / cudaMemcpy),
               ``d2h`` (Memcpy DtoH activity), ``pageable_h2d`` (Memcpy HtoD from pageable memory),
               ``launches`` (cudaLaunchKernel / cuLaunchKernel(Ex)), ``kernels`` (device kernel activity).

The raw cudaStreamSynchronize of a C++ op never reaches the sync debug mode, so both instruments are
needed. Index: N Gaussian items (seed 0), n_lists 256, n_probe 24, two clauses (``make_attrs``), bloom
k_hash 5 (Triton m_bits 1024; the official backends' width is ``OfficialConfig.b_multiplier``).

    PYTHONPATH=retrieve/src:retrieve python syncs.py BACKEND OUT.json [--d 128] [--n 200000] \
        [--bs 1,16,64] [--k 100,1000] [--score-path int32|fp16]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from retrieve import OfficialConfig, SilverTorch
from retrieve.interfaces import OFFICIAL_BACKENDS
from tests.conftest import make_attrs, make_query_attrs

DEV = torch.device("cuda")
CALLS = 10
SYNC_API = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaMemcpy",
)
LAUNCH_API = ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx")
CELLS = (("none", None), ("exact", None), ("bloom", "partial"), ("bloom", "full"))


def build(backend, mode, path, items, attrs, score_path):
    official = backend in OFFICIAL_BACKENDS
    kw = {"filter_mode": mode}
    if mode == "bloom":
        kw |= {"k_hash": 5, "m_bits": None if official else 1024}
        if not official:
            kw["bloom_path"] = path
    cfg = (
        OfficialConfig(score_path=score_path, bloom_path=path or "partial")
        if official
        else None
    )
    m = SilverTorch(
        k=1000,
        n_lists=256,
        n_probe=24,
        n_iter=5,
        seed=0,
        backend=backend,
        official=cfg,
        **kw,
    )
    m.register_index(items, *(() if mode == "none" else (attrs,)))
    return m


def count(fn):
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(CALLS):
            fn()
        torch.cuda.synchronize()
    names = Counter()
    kernels = 0
    for e in prof.events():
        names[e.name] += 1
        if e.device_type == torch.autograd.DeviceType.CUDA and not e.name.startswith(
            ("Memcpy", "Memset")
        ):
            kernels += 1
    # the profiler's own trailing synchronize is one cudaDeviceSynchronize outside the forwards
    names["cudaDeviceSynchronize"] -= 1
    return {
        "syncs": sum(names[n] for n in SYNC_API) / CALLS,
        "d2h": sum(v for n, v in names.items() if n.startswith("Memcpy DtoH")) / CALLS,
        "pageable_h2d": sum(
            v for n, v in names.items() if n.startswith("Memcpy HtoD (Pageable")
        )
        / CALLS,
        "launches": sum(names[n] for n in LAUNCH_API) / CALLS,
        "kernels": kernels / CALLS,
    }


def error_mode(fn):
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
        return "ok"
    except RuntimeError as e:
        return str(e).strip().splitlines()[0][:200]
    finally:
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend")
    ap.add_argument("out")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--bs", default="1,16,64")
    ap.add_argument("--k", default="100,1000")
    ap.add_argument("--score-path", default="int32")
    a = ap.parse_args()
    g = torch.Generator(device=DEV).manual_seed(0)
    items = torch.randn(a.n, a.d, device=DEV, generator=g)
    attrs = make_attrs(a.n, c=2, a_max=2)
    rows = []
    with torch.inference_mode():
        for mode, path in CELLS:
            m = build(a.backend, mode, path, items, attrs, a.score_path)
            for bs in map(int, a.bs.split(",")):
                q = torch.randn(bs, a.d, device=DEV, generator=g)
                prepared = (
                    None
                    if mode == "none"
                    else m.prepare_queries(make_query_attrs(bs, c=2))
                )
                for k in map(int, a.k.split(",")):
                    m.k = k
                    fn = lambda m=m, q=q, p=prepared: m(q, p)  # noqa: E731
                    fn()
                    row = {"backend": a.backend, "mode": mode, "bloom_path": path, "bs": bs, "k": k,
                           "d": a.d, "n": a.n, "score_path": a.score_path,
                           "error_mode": error_mode(fn), **count(fn)}  # fmt: skip
                    rows.append(row)
                    print(row, flush=True)
            del m
            torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
