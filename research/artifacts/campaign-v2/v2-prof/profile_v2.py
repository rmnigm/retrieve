"""V1 vs V2 at bs {1, 16}, p {0.001, 0.01, 1.0} on goodreads-synth d128: graph-captured like the
harness, timed in interleaved windows, then torch.profiler per-kernel tables (name, calls, us,
grid). Plus a component split at bs 16 (each op alone, CUDA-graphed) to locate the floor.

    cd evaluation && CUDA_VISIBLE_DEVICES=0 uv run python \
        ../docs/artifacts/campaign-v2/v2-prof/profile_v2.py /scratch/v2-prof/profile.json
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from bench import algos, inputs
from bench.config import load_dataset
from bench.measure import clocks, graph_callable, setup, warm_gpu_once
from retrieve.ops.triton.clause_compact import clause_compact
from retrieve.ops.triton.clause_mask import clause_mask
from retrieve.ops.triton.fused_masked_knn_topk import fused_masked_knn_topk

K = 100
SWEEPS = {"p0001": 0, "p001": 2, "p1": 6}
BATCHES = (1, 16)
PAIRS, CALLS, PROF_CALLS = 5, 300, 20


def timed(fn, n: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(n):
        fn(i)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / n


def kernels(fn, n: int) -> list[dict]:
    """Device kernels of ``n`` calls from the chrome trace: per name, calls per call of ``fn``,
    us per call of ``fn``, and the launch grid / block."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(n):
            fn(i)
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        events = json.load(open(f.name))["traceEvents"]
    agg: dict[str, dict] = defaultdict(lambda: {"calls": 0, "us": 0.0, "grid": None, "block": None})
    for e in events:
        if e.get("cat") != "kernel":
            continue
        a = agg[e["name"]]
        a["calls"] += 1
        a["us"] += e["dur"]
        a["grid"], a["block"] = e["args"].get("grid"), e["args"].get("block")
    rows = [
        {"kernel": k, "calls": v["calls"] / n, "us": v["us"] / n, "grid": v["grid"], "block": v["block"]}
        for k, v in agg.items()
    ]
    return sorted(rows, key=lambda r: -r["us"])


def graphed(fn):
    """Capture ``fn()`` (no host sync) once; return a replay closure."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return lambda _i: g.replay()


def main(out: str) -> None:
    dev = torch.device("cuda")
    setup(0)
    warm_gpu_once()
    ds = load_dataset(Path("config/goodreads-synth.yaml"), 128)
    inp = inputs.load_inputs(ds, dev)
    embs, n = inp["item_embs"], inp["n_items"]
    result: dict = {"n_items": n, "cells": [], "components": []}

    for fk in ("clause", "bloom"):
        filt = algos.build_filter(
            fk, inp["item_attrs"], clause_is_reverse=inp["clause_is_reverse"], backend="triton"
        )
        mods = {
            "v1": algos.build("linr_v1_filter_mask", embs, k=K, backend="triton", filter_kind=fk, filter_mod=filt),
            "v2": algos.build("linr_v2", embs, k=K, backend="triton", filter_kind=fk, filter_mod=filt),
        }
        for bs in BATCHES:
            torch._dynamo.reset()
            for sweep, clause in SWEEPS.items():
                qa_s, skip = inputs.sweep_qa(inp["qa"], (clause,))
                pool, qa_pool = inputs.query_pool(inp, qa_s, skip, bs=bs, seed=0, n_pool=64, device=dev)
                calls = {
                    arm: (lambda c: lambda i: c(pool[i % 64], qa_pool[i % 64]))(
                        graph_callable(m, pool[0], qa_pool[0])
                    )
                    for arm, m in mods.items()
                }
                windows: dict[str, list] = {a: [] for a in calls}
                mhz = []
                with torch.inference_mode():
                    for _ in range(PAIRS):
                        for arm, fn in calls.items():
                            windows[arm].append(timed(fn, CALLS))
                            mhz.append(clocks()["sm_mhz"])
                    for arm, fn in calls.items():
                        ks = kernels(fn, PROF_CALLS)
                        med = sorted(windows[arm])[PAIRS // 2]
                        result["cells"].append(
                            {"filter": fk, "sweep": sweep, "bs": bs, "arm": arm, "ms": med,
                             "windows_ms": windows[arm], "sm_mhz": mhz, "kernels": ks}
                        )  # fmt: skip
                        print(f"{fk:6} {sweep:6} bs{bs:<3} {arm} {med:.3f} ms  sm {min(mhz)}-{max(mhz)}")
                        for r in ks:
                            print(f"    {r['us']:8.1f} us  x{r['calls']:.0f}  grid {r['grid']}  {r['kernel'][:90]}")

    # Component split, clause filter, bs 16: each op alone, CUDA-graphed.
    filt = algos.build_filter("clause", inp["item_attrs"], clause_is_reverse=inp["clause_is_reverse"], backend="triton")
    e16 = embs.to(torch.float16)
    e16t = e16.t().contiguous()
    mods_v2 = algos.build("linr_v2", embs, k=K, backend="triton", filter_kind="clause", filter_mod=filt)
    for sweep, clause in SWEEPS.items():
        qa_s, skip = inputs.sweep_qa(inp["qa"], (clause,))
        pool, qa_pool = inputs.query_pool(inp, qa_s, skip, bs=16, seed=0, n_pool=1, device=dev)
        q, qa = pool[0].to(torch.float16), qa_pool[0]
        attrs, rev = filt.item_clause_attrs, filt.clause_is_reverse
        with torch.inference_mode():
            cand, counts = clause_compact(attrs, rev, qa)
            scores_full = torch.mm(q, e16t, out_dtype=torch.float32)
            maxc = int(counts.max())
            width = max(K, 1 << (maxc - 1).bit_length())
            narrow = cand[:, :width].contiguous()
            comps = {
                "clause_mask [B,N] bool": lambda: clause_mask(attrs, rev, qa),
                "clause_compact (pred+scan+scatter)": lambda: clause_compact(attrs, rev, qa),
                "mm fp16->fp32 [B,N]": lambda: torch.mm(q, e16t, out_dtype=torch.float32),
                "topk [B,N] fp32": lambda: torch.topk(scores_full, K, dim=1),
                f"topk [B,{width}] fp32": lambda: torch.topk(scores_full[:, :width].contiguous(), K, dim=1),
                "fmkt op at P=N (as V2)": lambda: fused_masked_knn_topk(q, e16, cand, counts, K),
                f"fmkt op at P={width} (narrowed)": lambda: fused_masked_knn_topk(q, e16, narrow, counts, K),
                "V2 forward (eager)": lambda: mods_v2(q, qa),
            }  # fmt: skip
            ref = fused_masked_knn_topk(q, e16, cand, counts, K)
            nar = fused_masked_knn_topk(q, e16, narrow, counts, K)
            same = bool(torch.equal(ref[0], nar[0]) and torch.equal(ref[1], nar[1]))
            for name, fn in comps.items():
                g = graphed(fn)
                ms = sorted(timed(g, CALLS) for _ in range(3))[1]
                ks = kernels(g, PROF_CALLS)
                result["components"].append(
                    {"sweep": sweep, "max_count": maxc, "component": name, "ms": ms,
                     "sm_mhz": clocks()["sm_mhz"], "kernels": ks, "narrow_equal": same}
                )  # fmt: skip
                print(f"[{sweep} max_count {maxc}] {name:40} {ms * 1e3:8.1f} us  narrow==full {same}")
                for r in ks:
                    print(f"    {r['us']:8.1f} us  x{r['calls']:.0f}  grid {r['grid']}  {r['kernel'][:90]}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
