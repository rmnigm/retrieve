"""META-FORK capture gate (plan §1.2, C2; ST-DLOOP's graph_launches.py for the fork arm). Per cell (filter
none / exact / bloom partial / bloom full x bs x k) of one backend's SilverTorch forward:

  static   one ``torch.cuda.graph`` capture on static query (+ prepared filter) buffers, then each of a pool
           of 16 query batches copied in and replayed: outputs ``torch.equal`` to that batch's eager forward
           (``static_equal``; the capture saw only batch 0). A prepared filter whose tensors change shape
           across the pool cannot be copied in (``static_equal: "shapes differ"``);
  harness  ``bench.measure.graph_callable`` (``torch.compile(mode="reduce-overhead", fullgraph=True)``, the
           harness ``graph`` mode): ``cudaGraphLaunch`` per call (``graph_launches``, must be 1), kernel
           launches outside the graph per call, and its outputs ``torch.equal`` to eager over the pool.

A capture error is recorded (its first line), not raised. Index as syncs.py.

    cd evaluation && PYTHONPATH=.:../retrieve/src:../retrieve python graph_launches.py BACKEND OUT.json \
        [--d 128] [--n 200000] [--bs 1,16,64] [--k 100,1000] [--score-path int32|fp16]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from bench.measure import NotCapturable, graph_callable
from tests.conftest import make_attrs, make_query_attrs

sys.path.insert(0, str(Path(__file__).parent))
from syncs import CELLS, DEV, LAUNCH_API, build  # noqa: E402

N_POOL = 16


def first_line(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).strip().splitlines()[0][:200] if str(e).strip() else ''}"


def tensors(prepared):
    return (
        [] if prepared is None else [t for t in prepared if isinstance(t, torch.Tensor)]
    )


def copy_prepared(static, src):
    for s, t in zip(tensors(static), tensors(src), strict=True):
        s.copy_(t)


def static_gate(m, pool, preps, eager):
    if any(
        [t.shape for t in tensors(p)] != [t.shape for t in tensors(preps[0])]
        for p in preps
    ):
        return "shapes differ"
    q = pool[0].clone()
    prep = None if preps[0] is None else type(preps[0])(*(t.clone() if isinstance(t, torch.Tensor) else t
                                                          for t in preps[0]))  # fmt: skip
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            m(q, prep)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = m(q, prep)
    ok = True
    for i in range(N_POOL):
        q.copy_(pool[i])
        if prep is not None:
            copy_prepared(prep, preps[i])
        g.replay()
        ok &= torch.equal(out[0], eager[i][0]) and torch.equal(out[1], eager[i][1])
    return ok


def harness_gate(m, pool, preps, eager):
    torch._dynamo.reset()
    c = graph_callable(m, pool[0], preps[0])
    with torch.inference_mode():
        outs = [c(pool[i], preps[i]) for i in range(N_POOL)]
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for i in range(N_POOL):
                c(pool[i], preps[i])
            torch.cuda.synchronize()
    names = Counter(e.name for e in prof.events())
    return {
        "graph_launches": names["cudaGraphLaunch"] / N_POOL,
        "launches_outside_graph": sum(names[n] for n in LAUNCH_API) / N_POOL,
        "harness_equal": all(
            torch.equal(o[0], e[0]) and torch.equal(o[1], e[1])
            for o, e in zip(outs, eager, strict=True)
        ),
    }


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
    for mode, path in CELLS:
        m = build(a.backend, mode, path, items, attrs, a.score_path)
        for bs in map(int, a.bs.split(",")):
            pool = torch.randn(N_POOL, bs, a.d, device=DEV, generator=g)
            preps = [None if mode == "none" else m.prepare_queries(make_query_attrs(bs, c=2, seed=10 + i))
                     for i in range(N_POOL)]  # fmt: skip
            for k in map(int, a.k.split(",")):
                m.k = k
                with torch.inference_mode():
                    eager = [m(pool[i], preps[i]) for i in range(N_POOL)]
                row = {"backend": a.backend, "mode": mode, "bloom_path": path, "bs": bs, "k": k, "d": a.d,
                       "n": a.n, "score_path": a.score_path}  # fmt: skip
                try:
                    with torch.inference_mode():
                        row["static_equal"] = static_gate(m, pool, preps, eager)
                except Exception as e:  # noqa: BLE001  (a failed capture is the gate's finding)
                    row["static_equal"] = first_line(e)
                try:
                    row |= harness_gate(m, pool, preps, eager)
                except (NotCapturable, Exception) as e:  # noqa: BLE001
                    row["harness"] = first_line(e)
                rows.append(row)
                print(row, flush=True)
        del m
        torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
