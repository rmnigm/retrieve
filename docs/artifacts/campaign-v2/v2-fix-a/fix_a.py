"""V2 Fix A: old (tag campaign-v2) vs new fused_masked_knn_topk in one process.

Phases (argv[2], comma-separated; default all):
  gate    torch.equal old vs new on ids and scores: goodreads-synth 7 rates + goodreads c0_genre,
          clause + bloom, bs {1, 16}, k {100, 1000}, op level on the same compaction, plus
          edge counts (0, 1, 31, 32, 33, k-1, k, N) and the bucketed _impl; V2 end to end too
  v3prof  V3 stage 1's indirect op alone (old / new at P = N, new narrowed), bs {1, 16},
          p {0.001, 0.01, 1}
  sweep   fmkt alone, graphed, at programs {864, ..., 13824}, bs {1, 16}, p {0.001, 0.01, 1}
  abab    V2 and V3 old vs new graph-captured (harness graph_callable), interleaved windows,
          sm_mhz per window, kernel tables; clause + bloom, bs {1, 16}, p {0.001, 0.01, 1}

    cd evaluation && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v2-fix-a \
        flock /scratch/gpu0.lock uv run python ../docs/artifacts/campaign-v2/v2-fix-a/fix_a.py \
        /scratch/v2-fix-a/out.json [gate,v3prof,sweep,abab]

Resume-safe: rerunning the same command into the same file skips the phases already written.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.profiler import ProfilerActivity, profile

import retrieve.interfaces as rif
import retrieve.ops.triton.fused_masked_knn_topk as fm
import retrieve.ops.triton.oporp_1bit_match_topk as om
from bench import algos, inputs
from bench.config import load_dataset
from bench.measure import clocks, graph_callable, setup, warm_gpu_once
from retrieve.ops.triton.clause_compact import clause_compact

SYNTH = ("p0001", "p0003", "p001", "p003", "p01", "p03", "p1")
TIMED = ("p0001", "p001", "p1")
PROGRAMS = (864, 1728, 3456, 6912, 13824)
PAIRS, CALLS, PROF_CALLS = 7, 300, 20
T_975 = {6: 2.447}  # two-sided 95 % t quantile, df = PAIRS - 1


def _load_old_file(name: str, ops: tuple[str, ...]):
    """``campaign-v2``'s ``ops/triton/<name>.py`` with its op and kernel names suffixed ``_old``."""
    src = subprocess.check_output(["git", "show", f"campaign-v2:retrieve/src/retrieve/ops/triton/{name}.py"], text=True)
    for op in ops:
        src = src.replace(f'"retrieve::{op}"', f'"retrieve::{op}_old"')
    src = src.replace(f"_{name}_kernel", f"_{name}_old_kernel")
    path = Path(f"/scratch/v2-fix-a/{name}_old.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"{name}_old", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_old"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_old():
    """Both campaign-v2 kernel files, and a ``triton_old`` backend namespace over their ops."""
    fo = _load_old_file("fused_masked_knn_topk", ("fused_masked_knn_topk",))
    oo = _load_old_file("oporp_1bit_match_topk", ("oporp_1bit_match_topk_full", "oporp_1bit_match_topk_indirect"))
    rif._OPS_LOADED["triton_old"] = SimpleNamespace(
        fused_masked_knn_topk=fo.fused_masked_knn_topk,
        oporp_1bit_match_topk_full=oo.oporp_1bit_match_topk_full,
        oporp_1bit_match_topk_indirect=oo.oporp_1bit_match_topk_indirect,
    )
    return SimpleNamespace(fm=fo, op=oo)


def build_pair(algo: str, inp, fk, f, k):
    """(old, new) modules of one algo over the same filter; old routes to ``triton_old``."""
    params = {"candidate_pool": 5000} if algo == "linr_v3" else None
    mods = [algos.build(algo, inp["item_embs"], k=k, backend="triton", filter_kind=fk, filter_mod=f, params=params)
            for _ in range(2)]  # fmt: skip
    for sub in ("idx", "stage1", "stage2"):
        if hasattr(mods[0], sub):
            getattr(mods[0], sub).backend = "triton_old"
    return mods


def timed(fn, n: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(n):
        fn(i)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / n


def kernels(fn, n: int) -> list[dict]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(n):
            fn(i)
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        events = json.load(open(f.name))["traceEvents"]
    agg: dict[str, dict] = defaultdict(lambda: {"calls": 0, "us": 0.0, "grid": None})
    for e in events:
        if e.get("cat") != "kernel":
            continue
        a = agg[e["name"]]
        a["calls"] += 1
        a["us"] += e["dur"]
        a["grid"] = e["args"].get("grid")
    rows = [{"kernel": k, "calls": v["calls"] / n, "us": v["us"] / n, "grid": v["grid"]} for k, v in agg.items()]
    return sorted(rows, key=lambda r: -r["us"])


def graphed(fn):
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


def same(a, b) -> bool:
    return bool(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))


def ci(ratios: list[float]) -> dict:
    logs = [math.log(r) for r in ratios]
    m = sum(logs) / len(logs)
    sd = math.sqrt(sum((x - m) ** 2 for x in logs) / (len(logs) - 1))
    h = T_975[len(logs) - 1] * sd / math.sqrt(len(logs))
    return {"ratio": math.exp(m), "lo": math.exp(m - h), "hi": math.exp(m + h)}


def spread(ws: list[float]) -> float:
    s = sorted(ws)
    return (s[-1] - s[0]) / s[len(s) // 2]


def data(name: str, dev):
    ds = load_dataset(Path(f"config/{name}.yaml"), 128)
    return ds, inputs.load_inputs(ds, dev)


def filt(fk, inp):
    return algos.build_filter(fk, inp["item_attrs"], clause_is_reverse=inp["clause_is_reverse"], backend="triton")


def gate(old, dev, out):
    res = out.setdefault("gate", {"cells": [], "edges": [], "e2e": []})
    for name, sweeps in (("goodreads-synth", SYNTH), ("goodreads", ("c0_genre",))):
        ds, inp = data(name, dev)
        e16 = inp["item_embs"].to(torch.float16)
        n = inp["n_items"]
        v3 = algos.build("linr_v3", inp["item_embs"], k=100, backend="triton", params={"candidate_pool": 5000})
        for fk in ("clause", "bloom"):
            f = filt(fk, inp)
            for sweep in sweeps:
                qa_s, skip = inputs.sweep_qa(inp["qa"], ds.clauses[fk][sweep])
                for bs in (1, 16):
                    n_pool = 64 if bs == 1 else 16
                    pool, qa_pool = inputs.query_pool(inp, qa_s, skip, bs=bs, seed=0, n_pool=n_pool, device=dev)
                    for op, k in (("fmkt", 100), ("fmkt", 1000), ("oporp", 5000)):
                        ok, zero, full, rows = True, 0, 0, 0
                        with torch.inference_mode():
                            for i in range(n_pool):
                                cand, counts = f.evaluate_indices(qa_pool[i])
                                if op == "fmkt":
                                    q = pool[i].to(torch.float16)
                                    a = old.fm.fused_masked_knn_topk(q, e16, cand, counts, k)
                                    b = fm.fused_masked_knn_topk(q, e16, cand, counts, k)
                                else:
                                    qb = v3.stage1._project_query(pool[i])
                                    a = old.op.oporp_1bit_match_topk_indirect(qb, v3.stage1.item_bits, k, cand, counts)
                                    b = om.oporp_1bit_match_topk_indirect(qb, v3.stage1.item_bits, k, cand, counts)
                                ok &= same(a, b)
                                zero += int((counts == 0).sum())
                                full += int((counts == n).sum())
                                rows += bs
                        res["cells"].append(
                            {"dataset": name, "filter": fk, "sweep": sweep, "bs": bs, "op": op, "k": k,
                             "equal": ok, "rows": rows, "rows_count0": zero, "rows_countN": full}
                        )  # fmt: skip
                        print(f"gate {name} {fk} {sweep} bs{bs} {op} k{k}: equal={ok} rows={rows} c0={zero} cN={full}")
            # End to end, old vs new modules on the same batches (one sweep per filter).
            for algo in ("linr_v2", "linr_v3"):
                for k in (100, 1000):
                    m_old, m_new = build_pair(algo, inp, fk, f, k)
                    sweep = sweeps[min(2, len(sweeps) - 1)]
                    qa_s, skip = inputs.sweep_qa(inp["qa"], ds.clauses[fk][sweep])
                    pool, qa_pool = inputs.query_pool(inp, qa_s, skip, bs=16, seed=0, n_pool=8, device=dev)
                    with torch.inference_mode():
                        ok = all(same(m_old(pool[i], qa_pool[i]), m_new(pool[i], qa_pool[i])) for i in range(8))
                    res["e2e"].append({"dataset": name, "algo": algo, "filter": fk, "sweep": sweep, "k": k, "equal": ok})
                    print(f"e2e {name} {algo} {fk} {sweep} k{k}: equal={ok}")
        if name == "goodreads-synth":
            # Edge counts on the p 1 compaction (every row has count N), rewritten per row.
            f = filt("clause", inp)
            qa_s, skip = inputs.sweep_qa(inp["qa"], ds.clauses["clause"]["p1"])
            pool, qa_pool = inputs.query_pool(inp, qa_s, skip, bs=16, seed=0, n_pool=1, device=dev)
            with torch.inference_mode():
                cand, _ = f.evaluate_indices(qa_pool[0])
                q = pool[0].to(torch.float16)
                for k in (100, 1000):
                    edge = torch.tensor(
                        [0, 1, 31, 32, 33, k - 1, k, n, 0, n, n - 1, 5000, 64, 0, 1 << 17, n], device=dev
                    )
                    narrow = cand[:, :3000].contiguous()
                    en = edge.clamp_max(3000)
                    r = {"op": "fmkt", "k": k, "counts": edge.tolist(),
                         "op_equal": same(old.fm.fused_masked_knn_topk(q, e16, cand, edge, k), fm.fused_masked_knn_topk(q, e16, cand, edge, k)),
                         "impl_equal": same(old.fm._fused_masked_knn_topk_impl(q, e16, cand, edge, k), fm._fused_masked_knn_topk_impl(q, e16, cand, edge, k)),
                         "impl_p3000_equal": same(old.fm._fused_masked_knn_topk_impl(q, e16, narrow, en, k), fm._fused_masked_knn_topk_impl(q, e16, narrow, en, k))}  # fmt: skip
                    res["edges"].append(r)
                    print(f"edges {r}")
                qb, bits = v3.stage1._project_query(pool[0]), v3.stage1.item_bits
                for k in (100, 5000):
                    edge = torch.tensor(
                        [0, 1, 511, 512, 513, k - 1, k, n, 0, n, n - 1, 5000, 64, 0, 1 << 17, n], device=dev
                    )
                    narrow = cand[:, :3000].contiguous()
                    en = edge.clamp_max(3000)
                    r = {"op": "oporp", "k": k, "counts": edge.tolist(),
                         "op_equal": same(old.op.oporp_1bit_match_topk_indirect(qb, bits, k, cand, edge), om.oporp_1bit_match_topk_indirect(qb, bits, k, cand, edge)),
                         "impl_equal": same(old.op._oporp_1bit_match_topk_impl(qb, bits, k, cand, edge), om._oporp_1bit_match_topk_impl(qb, bits, k, cand, edge)),
                         "impl_p3000_equal": same(old.op._oporp_1bit_match_topk_impl(qb, bits, k, narrow, en), om._oporp_1bit_match_topk_impl(qb, bits, k, narrow, en)),
                         "full_equal": same(old.op.oporp_1bit_match_topk_full(qb, bits, k), om.oporp_1bit_match_topk_full(qb, bits, k))}  # fmt: skip
                    res["edges"].append(r)
                    print(f"edges {r}")


def v3prof(old, dev, out):
    """V3 stage 1's indirect op alone, CUDA-graphed: old (bucketed, one tile per program) and new
    at P = N, and new narrowed to next_pow2(max count). The V3 forward tables are in ``abab``."""
    res = out.setdefault("v3prof", [])
    ds, inp = data("goodreads-synth", dev)
    f = filt("clause", inp)
    stage1 = algos.build("linr_v3", inp["item_embs"], k=100, backend="triton", params={"candidate_pool": 5000}).stage1
    for bs in (1, 16):
        for sw in TIMED:
            qa_s, skip = inputs.sweep_qa(inp["qa"], ds.clauses["clause"][sw])
            pool, qa_pool = inputs.query_pool(inp, qa_s, skip, bs=bs, seed=0, n_pool=1, device=dev)
            with torch.inference_mode():
                cand, counts = clause_compact(f.item_clause_attrs, f.clause_is_reverse, qa_pool[0])
                qb, bits = stage1._project_query(pool[0]), stage1.item_bits
                maxc = int(counts.max())
                width = max(5000, 1 << (maxc - 1).bit_length())
                narrow = cand[:, :width].contiguous()
                comps = {
                    "old at P=N": lambda: old.op.oporp_1bit_match_topk_indirect(qb, bits, 5000, cand, counts),
                    "new at P=N": lambda: om.oporp_1bit_match_topk_indirect(qb, bits, 5000, cand, counts),
                    f"new at P={width}": lambda: om.oporp_1bit_match_topk_indirect(qb, bits, 5000, narrow, counts),
                }  # fmt: skip
                for name, fn in comps.items():
                    g = graphed(fn)
                    ms = sorted(timed(g, CALLS) for _ in range(3))[1]
                    ks = kernels(g, PROF_CALLS)
                    res.append({"bs": bs, "sweep": sw, "max_count": maxc, "component": name, "ms": ms,
                                "sm_mhz": clocks()["sm_mhz"], "kernels": ks})  # fmt: skip
                    print(f"[v3 stage1 bs{bs} {sw} max_count {maxc}] {name:14} {ms * 1e3:8.1f} us")
                    for r in ks:
                        print(f"    {r['us']:8.1f} us  x{r['calls']:.0f}  grid {r['grid']}  {r['kernel'][:90]}")


def sweep(old, dev, out):
    res = out.setdefault("sweep", [])
    ds, inp = data("goodreads-synth", dev)
    f = filt("clause", inp)
    e16 = inp["item_embs"].to(torch.float16)
    for bs in (1, 16):
        for sw in TIMED:
            qa_s, skip = inputs.sweep_qa(inp["qa"], ds.clauses["clause"][sw])
            pool, qa_pool = inputs.query_pool(inp, qa_s, skip, bs=bs, seed=0, n_pool=1, device=dev)
            with torch.inference_mode():
                cand, counts = f.evaluate_indices(qa_pool[0])
                q = pool[0].to(torch.float16)
                ref = old.fm.fused_masked_knn_topk(q, e16, cand, counts, 100)

                def run(cfg):
                    launch = fm._fmkt_prep(q, e16, cand, counts, cfg, bucket=False)
                    fm._fused_masked_knn_topk_kernel[launch.grid](**launch.kwargs)
                    return fm._fmkt_finish(launch, 100, pad_to_k=False)

                arms = {"old": graphed(lambda: old.fm.fused_masked_knn_topk(q, e16, cand, counts, 100))}
                eq = {}
                for p in PROGRAMS:
                    cfg = dataclasses.replace(fm.DEFAULT_CONFIG, programs=p)
                    eq[p] = same(run(cfg), ref)
                    arms[p] = graphed(lambda cfg=cfg: run(cfg))
                ws: dict = {a: [] for a in arms}
                for _ in range(3):
                    for a, g in arms.items():
                        ws[a].append(timed(g, CALLS))
                row = {"bs": bs, "sweep": sw, "max_count": int(counts.max()), "sm_mhz": clocks()["sm_mhz"],
                       "ms": {str(a): sorted(v)[1] for a, v in ws.items()}, "equal": {str(p): v for p, v in eq.items()}}  # fmt: skip
                res.append(row)
                print(f"sweep bs{bs} {sw}: " + "  ".join(f"{a}={v * 1e3:.1f}us" for a, v in row["ms"].items()) + f"  eq={all(eq.values())}")


def abab(dev, out):
    res = out.setdefault("abab", [])
    ds, inp = data("goodreads-synth", dev)
    for algo, fk in (("linr_v2", "clause"), ("linr_v2", "bloom"), ("linr_v3", "clause"), ("linr_v3", "bloom")):
        f = filt(fk, inp)
        m_old, m_new = build_pair(algo, inp, fk, f, 100)
        for bs in (1, 16):
            torch._dynamo.reset()
            for sw in TIMED:
                qa_s, skip = inputs.sweep_qa(inp["qa"], ds.clauses[fk][sw])
                pool, qa_pool = inputs.query_pool(inp, qa_s, skip, bs=bs, seed=0, n_pool=64, device=dev)
                cs = {"old": graph_callable(m_old, pool[0], qa_pool[0]), "new": graph_callable(m_new, pool[0], qa_pool[0])}
                calls = {a: (lambda c: lambda i: c(pool[i % 64], qa_pool[i % 64]))(c) for a, c in cs.items()}
                ws: dict = {a: [] for a in calls}
                mhz: dict = {a: [] for a in calls}
                with torch.inference_mode():
                    eq = same(cs["old"](pool[1], qa_pool[1]), cs["new"](pool[1], qa_pool[1]))
                    for _ in range(PAIRS):
                        for a, fn in calls.items():
                            ws[a].append(timed(fn, CALLS))
                            mhz[a].append(clocks()["sm_mhz"])
                    ks = {a: kernels(fn, PROF_CALLS) for a, fn in calls.items()}
                c = ci([n / o for n, o in zip(ws["new"], ws["old"], strict=True)])
                row = {"algo": algo, "filter": fk, "bs": bs, "sweep": sw, "graph_equal": eq,
                       "old_ms": sorted(ws["old"])[PAIRS // 2], "new_ms": sorted(ws["new"])[PAIRS // 2],
                       "new_over_old": c, "windows_ms": ws, "sm_mhz": mhz,
                       "unstable": {a: spread(w) > 0.05 for a, w in ws.items()}, "kernels": ks}  # fmt: skip
                res.append(row)
                print(f"abab {algo} {fk} bs{bs} {sw}: old {row['old_ms']:.3f} new {row['new_ms']:.3f} ms  "
                      f"new/old {c['ratio']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]  eq={eq}  "
                      f"sm {min(mhz['old'] + mhz['new'])}-{max(mhz['old'] + mhz['new'])}  unstable={row['unstable']}")  # fmt: skip
                for a in ("old", "new"):
                    top = ks[a][0]
                    print(f"    {a}: {top['us']:.1f} us grid {top['grid']} {top['kernel'][:60]}")


def main(path: str, phases: str = "gate,v3prof,sweep,abab") -> None:
    dev = torch.device("cuda")
    setup(0)
    warm_gpu_once()
    old = load_old()
    out_path = Path(path)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out["env"] = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                  "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "-uno"], text=True).strip()),
                  "config": dataclasses.asdict(fm.DEFAULT_CONFIG)}  # fmt: skip
    done = out.setdefault("done", [])
    for ph in phases.split(","):
        if ph in done:  # resume: a rerun into the same file skips finished phases
            print(f"skip {ph}: done")
            continue
        out.pop(ph, None)
        {"gate": lambda: gate(old, dev, out), "v3prof": lambda: v3prof(old, dev, out),
         "sweep": lambda: sweep(old, dev, out), "abab": lambda: abab(dev, out)}[ph]()  # fmt: skip
        done.append(ph)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
