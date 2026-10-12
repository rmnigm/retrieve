"""V-GRAPH-IDS fix gates, run with this worktree's library and harness on PYTHONPATH.

1. Eager `quantize_int8` codes and scales equal the 408b1188 function (`OLD`, verbatim) over every
   arXiv d128 query; the compiled new function equals it too; the compiled old one is counted.
2. Triton SilverTorch graph == eager (`torch.equal` ids and scores): the 4 H2H-FINAL entries
   (seed 1, bs 16, first IDS_PROBE_BATCHES pool batches, none + bloom, k 100 + 1000) and every
   kept arXiv query in bs 16 chunks (none, clause `c0_maincat`, bloom; k 100 + 1000). Two graph
   arms side by side: `new` (this library) and `old` (compiled with OLD patched into `_host`).
3. Interleaved latency old vs new (`measure.latency_group`, ROUNDS rounds) on the clause
   cell, k 100, eager + graph, bs 1 + 16: median and 95 % CI per arm, paired ratio new / old.

usage: fix_gates.py OUT_DIR
"""

import argparse
import copy
import itertools
import json
from pathlib import Path

import retrieve
import torch
import torch._functorch.config
import torch._inductor.config
from bench import config, inputs, measure, run, stats
from retrieve.indexing.quantize import quantize_int8
from retrieve.modules.silvertorch import SilverTorch
from retrieve.ops.triton import _host

WT = Path(__file__).resolve().parents[4]
CFG = WT / "evaluation" / "config"
COMPILE = dict(mode="reduce-overhead", dynamic=False, fullgraph=True)
ROUNDS = 10
BS = 16

# Each arm compiles its own graph from the quantize installed at its compile time.
torch._inductor.config.fx_graph_cache = False
torch._functorch.config.enable_autograd_cache = False


def OLD(embs):
    abs_max = embs.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = (abs_max / 127.0).squeeze(1)
    codes = (embs / abs_max * 127.0).round().clamp(-128, 127).to(torch.int8)
    return codes, scales


class Old(SilverTorch):
    def forward(self, *a):
        return super().forward(*a)


class New(SilverTorch):
    def forward(self, *a):
        return super().forward(*a)


def arm(module, cls, quant, example):
    m = copy.copy(module)
    m.__class__ = cls
    _host.quantize_int8 = quant
    try:
        return measure.graph_callable(m, *example)
    finally:
        _host.quantize_int8 = quantize_int8


def patched(module, quant):
    """Eager arm: ``module`` with ``quant`` installed in ``_host`` for each call (both arms pay
    the same attribute swap)."""

    def call(*x):
        _host.quantize_int8 = quant
        try:
            return module(*x)
        finally:
            _host.quantize_int8 = quantize_int8

    return call


def chunks(rows):
    out = [rows[i : i + BS] for i in range(0, rows.numel() - BS + 1, BS)]
    if rows.numel() % BS:
        out.append(rows[-BS:])
    return out


@torch.inference_mode()
def compare(module, callees, batches):
    """Rows of ``batches`` whose ids or scores differ from eager, per callee."""
    diff = {name: 0 for name in callees}
    first = {name: None for name in callees}
    for j, x in enumerate(batches):
        ie, se = (t.clone() for t in module(*x))
        for name, c in callees.items():
            ig, sg = c(*x)
            bad = (ie != ig).any(1) | (se != sg).any(1)
            n = int(bad.sum())
            diff[name] += n
            if n and first[name] is None:
                first[name] = j
    return {
        "rows": sum(x[0].shape[0] for x in batches),
        "diff_rows": diff,
        "first_batch": first,
    }


def gate1(inp):
    q = inp["queries"].cuda()
    ce, se = quantize_int8(q)
    co, so = OLD(q)
    torch._dynamo.reset()
    cn, sn = (t.clone() for t in torch.compile(quantize_int8, **COMPILE)(q))
    torch._dynamo.reset()
    cc, sc = (t.clone() for t in torch.compile(OLD, **COMPILE)(q))
    return {
        "n_queries": q.shape[0],
        "eager_new_equals_old": bool(torch.equal(ce, co) and torch.equal(se, so)),
        "compiled_new_equals_old_eager": bool(
            torch.equal(cn, co) and torch.equal(sn, so)
        ),
        "compiled_new_code_elems_diff": int((cn != co).sum()),
        "compiled_new_scales_diff": int((sn != so).sum()),
        "compiled_old_code_rows_diff": int((cc != co).any(1).sum()),
        "compiled_old_code_elems_diff": int((cc != co).sum()),
        "compiled_old_scales_diff": int((sc != so).sum()),
    }


def timing(module, assets, inp, job, device):
    out = []
    module.k = 100
    for mode, bs in itertools.product(("eager", "graph"), (1, 16)):
        pool, qa = inputs.query_pool(
            inp, assets["qa_s"], assets["skip"], bs=bs, seed=job.seed, device=device
        )
        ex = (pool[0],) if qa is None else (pool[0], qa[0])
        torch._dynamo.reset()
        if mode == "graph":
            arms = {
                "old": arm(module, Old, OLD, ex),
                "new": arm(module, New, quantize_int8, ex),
            }
        else:
            arms = {"old": patched(module, OLD), "new": patched(module, quantize_int8)}
        fns = [run._rotate(c, pool, qa) for c in arms.values()]
        with torch.inference_mode():
            res = measure.latency_group(fns, bs=bs, mode=mode, windows=ROUNDS)
        w = {
            name: d["window_medians_ms"] for name, (d, _) in zip(arms, res, strict=True)
        }
        out.append(
            {
                "k": 100,
                "bs": bs,
                "mode": mode,
                "rounds": ROUNDS,
                **{
                    name: {
                        "median_ci": stats.median_ci(w[name]),
                        "window_medians_ms": w[name],
                        "window_sm_mhz": d["window_sm_mhz"],
                        "sm_mhz": d.get("sm_mhz"),
                        "unstable": d["unstable"],
                    }
                    for name, (d, _) in zip(arms, res, strict=True)
                },
                "ratio_new_over_old": stats.paired_ratio_ci(w["new"], w["old"]),
            }
        )
        print(json.dumps(out[-1]), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    assert Path(retrieve.__file__).is_relative_to(WT), retrieve.__file__
    device = torch.device("cuda")
    measure.setup(1)
    measure.warm_gpu_once()
    report = {
        "env": measure.provenance() | measure.clocks(),
        "retrieve": retrieve.__file__,
    }
    print(
        json.dumps({k: report["env"][k] for k in ("code_version", "dirty", "gpu")}),
        flush=True,
    )

    h2h = config.load_matrix(
        CFG / "arxiv.yaml",
        CFG / "suites.yaml",
        "h2h",
        algos=["silvertorch"],
        backends=["triton"],
        seeds=[1],
    )
    clause = config.load_matrix(
        CFG / "arxiv.yaml",
        CFG / "suites.yaml",
        "filter",
        algos=["silvertorch"],
        backends=["triton"],
        seeds=[1],
        filter_kinds=["clause"],
        sweeps=["c0_maincat"],
    )
    jobs = (
        [j for j in h2h if j.filter_kind == "none"]
        + clause
        + [j for j in h2h if j.filter_kind == "bloom"]
    )
    inp = inputs.load_inputs(jobs[0].data, device, with_filters=True)
    report["gate1"] = gate1(inp)
    print(json.dumps(report["gate1"]), flush=True)

    report["gate2"], report["timing"] = [], []
    for job in jobs:
        assets = run.sweep_assets(job, inp, max(job.ks), device)
        measure.setup(job.seed)
        module = run.build_module(
            job, inp, assets, max(job.ks), {**job.build, **job.query[0]}
        )
        qa_s = assets["qa_s"]
        keep = assets["keep"].nonzero().reshape(-1)
        sweep = [
            (inp["queries"][r].to(device),)
            if qa_s is None
            else (inp["queries"][r].to(device), qa_s[r].to(device))
            for r in chunks(keep)
        ]
        pool, qa = inputs.query_pool(
            inp, qa_s, assets["skip"], bs=BS, seed=job.seed, device=device
        )
        h2h_batches = [
            (pool[i],) if qa is None else (pool[i], qa[i])
            for i in range(run.IDS_PROBE_BATCHES)
        ]
        for k in job.ks:
            module.k = int(k)
            torch._dynamo.reset()
            callees = {
                "old": arm(module, Old, OLD, sweep[0]),
                "new": arm(module, New, quantize_int8, sweep[0]),
            }
            cell = {
                "filter_kind": job.filter_kind,
                "sweep": job.sweep,
                "k": int(k),
                "bs": BS,
                "seed": job.seed,
                "all_kept": compare(module, callees, sweep),
            }
            if job.suite == "h2h":
                cell["h2h_entries"] = compare(module, callees, h2h_batches)
            report["gate2"].append(cell)
            print(json.dumps(cell), flush=True)
        if job.filter_kind == "clause":
            report["timing"] = timing(module, assets, inp, job, device)
        del module
        run._release()
    (a.out / "fix_gates.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
