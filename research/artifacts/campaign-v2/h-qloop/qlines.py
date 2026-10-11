"""H-QLOOP: host time per line of `run.quality`'s chunk loop (a copy of the loop with perf_counter around each line, no
added syncs), on one cell built by the harness. Prints ms per chunk per line and where each tensor lives.

    python qlines.py DATASET SUITE ALGO SWEEP       (cwd evaluation/)"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from bench import inputs, measure, run
from bench.config import load_matrix
from bench.metrics import accumulate, accumulator

ds, suite, algo, sweep = sys.argv[1:]
dev = torch.device("cuda")
jobs = load_matrix(
    Path(f"config/{ds}.yaml"), Path("config/suites.yaml"), suite, algos=[algo]
)
job = next(
    j
    for j in jobs
    if j.backend == "triton"
    and j.sweep == sweep
    and j.seed == 0
    and j.filter_kind == "clause"
)
params = {**job.build, **job.query[0]}
measure.setup(0)
inp = inputs.load_inputs(job.data, dev, with_filters=True)
assets = run.sweep_assets(job, inp, max(job.ks), dev)
module = run.build_module(
    job, inp, assets, max(job.ks), run.resolve_pool(params, assets)
)
module.k = max(job.ks)
blob = assets["blob"]
for name, t in (("queries", inp["queries"]), ("targets", inp["targets"]), ("qa_s", assets["qa_s"]), ("keep", assets["keep"]),
                ("oracle_rows", assets["oracle_rows"]), ("heldout_rows", assets["heldout_rows"]), ("topk", blob["topk"]),
                ("targets_in_filter", blob["targets_in_filter"])):  # fmt: skip
    print(
        f"{name:18} {t.device} {tuple(t.shape)} {t.dtype} contiguous={t.is_contiguous()}"
    )
T = defaultdict(float)
ks = list(job.ks)


@torch.inference_mode()
def loop():
    rows = assets["keep"].nonzero().reshape(-1)
    acc_o, acc_h = accumulator(ks, dev), accumulator(ks, dev)
    for s in range(0, rows.numel(), 16):
        t = [time.perf_counter()]
        sel = rows[s : s + 16]
        q = inp["queries"][sel].to(dev, non_blocking=True)
        qa = assets["qa_s"][sel].to(dev, non_blocking=True)
        t.append(time.perf_counter())
        ids, scores = module(q, qa)
        t.append(time.perf_counter())
        ids.clone(), scores.float().clone()
        t.append(time.perf_counter())
        m = assets["oracle_rows"][sel]
        if bool(m.any()):
            idx = m.nonzero().reshape(-1).to(dev, non_blocking=True)
            t.append(time.perf_counter())
            tk = blob["topk"][sel][m].to(dev)
            t.append(time.perf_counter())
            accumulate(acc_o, ids.index_select(0, idx), tk, ranked=True)
            t.append(time.perf_counter())
        m = assets["heldout_rows"][sel]
        if bool(m.any()):
            idx = m.nonzero().reshape(-1).to(dev, non_blocking=True)
            tt = inp["targets"][sel][m].to(dev, non_blocking=True)
            t.append(time.perf_counter())
            tt = tt.masked_fill(~blob["targets_in_filter"][sel][m].to(dev), -1)
            t.append(time.perf_counter())
            accumulate(acc_h, ids.index_select(0, idx), tt, (tt != -1).sum(dim=1))
            t.append(time.perf_counter())
        names = [
            "inputs",
            "forward",
            "clone",
            "orows",
            "topk_gather",
            "acc_oracle",
            "targets_gather",
            "tif_mask",
            "acc_heldout",
        ]
        for n, a, b in zip(names, t, t[1:], strict=False):
            T[n] += b - a
    torch.cuda.synchronize()
    return rows.numel() // 16


loop()
T.clear()
t0 = time.perf_counter()
n = loop()
wall = time.perf_counter() - t0
print(f"wall {wall:.2f} s, {n} chunks, torch threads {torch.get_num_threads()}")
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f"  {k:15} {v / n * 1e3:7.3f} ms/chunk  {v:6.2f} s")
