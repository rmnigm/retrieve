"""C6 recall-ceiling check on goodreads / arXiv (controller 2026-10-12), modelled on yfcc-int8/int8_check.py: is SilverTorch's ceiling the one global int8
item scale? One SilverTorch triton index at the given (n_lists, n_probe), seed 0, built by the harness, its own probed clusters per query,
recall_oracle@100 over the kept queries with the probed (and filter-passing) items scored four ways:
  shipped  the module itself through run.quality (global item scale, per-row int8 query: the library path)
  global   the same arithmetic recomputed densely (q int8 per row · global-scale item codes): checks the method against `shipped`
  per_row  per-row item int8 scales (quantize_int8 per item) on the same int32 path (ST-6: not through Meta's kernel)
  fp16     fp16 query · fp16 items: the IVF ceiling without quantisation
Quality only, no timing. Usage (from evaluation/): python ceiling_check.py DATASET SUITE SWEEP N_LISTS N_PROBE OUT_JSON"""

import json
import sys
from pathlib import Path

import torch
from bench import inputs, measure, run
from bench.config import load_matrix
from bench.metrics import accumulate, accumulator, finalize
from retrieve.indexing.quantize import quantize_int8

ds, suite, sweep, n_lists, n_probe, out = sys.argv[1:]
dev = torch.device("cuda")
jobs = load_matrix(
    Path(f"config/{ds}.yaml"),
    Path("config/suites.yaml"),
    suite,
    algos=["silvertorch"],
)
job = next(
    j
    for j in jobs
    if j.backend == "triton"
    and j.filter_kind == "clause"
    and j.sweep == sweep
    and j.seed == 0
)
params = {**job.build, "n_lists": int(n_lists), "n_probe": int(n_probe)}
measure.setup(0)
inp = inputs.load_inputs(job.data, dev, with_filters=True)
assets = run.sweep_assets(job, inp, 100, dev)
m = run.build_module(job, inp, assets, 100, params)
m.k = 100
shipped = run.quality(m, inp, assets, [100], dev)[0]["oracle"]["recall@100"]

perm = m.sort_perm  # sorted (cluster-major) space, as the module's codes
# items scored in blocks: no [N, D] fp16 copy (10 M d768 is 15 GB each)
n, BLOCK = perm.numel(), 1 << 20
codes_r = torch.empty_like(m.item_codes)
scales_r = torch.empty(n, dtype=torch.float32, device=dev)
for b in range(0, n, BLOCK):
    c, s = quantize_int8(inp["item_embs"][perm[b : b + BLOCK]].float())
    codes_r[b : b + BLOCK], scales_r[b : b + BLOCK] = c, s


def blocked(rows, fn):
    out = torch.empty((rows, n), dtype=torch.float32, device=dev)
    for b in range(0, n, BLOCK):
        out[:, b : b + BLOCK] = fn(b, min(n, b + BLOCK))
    return out


cluster_of = torch.repeat_interleave(
    torch.arange(m.n_lists, device=dev), m.cluster_offsets.diff()
)
rows = assets["keep"].nonzero().reshape(-1)
acc = {v: accumulator([100], dev) for v in ("global", "per_row", "fp16")}
filt = assets["filter_mod"]
with torch.inference_mode():
    for s in range(0, rows.numel(), 64):
        sel = rows[s : s + 64]
        orow = assets["oracle_rows"][sel]
        if not bool(orow.any()):
            continue
        q = inp["queries"][sel].to(dev)
        qa = assets["qa_s"][sel].to(dev)
        probed = torch.zeros(q.shape[0], m.n_lists, dtype=torch.bool, device=dev)
        probed.scatter_(1, m._phase1_probe_ids(q), True)
        allowed = probed[:, cluster_of] & filt.evaluate_mask(qa)[:, perm]
        q_codes, q_scales = quantize_int8(q.float())
        q_codes = q_codes.to(torch.float16)
        q16 = q.to(torch.float16)

        def mm(lhs, codes):
            return blocked(
                lhs.shape[0],
                lambda b, e: torch.mm(
                    lhs, codes(b, e).to(torch.float16).t(), out_dtype=torch.float32
                ),
            )

        variants = {
            "global": lambda: (
                mm(q_codes, lambda b, e: m.item_codes[b:e])
                * q_scales[:, None]
                * m._global_scale_f
            ),
            "per_row": lambda: (
                mm(q_codes, lambda b, e: codes_r[b:e])
                * q_scales[:, None]
                * scales_r[None, :]
            ),
            "fp16": lambda: mm(q16, lambda b, e: inp["item_embs"][perm[b:e]]),
        }  # fmt: skip (same arithmetic and order as before, per item block)
        idx = orow.nonzero().reshape(-1).to(dev)
        targets = assets["blob"]["topk"][sel][orow].to(dev)
        for v, score in variants.items():
            sc = score().masked_fill_(~allowed, float("-inf"))
            top, pos = torch.topk(sc, 100, dim=1)
            ids = torch.where(torch.isfinite(top), perm[pos], torch.full_like(pos, -1))
            accumulate(acc[v], ids.index_select(0, idx), targets, ranked=True)
            del sc
res = {"dataset": ds, "sweep": sweep, "params": params, "pass_rate": assets["pass_rate"], "code_version": measure.code_version(),
       "recall_oracle@100": {"shipped": shipped, **{v: finalize(a)["recall@100"] for v, a in acc.items()}}}  # fmt: skip
Path(out).write_text(json.dumps(res, indent=1))
print(json.dumps(res))
