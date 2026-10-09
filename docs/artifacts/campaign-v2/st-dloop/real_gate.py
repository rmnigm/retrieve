"""ST-DLOOP bit-exact gate on real data: the campaign-v2.1 `_impl`s (package `retrieve_v21`) against the
working tree's, on one SilverTorch index at the bench's build (n_lists 1024, bloom 1024 / 5, seed 0):
ids and scores torch.equal for none, bloom and exact sweeps × bs {1, 16} × k {100, 1000} × n_probe
{24, 1024}, 4 query batches each from ``inputs.query_pool`` (seed 0).

    cd evaluation && PYTHONPATH=<v21 pkg>:... python real_gate.py pubmed 768 out.json \
        --bloom c0_mesh --exact c0_mesh,all5
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path

import torch

from bench import algos, config, inputs

DEV = torch.device("cuda")
FILES = {"none": "codesigned_probe_score", "bloom": "codesigned_probe_score",
         "exact": "codesigned_probe_score_exact"}  # fmt: skip
ARMS = {
    arm: {m: importlib.import_module(f"{pkg}.ops.triton.{f}") for m, f in FILES.items()}
    for arm, pkg in (("before", "retrieve_v21"), ("after", "retrieve"))
}

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("dim", type=int)
ap.add_argument("out")
ap.add_argument("--bloom", default="c0_mesh")
ap.add_argument("--exact", default="c0_mesh,all5")
ap.add_argument("--none", action=argparse.BooleanOptionalAction, default=True)
args = ap.parse_args()

ds = config.load_dataset(Path(f"config/{args.dataset}.yaml"), args.dim)
inp = inputs.load_inputs(ds, DEV, with_filters=True)
res = {"dataset": args.dataset, "dim": args.dim, "n_items": inp["n_items"], "rows": []}


def impl(mod, mode, m, q, probe, qa, k, width):
    lay = (
        q,
        probe,
        m.cluster_offsets,
        m.item_codes,
        m.sort_perm,
        m._global_scale_f,
        k,
        width,
    )
    if mode == "exact":
        return mod._codesigned_probe_score_exact_impl(
            *lay, item_clause_attrs=m.item_clause_attrs, clause_is_reverse=m.clause_is_reverse,
            query_clause_attrs=qa.long())  # fmt: skip
    if mode == "bloom":
        return mod._codesigned_probe_score_impl(
            *lay, query_bit_positions=m._query_bit_positions(qa), bloom_transposed=m.bloom_transposed)  # fmt: skip
    return mod._codesigned_probe_score_impl(*lay)


cells = [("none", None)] if args.none else []
cells += [("bloom", s) for s in args.bloom.split(",") if s]
cells += [("exact", s) for s in args.exact.split(",") if s]
mods: dict[str, torch.nn.Module] = {}
m = None
for mode, sweep in cells:
    if mode not in mods:
        mods.clear()
        # The previous index must go before the next is built; SilverTorch holds a bound method of
        # itself (its forward dispatch), so only the cycle collector frees it.
        m = None
        gc.collect()
        torch.cuda.empty_cache()
        kind = {"none": "none", "bloom": "bloom", "exact": "clause"}[mode]
        mods[mode] = algos.build(
            "silvertorch", inp["item_embs"], k=100, backend="triton", filter_kind=kind,
            item_attrs=inp["item_attrs"], clause_is_reverse=inp["clause_is_reverse"], seed=0,
        )  # fmt: skip
        if all(c[0] == mode for c in cells):
            # The fp32 table (30 GB at PubMed) is only for the build; k 1000 × n_probe 1024's
            # probe_ids_kernel spills 122 KB a thread, and the driver's local-memory pool for it is
            # ~25 GB on top of the index.
            inp["item_embs"] = None
            gc.collect()
            torch.cuda.empty_cache()
    m = mods[mode]
    sizes = m.cluster_sizes.sort(descending=True).values
    clauses = (
        None
        if sweep is None
        else ds.clauses["clause" if mode == "exact" else "bloom"][sweep]
    )
    qa_sweep, skip = inputs.sweep_qa(inp["qa"], clauses)
    for bs in (1, 16):
        pool_q, pool_a = inputs.query_pool(
            inp, qa_sweep, skip, bs=bs, seed=0, n_pool=4, device=DEV
        )
        for n_probe in (24, m.n_lists):
            width = int(sizes[:n_probe].sum())
            for k in (100, 1000):
                same, n_inf = True, 0
                for i in range(pool_q.shape[0]):
                    q = pool_q[i]
                    qa = None if pool_a is None else pool_a[i]
                    probe = torch.topk(q @ m.centroids.t(), n_probe, dim=1).indices
                    ib, sb = impl(ARMS["before"][mode], mode, m, q, probe, qa, k, width)
                    ia, sa = impl(ARMS["after"][mode], mode, m, q, probe, qa, k, width)
                    same &= torch.equal(ib, ia) and torch.equal(sb, sa)
                    n_inf += int(torch.isinf(sb).sum())
                row = {"mode": mode, "sweep": sweep, "bs": bs, "n_probe": n_probe, "k": k,
                       "equal": same, "inf_slots": n_inf}  # fmt: skip
                res["rows"].append(row)
                print(row, flush=True)
    free, total = torch.cuda.mem_get_info()
    print(f"after {mode}/{sweep}: allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, "
          f"reserved {torch.cuda.memory_reserved() / 2**30:.1f}, in use {(total - free) / 2**30:.1f}",
          flush=True)  # fmt: skip
Path(args.out).write_text(json.dumps(res, indent=1))
print("ALL EQUAL" if all(r["equal"] for r in res["rows"]) else "MISMATCH", flush=True)
