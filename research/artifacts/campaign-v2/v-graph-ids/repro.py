"""V-GRAPH-IDS: Triton SilverTorch eager vs graph (torch.compile reduce-overhead) on the
H2H-FINAL arXiv cells, built and fed exactly as `bench run` does (seed, pool, k, bs).

Per (kind, k): the first IDS_PROBE_BATCHES pool batches through the eager module and through
`measure.graph_callable`, ids + scores dumped and compared with torch.equal. Then the two
torch-side stages the compiled graph re-generates (phase 1 `query @ centroids.t()` + topk, and
`quantize_int8(query)`) are run eager and under the same torch.compile mode, and the eager
Triton op is fed the compiled stage's output to see which stage reproduces the graph result.

`--emulate-div` sets inductor's `emulate_divison_rounding` (div_rn, as eager) in this process.

usage: repro.py OUT_DIR [--seed 1] [--bs 16] [--emulate-div]
"""

import argparse
import json
from pathlib import Path

import torch
import torch._inductor.config
from bench import config, inputs, measure, run
from retrieve.indexing.quantize import quantize_int8
from retrieve.ops.triton.codesigned_probe_score import _codesigned_probe_score_impl

ROOT = Path(__file__).resolve().parents[4] / "evaluation" / "config"
COMPILE = dict(mode="reduce-overhead", dynamic=False, fullgraph=True)


def first_diff(a_ids, a_s, b_ids, b_s, k):
    rows = (a_ids != b_ids).any(1) | (a_s != b_s).any(1)
    out = []
    for r in rows.nonzero().reshape(-1).tolist():
        sa, sb = a_s[r], b_s[r]
        ia, ib = set(a_ids[r].tolist()), set(b_ids[r].tolist())
        out.append(
            {
                "row": r,
                "scores_equal": bool(torch.equal(sa, sb)),
                "score_max_abs_diff": float((sa - sb).abs().max()),
                "n_score_slots_diff": int((sa != sb).sum()),
                "id_sets_equal": ia == ib,
                "only_eager": sorted(ia - ib)[:10],
                "only_graph": sorted(ib - ia)[:10],
                "kth_eager": float(sa[k - 1]),
                "kth_graph": float(sb[k - 1]),
                "kth_minus1_eager": float(sa[k - 2]),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--emulate-div", action="store_true")
    a = ap.parse_args()
    torch._inductor.config.emulate_divison_rounding = a.emulate_div
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    measure.setup(a.seed)
    measure.warm_gpu_once()

    jobs = config.load_matrix(
        ROOT / "arxiv.yaml",
        ROOT / "suites.yaml",
        "h2h",
        algos=["silvertorch"],
        backends=["triton"],
        seeds=[a.seed],
    )
    inp = inputs.load_inputs(jobs[0].data, device, with_filters=True)
    report = {"env": measure.provenance() | measure.clocks(), "cells": []}
    for job in jobs:
        assets = run.sweep_assets(job, inp, max(job.ks), device)
        params = {**job.build, **job.query[0]}
        measure.setup(job.seed)
        module = run.build_module(job, inp, assets, max(job.ks), params)
        pool, qa_pool = inputs.query_pool(
            inp, assets["qa_s"], assets["skip"], bs=a.bs, seed=job.seed, device=device
        )
        n = run.IDS_PROBE_BATCHES
        args = [
            (pool[i],) if qa_pool is None else (pool[i], qa_pool[i]) for i in range(n)
        ]
        for k in job.ks:
            module.k = int(k)
            with torch.inference_mode():
                eager = [tuple(t.clone() for t in module(*x)) for x in args]
                torch._dynamo.reset()
                g = measure.graph_callable(module, *args[0])
                graph = [tuple(t.clone() for t in g(*x)) for x in args]
            ids_e = torch.cat([e[0] for e in eager]).cpu()
            sc_e = torch.cat([e[1] for e in eager]).float().cpu()
            ids_g = torch.cat([e[0] for e in graph]).cpu()
            sc_g = torch.cat([e[1] for e in graph]).float().cpu()
            stem = f"{job.filter_kind}-k{k}-bs{a.bs}-seed{job.seed}"
            torch.save(
                {
                    "ids_eager": ids_e,
                    "scores_eager": sc_e,
                    "ids_graph": ids_g,
                    "scores_graph": sc_g,
                },
                a.out / f"{stem}.pt",
            )

            # Stage isolation, eager vs the same compile mode, over the same batches.
            torch._dynamo.reset()
            p1c = torch.compile(module._phase1_probe_ids, **COMPILE)
            qc = torch.compile(quantize_int8, **COMPILE)
            stages = {
                "probe_ids_rows_diff": 0,
                "q_codes_rows_diff": 0,
                "q_scales_rows_diff": 0,
            }
            replay = []
            with torch.inference_mode():
                for x in args:
                    q = x[0]
                    pe, pc = module._phase1_probe_ids(q), p1c(q).clone()
                    ce, se = quantize_int8(q)
                    cc, scc = (t.clone() for t in qc(q))
                    stages["probe_ids_rows_diff"] += int(
                        (pe.sort(1).values != pc.sort(1).values).any(1).sum()
                    )
                    stages["q_codes_rows_diff"] += int((ce != cc).any(1).sum())
                    stages["q_scales_rows_diff"] += int((se != scc).sum())
                    if module.filter_mode == "none":
                        replay.append(
                            _codesigned_probe_score_impl(
                                q,
                                pc,
                                module.cluster_offsets,
                                module.item_codes,
                                module.sort_perm,
                                module._global_scale_f,
                                module.k,
                                module._probe_width,
                            )
                        )
            cell = {
                "filter_kind": job.filter_kind,
                "k": int(k),
                "bs": a.bs,
                "seed": job.seed,
                "ids_equal": bool(torch.equal(ids_e, ids_g)),
                "scores_equal": bool(torch.equal(sc_e, sc_g)),
                "rows": ids_e.shape[0],
                "diff_rows": first_diff(ids_e, sc_e, ids_g, sc_g, int(k)),
                "stages": stages,
            }
            if replay:
                ids_r = torch.cat([r[0] for r in replay]).cpu()
                sc_r = torch.cat([r[1] for r in replay]).float().cpu()
                cell["eager_op_on_compiled_probe_ids_equals_graph"] = bool(
                    torch.equal(ids_r, ids_g) and torch.equal(sc_r, sc_g)
                )
            report["cells"].append(cell)
            print(
                json.dumps({k_: v for k_, v in cell.items() if k_ != "diff_rows"}),
                "diff_rows:",
                len(cell["diff_rows"]),
                flush=True,
            )
            for d in cell["diff_rows"][:3]:
                print("   ", json.dumps(d), flush=True)
        del module
        run._release()
    (a.out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
