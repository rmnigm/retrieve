#!/usr/bin/env python3
"""d1a_mode_ids.py — WP-5's "ids identical across `mode`" clause, which the records cannot
decide (quality is measured once, eager; the perf loop keeps no ids).

For one representative cell per (dataset, algo, backend) of stage a — the headline clause
sweep at the algo's default params, seed 0 — it builds the module exactly as `run.py` does
(`run.sweep_assets` / `run.build_module`), then for every `bs` of the suite and every `k`
compares the eager `forward` against `measure.graph_callable`'s compiled+captured one on the
same fixed-seed pool batches: `torch.equal` on ids, and the max |Δscore|.

`official` is eager-only (O D7) and is skipped, with the reason printed.

    python3 d1a_mode_ids.py [--batches 8] [--out d1a_mode_ids.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[4] / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

import torch  # noqa: E402
from loguru import logger  # noqa: E402

from bench import inputs, measure, run as run_mod  # noqa: E402
from bench.config import QUERY_PARAMS, load_matrix  # noqa: E402

# goodreads only: stage a was stopped at the dataset boundary (orchestrator), so arxiv
# has no campaign records this clause would belong to.
SWEEP = {"goodreads": "c0_genre"}
KS = (100, 1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "d1a_mode_ids.json")
    a = ap.parse_args()
    device = torch.device("cuda")
    measure.setup(0)
    measure.warm_gpu_once()
    rows = []
    for ds, sweep in SWEEP.items():
        jobs = load_matrix(
            EVAL_DIR / "config" / f"{ds}.yaml",
            EVAL_DIR / "config" / "suites.yaml",
            "filter",
            dims=[128],
            seeds=[0],
            sweeps=[sweep],
            filter_kinds=["clause"],
        )
        inp = inputs.load_inputs(jobs[0].data, device, with_filters=True)
        seen = set()
        for job in jobs:
            if (job.algo, job.backend) in seen:
                continue
            seen.add((job.algo, job.backend))
            params = job.cells()[0]
            k_max = max(job.ks)
            assets = run_mod.sweep_assets(job, inp, k_max, device)
            measure.setup(job.seed)
            module = run_mod.build_module(job, inp, assets, k_max, params)
            q = {k: v for k, v in params.items() if k in QUERY_PARAMS}
            if q:
                module.set_query_params(**q)
            for bs in job.batch_sizes:
                pool, qa_pool = inputs.query_pool(
                    inp, assets["qa_s"], assets["skip"], bs=bs, seed=job.seed, device=device
                )
                for k in KS:
                    module.k = int(k)
                    row = {
                        "dataset": ds, "sweep": sweep, "algo": job.algo,
                        "backend": job.backend, "params": params, "bs": bs, "k": k,
                    }  # fmt: skip
                    example = (pool[0],) if qa_pool is None else (pool[0], qa_pool[0])
                    try:
                        compiled = measure.graph_callable(module, *example)
                    except measure.NotCapturable as exc:
                        row.update(skipped=str(exc))
                        rows.append(row)
                        logger.info("{} {} {} bs={} k={}: {}", ds, job.algo, job.backend, bs, k, exc)
                        continue
                    neq = 0
                    dmax = 0.0
                    with torch.inference_mode():
                        for i in range(min(a.batches, pool.shape[0])):
                            args = (pool[i],) if qa_pool is None else (pool[i], qa_pool[i])
                            ie, se = module(*args)
                            ie, se = ie.clone(), se.clone()
                            ig, sg = compiled(*args)
                            ig, sg = ig.clone(), sg.clone()
                            torch.cuda.synchronize()
                            neq += int((ie != ig).any(dim=1).sum())
                            dmax = max(dmax, float((se.float() - sg.float()).abs().max()))
                    row.update(rows_compared=min(a.batches, pool.shape[0]) * bs,
                               rows_differing=neq, score_max_abs_diff=dmax)
                    rows.append(row)
                    logger.info(
                        "{} {} {} bs={} k={}: rows_differing={} dscore={:.3e}",
                        ds, job.algo, job.backend, bs, k, neq, dmax,
                    )
                    del compiled
                    torch._dynamo.reset()
            del module
            run_mod._release()
        del inp, assets
        run_mod._release()
    a.out.write_text(json.dumps(rows, indent=1))
    measured = [r for r in rows if "rows_differing" in r]
    bad = [r for r in measured if r["rows_differing"]]
    print(f"\n{len(measured)} (algo, backend, bs, k) comparisons, "
          f"{len(rows) - len(measured)} skipped (not capturable)")
    print("ids identical across mode:", "PASS" if not bad else f"FAIL on {len(bad)}")
    for r in bad:
        print("  ", r)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
