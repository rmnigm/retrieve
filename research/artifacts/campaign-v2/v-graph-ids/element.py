"""V-GRAPH-IDS follow-up to repro.py: the query element whose int8 code the compiled
`quantize_int8` rounds differently, the inductor code for it, and the causal check: the eager
module with `_host.quantize_int8` swapped (in this process only) for the compiled one must
reproduce the graph ids and scores that repro.py dumped, bit for bit.

usage: element.py REPRO_DIR [--seed 1] [--bs 16]
"""

import argparse
import json
from pathlib import Path

import torch
from bench import config, inputs, measure, run
from retrieve.indexing.quantize import quantize_int8
from retrieve.ops.triton import _host
from torch._inductor.utils import run_and_get_code

ROOT = Path(__file__).resolve().parents[4] / "evaluation" / "config"
COMPILE = dict(mode="reduce-overhead", dynamic=False, fullgraph=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repro", type=Path)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    device = torch.device("cuda")
    measure.setup(a.seed)
    jobs = config.load_matrix(
        ROOT / "arxiv.yaml",
        ROOT / "suites.yaml",
        "h2h",
        algos=["silvertorch"],
        backends=["triton"],
        seeds=[a.seed],
    )
    inp = inputs.load_inputs(jobs[0].data, device, with_filters=True)
    qc = torch.compile(quantize_int8, **COMPILE)
    out = {"elements": [], "causal": []}
    for job in jobs:
        assets = run.sweep_assets(job, inp, max(job.ks), device)
        pool, qa_pool = inputs.query_pool(
            inp, assets["qa_s"], assets["skip"], bs=a.bs, seed=job.seed, device=device
        )
        n = run.IDS_PROBE_BATCHES
        with torch.inference_mode():
            for i in range(n):
                q = pool[i]
                ce, se = quantize_int8(q)
                cc, _ = (t.clone() for t in qc(q))
                for r, c in (ce != cc).nonzero().tolist():
                    x, m = q[r, c], q[r].abs().amax()
                    out["elements"].append(
                        {
                            "filter_kind": job.filter_kind,
                            "batch": i,
                            "row": r,
                            "col": c,
                            "flat_row": i * a.bs + r,
                            "x": float(x),
                            "abs_max": float(m),
                            "code_eager": int(ce[r, c]),
                            "code_compiled": int(cc[r, c]),
                            "eager_x_div_m_mul_127_f32": float(x / m * 127.0),
                            "x_mul_recip_m_mul_127_f32": float(x * (1.0 / m) * 127.0),
                            "exact_f64": float(x.double() / m.double() * 127.0),
                        }
                    )
            if job.filter_kind == "none":
                torch._dynamo.reset()
                _, code = run_and_get_code(
                    torch.compile(quantize_int8, **COMPILE), pool[0]
                )
                (a.repro / "quantize_int8_inductor.py").write_text("\n\n".join(code))

        measure.setup(job.seed)
        params = {**job.build, **job.query[0]}
        module = run.build_module(job, inp, assets, max(job.ks), params)
        args = [
            (pool[i],) if qa_pool is None else (pool[i], qa_pool[i]) for i in range(n)
        ]
        orig = _host.quantize_int8
        for k in job.ks:
            module.k = int(k)
            dump = torch.load(
                a.repro / f"{job.filter_kind}-k{k}-bs{a.bs}-seed{job.seed}.pt"
            )
            with torch.inference_mode():
                _host.quantize_int8 = lambda t: tuple(x.clone() for x in qc(t))
                try:
                    res = [tuple(t.clone() for t in module(*x)) for x in args]
                finally:
                    _host.quantize_int8 = orig
            ids = torch.cat([r[0] for r in res]).cpu()
            sc = torch.cat([r[1] for r in res]).float().cpu()
            out["causal"].append(
                {
                    "filter_kind": job.filter_kind,
                    "k": int(k),
                    "eager_with_compiled_quantize_equals_graph": bool(
                        torch.equal(ids, dump["ids_graph"])
                        and torch.equal(sc, dump["scores_graph"])
                    ),
                    "equals_eager": bool(
                        torch.equal(ids, dump["ids_eager"])
                        and torch.equal(sc, dump["scores_eager"])
                    ),
                }
            )
            print(json.dumps(out["causal"][-1]), flush=True)
        del module
        run._release()
    for e in out["elements"]:
        print(json.dumps(e))
    (a.repro / "element.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
