"""V1-BS1 large-N gate, scores + path: for each (dataset, sweep) of the suite `v1g` in CONFIG_DIR, build LiNR V1 triton through the harness of the tree
this venv imports, record which bs-1 scorer its register-time check chose (`gemv_exact` on the module's PostfilterKNN, absent before V1-BS1), and dump
ids + scores for the first 64 kept queries at bs 1 (first 8), 16 and 64; compare two dumps with torch.equal.
    python score_dump.py dump CONFIG_DIR OUT.pt        (from the tree's evaluation/, its venv)
    python score_dump.py compare A.pt B.pt"""

import sys
from pathlib import Path

import torch

DATASETS = {"laion30m": 256, "laion30m-synth": 256, "yfcc10m-synth": 192}


def dump(cfg: Path, out: Path) -> None:
    from bench import config, inputs, run

    dev = torch.device("cuda")
    res = {}
    for ds, dim in DATASETS.items():
        jobs = config.load_matrix(
            cfg / f"{ds}.yaml", cfg / "suites.yaml", "v1g", dims=[dim]
        )
        inp = inputs.load_inputs(jobs[0].data, dev)
        for j in jobs:
            assets = run.sweep_assets(j, inp, 100, dev)
            m = run.build_module(j, inp, assets, 100, {})
            gemv = [
                getattr(s, "gemv_exact", None)
                for s in m.modules()
                if hasattr(s, "item_embs_t")
            ]
            rows = assets["keep"].nonzero().reshape(-1)[:64]
            prep = getattr(m, "prepare_queries", None)
            with torch.inference_mode():
                for bs, n in ((1, 8), (16, 64), (64, 64)):
                    ids, sc = [], []
                    for s in range(0, n, bs):
                        r = rows[s : s + bs]
                        qa = assets["qa_s"][r].to(dev)
                        i, c = m(inp["queries"][r].to(dev), prep(qa) if prep else qa)
                        ids.append(i.cpu())
                        sc.append(c.float().cpu())
                    res[ds, j.sweep, bs] = (torch.cat(ids), torch.cat(sc))
            res[ds, j.sweep, "gemv_exact"] = gemv
            print(ds, j.sweep, "gemv_exact", gemv, flush=True)
            del m, assets
            torch.cuda.empty_cache()
        del inp
        torch.cuda.empty_cache()
    torch.save(res, out)


def compare(a: Path, b: Path) -> None:
    x, y = torch.load(a), torch.load(b)
    for k in sorted((k for k in x if k[2] != "gemv_exact"), key=str):
        print(
            k,
            "ids equal",
            torch.equal(x[k][0], y[k][0]),
            "scores equal",
            torch.equal(x[k][1], y[k][1]),
        )
    for k in sorted((k for k in x if k[2] == "gemv_exact"), key=str):
        print(k[:2], "gemv_exact A", x[k], "B", y[k])


if __name__ == "__main__":
    {"dump": dump, "compare": compare}[sys.argv[1]](
        Path(sys.argv[2]), Path(sys.argv[3])
    )
