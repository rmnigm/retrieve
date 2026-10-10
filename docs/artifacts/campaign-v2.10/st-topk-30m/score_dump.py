"""ST-TOPK 30 M gate, scores: dump our SilverTorch triton top-k ids + scores (laion30m d256 bloom partial, n_lists 16384, n_probe 128, k 100) for the first
64 kept queries of each sweep, at bs 1 (first 8), 16 and 64, eager, through the harness of the tree this venv imports; compare two dumps with torch.equal.
    python score_dump.py dump CONFIG_DIR OUT.pt        (run from the tree's evaluation/, its venv)
    python score_dump.py compare A.pt B.pt"""

import sys
from pathlib import Path

import torch


def dump(cfg: Path, out: Path) -> None:
    from bench import config, inputs, run

    dev = torch.device("cuda")
    jobs = [
        j
        for j in config.load_matrix(
            cfg / "laion30m.yaml",
            cfg / "suites.yaml",
            "stt-laion30m",
            backends=["triton"],
        )
        if j.seed == 0
    ]
    inp = inputs.load_inputs(jobs[0].data, dev)
    res = {}
    for j in jobs:
        assets = run.sweep_assets(j, inp, 100, dev)
        m = run.build_module(j, inp, assets, 100, {**j.build, **j.query[0]})
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
                res[j.sweep, bs] = (torch.cat(ids), torch.cat(sc))
        del m, assets
        torch.cuda.empty_cache()
    torch.save(res, out)


def compare(a: Path, b: Path) -> None:
    x, y = torch.load(a), torch.load(b)
    for k in sorted(x):
        print(k, "ids equal", torch.equal(x[k][0], y[k][0]), "scores equal", torch.equal(x[k][1], y[k][1]),
              "max |d|", float((x[k][1] - y[k][1]).abs().max()))  # fmt: skip


if __name__ == "__main__":
    {"dump": dump, "compare": compare}[sys.argv[1]](
        Path(sys.argv[2]), Path(sys.argv[3])
    )
