"""Eager vs ``torch.compile(mode="max-autotune")`` on one cell (campaign-v2 #13 smoke).

Builds the same module twice through ``bench.algos.build`` (same seed, same params; the second
with ``compile``), runs every kept query of one sweep through both at ``k``, and prints the
share of rows whose top-``k`` ids are identical as sets and in order, the mean Jaccard, and the
max |score diff| over finite pairs. No tolerance is applied: the numbers are reported as found.

    CUDA_VISIBLE_DEVICES=0 uv run --directory evaluation python \\
        ../docs/artifacts/cv2-harness-data/compile_parity.py --dataset goodreads \\
        --algo silvertorch --filter-kind clause --sweep c0_genre --param n_probe=24
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bench import algos, run
from bench.config import load_dataset
from bench.inputs import load_inputs
from bench.metrics import jaccard_at_k


def _forward(module, inp, assets, device, k):
    module.k = k
    rows = assets["keep"].nonzero().reshape(-1)
    ids, scores = [], []
    with torch.inference_mode():
        for s in range(0, rows.numel(), run.QUALITY_CHUNK):
            sel = rows[s : s + run.QUALITY_CHUNK]
            qa = assets["qa_s"][sel].to(device) if assets["qa_s"] is not None else None
            i, sc = module(inp["queries"][sel].to(device), qa)
            ids.append(i.cpu())
            scores.append(sc.float().cpu())
    return torch.cat(ids), torch.cat(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--filter-kind", default="clause")
    ap.add_argument("--sweep", required=True)
    ap.add_argument(
        "--param", action="append", default=[], help="name=int, e.g. n_probe=24"
    )
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()
    device = torch.device("cuda")
    ds = load_dataset(Path(f"config/{a.dataset}.yaml"), a.dim)
    params = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.param}
    inp = load_inputs(ds, device)

    class _Job:  # the fields sweep_assets / build_module read
        filter_kind, sweep, algo, backend, seed = (
            a.filter_kind,
            a.sweep,
            a.algo,
            a.backend,
            0,
        )
        clauses = ds.clauses[a.filter_kind][a.sweep]
        bloom = dict(algos.BLOOM_DEFAULTS)
        data = ds

    assets = run.sweep_assets(_Job, inp, a.k, device)
    out = {}
    for name, extra in (("eager", {}), ("compiled", {"compile": "max-autotune"})):
        t0 = time.perf_counter()
        module = run.build_module(_Job, inp, assets, a.k, {**params, **extra})
        warm = run.compile_warmup(module, inp, assets, device) if extra else None
        out[name] = _forward(module, inp, assets, device, a.k)
        print(f"{name}: build+warm {time.perf_counter() - t0:.1f}s compile_s {warm}")
        del module
        torch._dynamo.reset()
    (ids_e, sc_e), (ids_c, sc_c) = out["eager"], out["compiled"]
    both = torch.isfinite(sc_e) & torch.isfinite(sc_c)
    diff = (sc_e - sc_c).abs()[both]
    same_set = sum(
        set(x.tolist()) == set(y.tolist()) for x, y in zip(ids_e, ids_c, strict=True)
    )
    res = {
        "dataset": a.dataset, "algo": a.algo, "filter_kind": a.filter_kind, "sweep": a.sweep,
        "params": params, "k": a.k, "rows": int(ids_e.shape[0]),
        "rows_ids_equal_ordered": int((ids_e == ids_c).all(dim=1).sum()),
        "rows_ids_equal_as_sets": int(same_set),
        "mean_jaccard": jaccard_at_k(ids_c, ids_e, a.k),
        "score_max_abs_diff": diff.max().item() if diff.numel() else 0.0,
        "scores_bit_equal": bool(torch.equal(sc_e, sc_c)),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }  # fmt: skip
    print(json.dumps(res))


if __name__ == "__main__":
    main()
