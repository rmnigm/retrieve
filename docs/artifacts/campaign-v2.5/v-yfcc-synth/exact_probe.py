"""YFCC synth exact-gate probe (controller 2026-10-10 17:00), quality only, no gate: V1 + V2 triton recall_oracle@100 / @1000
per rate through the harness's own inputs, assets, module build and quality pass (k_max 1000, the cached oracle blobs); at p001
also the library's torch V2 path and an fp32-math exact top-k over the item table, fp16-rounded and fp32 (is the residual the storage?).
Run from evaluation/: python exact_probe.py OUT.json"""

import dataclasses
import json
import sys
from pathlib import Path

import torch

from bench import algos, inputs, run
from bench.config import load_matrix

dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
jobs = load_matrix(
    Path("config/yfcc10m-synth.yaml"),
    Path("config/suites.yaml"),
    "synth",
    algos=["linr_v1_filter_mask", "linr_v2"],
    backends=["triton"],
    ks=[100, 1000],
)
inp = inputs.load_inputs(jobs[0].data, dev)
out = {}


def recalls(module, assets):
    q, _, ids, _ = run.quality(module, inp, assets, [100, 1000], dev)
    return {k: q["oracle"][f"recall@{k}"] for k in (100, 1000)}, ids


@torch.inference_mode()
def table_exact(assets, half, k=1000):
    """Exact masked top-k with fp32 math over the item table, rounded to fp16 first (the exact arms' storage) or not."""
    emb16 = inp["item_embs"].half().float() if half else inp["item_embs"]
    rows = assets["oracle_rows"].nonzero().reshape(-1)
    topk = assets["blob"]["topk"]
    hit = {100: 0.0, 1000: 0.0}
    for s in range(0, rows.numel(), 64):
        r = rows[s : s + 64]
        sc = inp["queries"][r].to(dev) @ emb16.T
        mask = assets["filter_mod"](assets["qa_s"][r].to(dev))
        sc = sc.masked_fill(~mask, float("-inf"))
        got = sc.topk(k, dim=1).indices.cpu()
        for g, ref in zip(got, topk[r], strict=True):
            for kk in hit:
                rf = ref[:kk][ref[:kk] >= 0]
                hit[kk] += len(set(g[:kk].tolist()) & set(rf.tolist())) / max(
                    len(rf), 1
                )
    return {kk: v / rows.numel() for kk, v in hit.items()}


for sweep in sorted({j.sweep for j in jobs}):
    unit = [j for j in jobs if j.sweep == sweep]
    assets = run.sweep_assets(unit[0], inp, 1000, dev)
    res, ids_by = {"pass_rate": assets["pass_rate"]}, {}
    for j in unit:
        m = run.build_module(j, inp, assets, 1000, {})
        res[j.algo], ids_by[j.algo] = recalls(m, assets)
        del m
    a, b = ids_by["linr_v1_filter_mask"], ids_by["linr_v2"]
    res["v1_v2_ids_equal_rows"] = (a == b).all(dim=1).float().mean().item()
    if sweep == "p001":
        j = dataclasses.replace(unit[1], backend="torch")
        m = algos.build(
            "linr_v2",
            inp["item_embs"],
            k=1000,
            backend="torch",
            filter_kind="clause",
            filter_mod=assets["filter_mod"],
            item_attrs=inp["item_attrs"],
            clause_is_reverse=inp["clause_is_reverse"],
            params={},
            seed=0,
        )
        res["linr_v2_torch"], _ = recalls(m, assets)
        del m
        res["fp16_table_fp32_math"] = table_exact(assets, True)
        res["fp32_table_fp32_math"] = table_exact(assets, False)
    out[sweep] = res
    print(sweep, json.dumps(res), flush=True)
    torch.cuda.empty_cache()
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
