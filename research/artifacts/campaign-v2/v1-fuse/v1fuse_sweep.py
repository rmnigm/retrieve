"""V1-FUSE: tile sweep of the two score-masking kernels against the old mask + where pair, interleaved
(``measure.latency_group``), on the real attrs of DATASET (sweep p01, bs 1 and 16). Run from evaluation/, GPU 0.
  python v1fuse_sweep.py DATASET OUT.json"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from bench import algos, inputs, measure
from bench.config import load_dataset
from retrieve.ops.triton.bloom_match import _bloom_match_scores_impl, bloom_match
from retrieve.ops.triton.clause_mask import ClauseMaskConfig, _clause_mask_impl, clause_mask

sys.path.insert(0, str(Path(__file__).parent))
from v1fuse_gpu import DEV, LAT, _paired  # noqa: E402

TILES = [(128, 4), (256, 4), (512, 4), (128, 8), (256, 8), (512, 2), (1024, 8)]
dataset, out_path = sys.argv[1:]
ds = load_dataset(Path("config") / f"{dataset}.yaml", 128)
inp = inputs.load_inputs(ds, DEV)
rows = []
for fk in ("clause", "bloom"):
    f = inputs.build_filters(fk, inp, ["triton"], bloom=algos.BLOOM_DEFAULTS)["triton"]
    qa_s, skip = inputs.sweep_qa(inp["qa"], ds.clauses[fk]["p01"])
    for bs in (1, 16):
        qa = qa_s[~skip][:bs].to(DEV) if skip is not None else qa_s[:bs].to(DEV)
        scores = torch.randn(bs, inp["n_items"], device=DEV)
        if fk == "clause":
            a = (f.item_clause_attrs, f.clause_is_reverse, qa)
            old = lambda: torch.where(clause_mask(*a), scores, float("-inf"))  # noqa: E731
            arms = {t: (lambda t=t: _clause_mask_impl(*a, config=ClauseMaskConfig(*t), scores=scores))
                    for t in TILES}  # fmt: skip
            want = old()
            assert all(torch.equal(fn(), want) for fn in arms.values())
        else:
            qb = f._build_query_sigs(qa)
            old = lambda: torch.where(bloom_match(qb, f.bloom_sigs), scores, float("-inf"))  # noqa: E731
            arms = {t: (lambda t=t: _bloom_match_scores_impl(scores, qb, f.bloom_sigs, block_n=t[0],
                                                             num_warps=t[1])) for t in TILES}  # fmt: skip
            want = old()
            assert all(torch.equal(fn(), want) for fn in arms.values())
        with torch.inference_mode():
            timed = measure.latency_group([old, *arms.values()], bs=bs, mode="eager", **LAT)
        base = timed[0][0]
        row = {"filter_kind": fk, "bs": bs, "old_mask_where_us": base["median_ms"] * 1e3, "tiles": {}}
        for t, (d, _) in zip(arms, timed[1:], strict=True):
            row["tiles"][f"{t[0]}x{t[1]}"] = {
                "us": d["median_ms"] * 1e3,
                "over_old": _paired(d["window_medians_ms"], base["window_medians_ms"]),
            }
        print(json.dumps(row), flush=True)
        rows.append(row)
Path(out_path).write_text(json.dumps({"dataset": dataset, "rows": rows}, indent=1))
