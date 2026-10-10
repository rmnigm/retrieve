"""PubMed diagnostic (a): why the exact arms (V1 filter-mask, V2; triton clause) score recall_oracle@100 0.996-0.999, not 1.

Runs the real `filter` suite's V1 and V2 cells (seed 0, k_max 1000 as in F3) through the harness's own quality pass, checks that
the recall reproduces the F3 record, then classifies every item of the oracle top-100 missing from the arm's top-100 against the
arm's lowest-scored intruder `a` (the last item of the arm's top-100 that is not in the oracle's: the one m should have displaced), in the arm's own scores (V1's scorer: fp16 query x fp16 items, fp32 out, cuBLAS) and in exact fp32 (the
oracle's precision):

  arm_tie      arm score(m) == arm score(a): a tie at rank k in the fp16-input scores, broken by id / kernel order
  fp16_order   arm score(m) <  arm score(a) and fp32(m) > fp32(a): fp16 rounding of the inputs swaps the pair
  oracle_tie   fp32(m) == fp32(a): a tie at rank k in the oracle's fp32 scores, broken to the lowest id
  other        anything else (arm score(m) > arm score(a) means the arm dropped a better passing item: a defect)

V2's ids are also compared with V1's per row. Writes OUT.json (counts per sweep / arm) and OUT.csv (one line per miss).

    python exact_miss.py CONFIG_DIR OUT_STEM [SWEEP ...]   (from evaluation/, PYTHONPATH to the tree under test)
"""

import csv
import json
import sys
from pathlib import Path

import torch
from bench import inputs, measure, run
from bench.config import load_matrix

cfg, stem = Path(sys.argv[1]), sys.argv[2]
sweeps = sys.argv[3:] or ["c0_mesh", "all5", "c3_journal_reverse"]
K = 100
device = torch.device("cuda")
measure.setup(0)
jobs = load_matrix(cfg / "pubmed.yaml", cfg / "suites.yaml", "filter", algos=["linr_v1_filter_mask", "linr_v2"],
                   backends=["triton"], filter_kinds=["clause"], sweeps=sweeps, seeds=[0])  # fmt: skip
inp = inputs.load_inputs(jobs[0].data, device, with_filters=True)
items = inp["item_embs"]
summary, lines = {}, []
for sweep in sweeps:
    ids_by, scorer = {}, None
    for job in sorted([j for j in jobs if j.sweep == sweep], key=lambda j: j.algo != "linr_v1_filter_mask"):
        k_max = max(job.ks)
        assets = run.sweep_assets(job, inp, k_max, device)
        params = run.resolve_pool(job.cells()[0], assets)
        module = run.build_module(job, inp, assets, k_max, params).to(device)
        metrics, _, ids, scores = run.quality(module, inp, assets, job.ks, device)
        rows = assets["keep"].nonzero().reshape(-1)
        orow = assets["oracle_rows"][rows]
        rows, ids, scores = rows[orow], ids[orow.to(ids.device)], scores[orow.to(scores.device)]
        ids_by[job.algo] = (rows, ids[:, :K])
        oracle = assets["blob"]["topk"][rows, :K].to(device)
        got = ids[:, :K].to(device)
        valid = oracle != -1
        hit = (oracle.unsqueeze(2) == got.unsqueeze(1)).any(2) & valid
        miss = valid & ~hit
        recall = (hit.sum(1).float() / valid.sum(1).clamp_min(1)).mean().item()
        counts = {"arm_tie": 0, "fp16_order": 0, "oracle_tie": 0, "other": 0}
        if job.algo == "linr_v1_filter_mask":  # V2's misses are scored with V1's scorer (same fp16 inputs; ids compared below)
            scorer = module.idx
        for r in miss.any(1).nonzero().reshape(-1).tolist():
            u = int(rows[r])
            q = inp["queries"][u : u + 1].to(device)
            arm = scorer.score(q)[0]
            a = int(got[r][~(got[r].unsqueeze(1) == oracle[r].unsqueeze(0)).any(1)][-1])
            for m in oracle[r][miss[r]].tolist():
                s_arm_m, s_arm_a = arm[m].item(), arm[a].item()
                s32_m, s32_a = (q[0] @ items[m]).item(), (q[0] @ items[a]).item()
                if s_arm_m == s_arm_a:
                    c = "arm_tie"
                elif s_arm_m < s_arm_a and s32_m > s32_a:
                    c = "fp16_order"
                elif s32_m == s32_a:
                    c = "oracle_tie"
                else:
                    c = "other"
                counts[c] += 1
                lines.append([sweep, job.algo, u, m, a, s_arm_m, s_arm_a, s32_m, s32_a, c])
        n_miss = int(miss.sum())
        summary[f"{sweep}/{job.algo}"] = {
            "recall_oracle@100_record": metrics["oracle"]["recall@100"], "recall_recomputed": recall,
            "rows": int(rows.numel()), "rows_with_miss": int(miss.any(1).sum()), "missed_items": n_miss, **counts,
        }  # fmt: skip
        print(sweep, job.algo, summary[f"{sweep}/{job.algo}"], flush=True)
        del module
        run._release()
    del scorer
    run._release()
    (r1, i1), (r2, i2) = ids_by["linr_v1_filter_mask"], ids_by["linr_v2"]
    same_sets = sum(set(a.tolist()) == set(b.tolist()) for a, b in zip(i1.cpu(), i2.cpu()))
    summary[f"{sweep}/v1_vs_v2"] = {"rows_equal": bool(torch.equal(r1, r2)), "ids_equal": bool(torch.equal(i1.cpu(), i2.cpu())),
                                    "same_id_sets": same_sets, "rows": int(r1.numel())}  # fmt: skip
    print(sweep, "v1 vs v2", summary[f"{sweep}/v1_vs_v2"], flush=True)
Path(f"{stem}.json").write_text(json.dumps(summary, indent=1))
with open(f"{stem}.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["sweep", "algo", "user_row", "missed_item", "arm_kth_item", "arm_score_m", "arm_score_a", "fp32_m", "fp32_a", "class"])
    w.writerows(lines)
