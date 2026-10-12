"""C4 on SilverTorch's own bloom (controller ruling, method-diff finding): the adapted bloom index's false-positive rate before any
recheck, per (dataset, sweep, m_bits, k_hash), on the final pass's width grid. The index is built by the adapted code (``official-fork``,
configured width = m_bits bits per doc, build k = search k = k_hash, Meta's other defaults) over the items in their own order (no k-means:
the bloom does not depend on it); its full survivor mask comes from ``bloom_index_search_batch_prepared`` per query. Per query,
FPR = (bloom survivors - exact passes) / (N - exact passes), with the exact pass counts from the sweep's oracle blob; a survivor count below
the exact count (a false negative) would be a bug and is counted. Quality only, no timing, no recheck anywhere.

usage (from a tree's evaluation/): fp_adapted.py SUITES_YAML OUT_CSV"""

import csv
import sys
from pathlib import Path

import torch
from bench import inputs, measure, run
from bench.config import load_matrix
from retrieve.ops import official as official_mod

SWEEPS = {"goodreads": ["c0_genre"], "arxiv": ["all4", "c0_maincat", "c3_nversions"], "pubmed": ["c0_mesh"]}
DIM = {"goodreads": 128, "arxiv": 128, "pubmed": 768}
WIDTHS, KS = [64, 128, 256, 512, 1024, 2048], [3, 5]
suites, out = Path(sys.argv[1]), Path(sys.argv[2])
cfg = official_mod.DEFAULT_CONFIG
dev = torch.device("cuda")
rows_out = []
for ds, sweeps in SWEEPS.items():
    jobs = [j for j in load_matrix(Path("config") / f"{ds}.yaml", suites, "n-bloomwidth") if j.dim == DIM[ds]]
    measure.setup(0)
    inp = inputs.load_inputs(jobs[0].data, dev, with_filters=True)
    n = inp["n_items"]
    attrs = inp["item_attrs"].long().contiguous()
    assets = {sw: run.sweep_assets(next(j for j in jobs if j.sweep == sw), inp, 1000, dev) for sw in sweeps}
    for m_bits in WIDTHS:
        for k in KS:
            index, offsets = official_mod.build_bloom_index(
                attrs, b_multiplier=cfg.b_multiplier, build_k=k, bits_per_doc=m_bits, backend="official-fork"
            )
            for sw in sweeps:
                a = assets[sw]
                rows = a["keep"].nonzero().reshape(-1)
                fpr, ratio, neg = [], [], 0
                with torch.inference_mode():
                    for s in range(0, rows.numel(), 64):
                        sel = rows[s : s + 64]
                        qa = a["qa_s"][sel].to(dev)
                        plans = official_mod.prepare_plans(
                            official_mod.queries_to_expressions(qa), cfg.n_stored_hashes, cfg.max_sub_queries, qa.shape[1], dev,
                            slots=qa.shape[0], backend="official-fork",
                        )  # fmt: skip
                        mask = official_mod.bloom_full_mask(
                            index, offsets, plans, k, cfg.n_stored_hashes, return_bool_mask=True, backend="official-fork"
                        )[:, :n]
                        surv = mask.sum(1).cpu().double()
                        exact = a["blob"]["pass_counts"][sel].double()
                        neg += int((surv < exact).sum())
                        fpr.append((surv - exact) / (n - exact).clamp_min(1))
                        ratio.append(surv / exact.clamp_min(1))
                fpr, ratio = torch.cat(fpr), torch.cat(ratio)
                rows_out.append(
                    dict(
                        dataset=ds, sweep=sw, m_bits=m_bits, k_hash=k, n_queries=int(fpr.numel()),
                        fp_rate_mean=float(fpr.mean()), fp_rate_median=float(fpr.median()), fp_rate_max=float(fpr.max()),
                        survivors_over_exact_median=float(ratio.median()), false_negative_rows=neg,
                        index_mib=index.numel() * 8 / 2**20, code_version=measure.code_version(),
                    )
                )  # fmt: skip
                print(rows_out[-1], flush=True)
            del index, offsets
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    w.writeheader()
    w.writerows(rows_out)
