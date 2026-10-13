"""Untimed grading pass for SilverTorch's bloom (user's bloom-reporting ruling, 2026-10-13): at given (dataset, sweep, n_lists, n_probe) points,
count the adapted bloom's false positives on the docs each query probes. The index is built by the adapted code (``official-fork``, seed 0,
the suites' bloom width, recheck off) as in the timed legs, so its probes are theirs. Per query, over the probed clusters: survivors = docs
SilverTorch's own bloom passes (its full survivor mask, ``bloom_index_search_batch_prepared``, restricted to the probed clusters); exact = docs
the exact filter passes there; rejects = survivors - exact (what a recheck would drop; the bloom has no false negatives, counted if any).
Recorded per point: rejects / survivors and rejects / (probed - exact), mean over queries. Recall with the recheck off and on comes from the
records (B1 and the clause curves), not from here. Nothing here is timed. The probed clusters are the fp32 centroid top-n_probe, which can differ
at ties from the kernel's own probe selection. The bloom and exact masks are computed once per query chunk for every n_probe.

usage (from a tree's evaluation/): [SEED=s] grade_bloom.py CONFIG_DIR SUITE POINTS_JSON OUT_CSV
  SEED (default 0): the index build's seed, as the timed leg's job seed
  POINTS_JSON: [[dataset, sweep, n_lists, [n_probe, ...]], ...]"""

import csv
import dataclasses
import json
import os
import sys
from pathlib import Path

import torch
from bench import inputs, measure, run
from bench.config import load_matrix
from retrieve import ExactAttributeFilter
from retrieve.ops import official as official_mod

DIM = {"goodreads": 128, "arxiv": 128, "pubmed": 768}
cfg_dir, suite, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[4])
SEED = int(os.environ.get("SEED", "0"))
points = json.loads(Path(sys.argv[3]).read_text())
dev = torch.device("cuda")
ocfg = official_mod.DEFAULT_CONFIG
rows_out = []
by_ds = {}
for ds, sw, nl, probes in points:
    by_ds.setdefault(ds, []).append((sw, nl, probes))
for ds, items in by_ds.items():
    dim = DIM.get(ds.removesuffix("-synth"), 256)
    jobs = [
        dataclasses.replace(j, seed=SEED)
        for j in load_matrix(cfg_dir / f"{ds}.yaml", cfg_dir / "suites.yaml", suite)
        if j.dim == dim and j.filter_kind == "bloom"
    ]
    measure.setup(SEED)
    inp = inputs.load_inputs(jobs[0].data, dev, with_filters=True)
    exact = ExactAttributeFilter(backend="torch")
    exact.register_index(inp["item_attrs"], clause_is_reverse=inp["clause_is_reverse"])
    for nl in sorted({nl for _, nl, _ in items}):
        job0 = next(j for j in jobs if j.sweep == items[0][0])
        a0 = run.sweep_assets(job0, inp, max(job0.ks), dev)
        m = run.build_module(job0, inp, a0, 100, {"n_lists": nl, "n_probe": 1, "recheck": False})
        n, perm = m.sort_perm.numel(), m.sort_perm
        cluster_of = torch.repeat_interleave(torch.arange(m.n_lists, device=dev), m.cluster_offsets.diff())
        for sw, nl2, probes in items:
            if nl2 != nl:
                continue
            job = next(j for j in jobs if j.sweep == sw)
            a = run.sweep_assets(job, inp, max(job.ks), dev)
            sel_rows = a["keep"].nonzero().reshape(-1)
            acc = {npb: {"survivors": [], "exact": [], "probed": [], "neg": 0} for npb in probes}
            with torch.inference_mode():
                for s in range(0, sel_rows.numel(), 32):
                    sel = sel_rows[s : s + 32]
                    q = inp["queries"][sel].to(dev)
                    qa = a["qa_s"][sel].to(dev)
                    top = (q @ m.centroids.t()).topk(max(probes), dim=1).indices
                    plans = official_mod.prepare_plans(
                        official_mod.queries_to_expressions(qa), ocfg.n_stored_hashes, ocfg.max_sub_queries, qa.shape[1], dev,
                        slots=qa.shape[0], backend="official-fork",
                    )  # fmt: skip
                    bloom = official_mod.bloom_full_mask(
                        m.bloom_index, m.bundle_b_offsets, plans, m.k_hash, ocfg.n_stored_hashes, return_bool_mask=True, backend="official-fork"
                    )[:, :n]  # fmt: skip
                    ex = torch.cat([exact.evaluate_mask(qa, b, min(n, b + (1 << 20))) for b in range(0, n, 1 << 20)], 1)[:, perm]
                    for npb in probes:
                        probed = torch.zeros(q.shape[0], m.n_lists, dtype=torch.bool, device=dev)
                        probed.scatter_(1, top[:, :npb], True)
                        pdoc = probed[:, cluster_of]
                        surv, exc = (bloom & pdoc).sum(1), (ex & pdoc).sum(1)
                        acc[npb]["neg"] += int((ex & pdoc & ~bloom).any(1).sum())
                        acc[npb]["survivors"].append(surv.cpu())
                        acc[npb]["exact"].append(exc.cpu())
                        acc[npb]["probed"].append(pdoc.sum(1).cpu())
                    del bloom, ex
            for npb in probes:
                surv, exc, prb = (torch.cat(acc[npb][k]).double() for k in ("survivors", "exact", "probed"))
                rej = surv - exc
                rows_out.append(
                    dict(
                        dataset=ds, sweep=sw, n_lists=nl, n_probe=npb, n_queries=int(surv.numel()),
                        rejects_over_survivors=float((rej / surv.clamp_min(1)).mean()),
                        rejects_over_exact_failing=float((rej / (prb - exc).clamp_min(1)).mean()),
                        survivors_mean=float(surv.mean()), exact_mean=float(exc.mean()), false_negative_rows=acc[npb]["neg"],
                        m_bits=m.m_bits, k_hash=m.k_hash, code_version=measure.code_version(),
                    )
                )  # fmt: skip
                print(rows_out[-1], flush=True)
        m = None
        torch.cuda.empty_cache()
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    w.writeheader()
    w.writerows(rows_out)
