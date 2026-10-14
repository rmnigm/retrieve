"""C6 ceiling split on the study's arm (controller c3 ruling 2): is SilverTorch's recall ceiling the int8 storage or the IVF
partition? One index per (dataset, n_lists) built by the adapted backend (``official-fork``, seed 0), probed by its own
centroids; recall_oracle@100 over the kept queries, the probed and filter-passing items scored four ways:
  shipped  the adapted module itself (run.quality: Meta's adapted kernels, global item scale)
  global   the same arithmetic in plain torch (per-row int8 query · global-scale item codes): checks the method
  per_row  per-row item int8 scales on the same int32-equivalent path (ST-6)
  fp16     fp16 query · fp16 items over the same probed lists: the IVF ceiling without quantisation
The exact filter mask is ExactAttributeFilter(backend="torch"); no Triton kernel scores or filters. Quality only, no timing.
usage (from a fork tree's evaluation/): [SEED=s] [SUITE=name] ceiling_adapted.py CONFIG_DIR OUT_DIR [GROUP ...]
  SEED (default 0) is the index build's seed; SUITE (default n-ceil-adapted; FULL-RERUN: fr-ceil-adapted) the suite whose clause
  jobs give the inputs."""

import dataclasses
import json
import os
import sys
from pathlib import Path

import torch
from bench import inputs, measure, run
from bench.config import load_matrix
from bench.metrics import accumulate, accumulator, drop_self, finalize
from retrieve import ExactAttributeFilter
from retrieve.indexing.quantize import quantize_int8

GROUPS = {  # name: (dataset config, n_lists, n_probe, sweeps), the campaign-final / A1b ceiling cells
    "arxiv": ("arxiv", 16384, 4096, ["all4", "c0_maincat", "c3_nversions"]),
    "arxiv-synth": ("arxiv-synth", 16384, 4096, ["p001", "p01", "p1"]),
    "goodreads": ("goodreads", 16384, 4096, ["all4", "c1_lang_reverse"]),
    "goodreads-8192": ("goodreads", 8192, 2048, ["c0_genre"]),
    "goodreads-synth": ("goodreads-synth", 16384, 4096, ["p001", "p01", "p1"]),
    "pubmed": ("pubmed", 16384, 4096, ["all5", "c0_mesh", "c3_journal_reverse"]),
    "yfcc10m-synth": ("yfcc10m-synth", 4096, 256, ["p001", "p1"]),
    "laion1m": ("laion1m", 8192, 2048, ["c0_domain", "tags4"]),
    "laion3m": ("laion3m", 16384, 4096, ["c0_domain", "tags4"]),
    "laion30m": ("laion30m", 16384, 4096, ["c0_domain", "tags4"]),
    "laion30m-synth": ("laion30m-synth", 16384, 4096, ["p001", "p01", "p1"]),
}
SUITE = os.environ.get("SUITE", "n-ceil-adapted")
SEED = int(os.environ.get("SEED", "0"))
cfg, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
names = sys.argv[3:] or list(GROUPS)
dev = torch.device("cuda")
BLOCK = 1 << 20
out_dir.mkdir(parents=True, exist_ok=True)


def mask_torch(filt, qa, n):
    m = torch.empty((qa.shape[0], n), dtype=torch.bool, device=dev)
    for b in range(0, n, BLOCK):
        m[:, b : min(n, b + BLOCK)] = filt.evaluate_mask(qa, b, min(n, b + BLOCK))
    return m


for name in names:
    ds, n_lists, n_probe, sweeps = GROUPS[name]
    dim = {"arxiv": 128, "goodreads": 128, "pubmed": 768, "yfcc10m": 192}.get(
        ds.removesuffix("-synth"), 256
    )
    jobs = [
        j
        for j in load_matrix(cfg / f"{ds}.yaml", cfg / "suites.yaml", SUITE)
        if j.dim == dim
    ]
    jobs = [dataclasses.replace(j, seed=SEED) for j in jobs]
    measure.setup(SEED)
    inp = inputs.load_inputs(jobs[0].data, dev, with_filters=True)
    params = {"n_lists": n_lists, "n_probe": n_probe}
    m = None
    filt = ExactAttributeFilter(backend="torch")
    filt.register_index(inp["item_attrs"], clause_is_reverse=inp["clause_is_reverse"])
    for sweep in sweeps:
        job = next(j for j in jobs if j.sweep == sweep and j.filter_kind == "clause")
        assets = run.sweep_assets(job, inp, max(job.ks), dev)
        extra = int(job.self_excluded)  # self-match: score k + 1, drop the own item
        if m is None:
            m = run.build_module(job, inp, assets, 100 + extra, params)
            perm = m.sort_perm
            n = perm.numel()
            codes_r = torch.empty_like(m.item_codes)
            scales_r = torch.empty(n, dtype=torch.float32, device=dev)
            for b in range(0, n, BLOCK):
                c, s = quantize_int8(inp["item_embs"][perm[b : b + BLOCK]].float())
                codes_r[b : b + BLOCK], scales_r[b : b + BLOCK] = c, s
            cluster_of = torch.repeat_interleave(
                torch.arange(m.n_lists, device=dev), m.cluster_offsets.diff()
            )
        m.k = 100 + extra
        shipped = run.quality(m, inp, assets, [100], dev)[0]["oracle"]["recall@100"]
        acc = {v: accumulator([100], dev) for v in ("global", "per_row", "fp16")}
        rows = assets["keep"].nonzero().reshape(-1)
        with torch.inference_mode():
            for s in range(0, rows.numel(), 32):
                sel = rows[s : s + 32]
                orow = assets["oracle_rows"][sel]
                if not bool(orow.any()):
                    continue
                q = inp["queries"][sel].to(dev)
                qa = assets["qa_s"][sel].to(dev)
                probed = torch.zeros(
                    q.shape[0], m.n_lists, dtype=torch.bool, device=dev
                )
                probed.scatter_(1, m._phase1_probe_ids(q), True)
                allowed = probed[:, cluster_of] & mask_torch(filt, qa, n)[:, perm]
                q_codes, q_scales = quantize_int8(q.float())
                q_codes = q_codes.to(torch.float16)
                q16 = q.to(torch.float16)

                def mm(lhs, codes):
                    o = torch.empty((lhs.shape[0], n), dtype=torch.float32, device=dev)
                    for b in range(0, n, BLOCK):
                        e = min(n, b + BLOCK)
                        o[:, b:e] = torch.mm(
                            lhs,
                            codes(b, e).to(torch.float16).t(),
                            out_dtype=torch.float32,
                        )
                    return o

                variants = {
                    "global": lambda: (
                        mm(q_codes, lambda b, e: m.item_codes[b:e])
                        * q_scales[:, None]
                        * m._global_scale_f
                    ),
                    "per_row": lambda: (
                        mm(q_codes, lambda b, e: codes_r[b:e])
                        * q_scales[:, None]
                        * scales_r[None, :]
                    ),
                    "fp16": lambda: mm(q16, lambda b, e: inp["item_embs"][perm[b:e]]),
                }
                idx = orow.nonzero().reshape(-1).to(dev)
                targets = assets["oracle_topk"][sel][orow][:, :100].to(dev)
                own = assets["self_ids"][sel].to(dev) if extra else None
                for v, score in variants.items():
                    sc = score().masked_fill_(~allowed, float("-inf"))
                    top, pos = torch.topk(sc, 100 + extra, dim=1)
                    ids = torch.where(
                        torch.isfinite(top), perm[pos], torch.full_like(pos, -1)
                    )
                    if extra:
                        ids = drop_self(ids, own)
                    accumulate(acc[v], ids.index_select(0, idx), targets, ranked=True)
                    del sc
        res = {
            "dataset": ds,
            "sweep": sweep,
            "params": params,
            "backend": "official-fork",
            "pass_rate": assets["pass_rate"],
            "code_version": measure.code_version(),
            "recall_oracle@100": {
                "shipped": shipped,
                **{v: finalize(a)["recall@100"] for v, a in acc.items()},
            },
        }
        (out_dir / f"ceiling-{ds}-{sweep}-{n_lists}.json").write_text(
            json.dumps(res, indent=1)
        )
        print(json.dumps(res), flush=True)
    m = inp = codes_r = scales_r = None
    torch.cuda.empty_cache()
