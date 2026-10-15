"""C6, local pass rate on the study's arm (controller c3 ruling 3): does IVF recall follow a query's local pass rate l_q
better than its global pass rate p_q? Per query, l_q = |unfiltered top-100 ∩ filtered top-100| / 100 (the synth `p1`
oracle blob is the unfiltered one: same queries, same items), p_q = pass_count / N; recall@100 from the adapted
code's per-query sidecars (FULL-RERUN curve leg fr/curve-d128-pode-s0, official-fork, bloom with the recheck off and clause)
at every curve point of the smallest n_lists.
Spearman of recall against l_q and against p_q per (dataset, sweep, n_probe). The oracle blob of a sweep is the one
whose pass counts equal the sidecar's. CPU only.

usage: gls_adapted.py FR OUT_DIR   (FR: one bench-fetched directory per fr/ subtree)"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HUB, OUT = Path(sys.argv[1]), Path(sys.argv[2])
GT = {
    "goodreads": "/data/goodreads-work-id/gt_d128",
    "arxiv": "/data/arxiv-papers/gt_d128",
}
SWEEPS = {
    "goodreads": ["c0_genre", "c1_lang_reverse", "all4"],
    "arxiv": ["c0_maincat", "c3_nversions", "all4"],
}
K = 100

leg = HUB / "curve-d128-pode-s0"
df = pd.read_parquet(leg / "results.parquet")
df = df[(df.backend == "official-fork") & (df.dim == 128) & (df.seed == 0)]
p = df.params.map(json.loads)
df = df.assign(
    n_lists=p.map(lambda d: d["n_lists"]), n_probe=p.map(lambda d: d["n_probe"])
)


def blobs(gt, sweep):
    return [
        torch.load(f, map_location="cpu", weights_only=False)
        for f in sorted(glob.glob(f"{gt}/**/oracle_v4_{sweep}_*.pt", recursive=True))
    ]


rows_out = []
for ds, sweeps in SWEEPS.items():
    unf = [b for b in blobs(GT[ds], "p1") if b["k_gt"] >= K]
    for (sw, kind), x in df[df.dataset == ds].groupby(["sweep", "filter_kind"]):
        if sw not in sweeps:
            continue
        x = x[x.n_lists == x.n_lists.min()].drop_duplicates("n_probe")
        for r in x.itertuples():
            z = np.load(leg / r.per_query)
            rows = torch.from_numpy(z["rows"].astype(np.int64))
            filt = next(
                (
                    b
                    for b in blobs(GT[ds], sw)
                    if b["k_gt"] >= K
                    and torch.equal(
                        b["pass_counts"][rows], torch.from_numpy(z["pass_count"])
                    )
                ),
                None,
            )
            if (
                filt is None
            ):  # this box's oracle blobs are not the run's (pod e built arXiv's)
                print(
                    f"skip {ds} {sw}: no oracle blob matches the sidecar's pass counts",
                    flush=True,
                )
                break
            ft = filt["topk"][rows, :K]
            best = max(
                unf,
                key=lambda b: float((b["topk"][rows, :1] == ft[:, :1]).float().mean()),
            )
            ut = best["topk"][rows, :K]
            lq = (ut.unsqueeze(2) == ft.unsqueeze(1)).any(2).float().mean(1).numpy()
            pq = z["pass_count"] / r.n_items
            rec = z["recall_oracle@100"]
            rows_out.append(
                dict(
                    dataset=ds,
                    sweep=sw,
                    filter_kind=kind,
                    n_lists=r.n_lists,
                    n_probe=r.n_probe,
                    n_queries=len(rec),
                    recall_mean=float(rec.mean()),
                    spearman_lq=float(spearmanr(rec, lq)[0]),
                    spearman_pq=float(spearmanr(rec, pq)[0]),
                )
            )
            print(rows_out[-1], flush=True)
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows_out).to_csv(OUT / "gls_adapted.csv", index=False)
