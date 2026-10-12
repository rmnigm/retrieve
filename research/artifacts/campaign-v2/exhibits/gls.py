"""Idea #2, GLS alignment vs recall (README.md § gls): does the filter's alignment with a query's own
neighbourhood explain IVF recall better than the pass rate?

usage: gls.py OUT TREE [TREE ...]

Per query, l_q = share of its K = 100 unfiltered nearest neighbours that pass its filter. Every unfiltered
top-K item that passes also lies in the filtered top-K, so l_q = |unfiltered top-K ∩ filtered top-K| / K,
read from two oracle blobs (the synth `p1` sweep = unfiltered, same queries and item embeddings; the real
sweep's blob). GLS_q = l_q / p_q with p_q = pass_count / N (arXiv 2602.11443). Blob pairs are chosen by
top-1 agreement (same encoder), rows checked against the sidecar's pass counts. Oracle blobs are read
from the datasets' gt dirs, read-only.
"""

import collections
import csv
import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from router import TARGET, curves, ivf_latency, oracle, per_query, table  # noqa: E402

from load import load  # noqa: E402

K = 100
GT = {"goodreads": "/data/goodreads-work-id/gt_d128", "arxiv": "/data/arxiv-papers/gt_d128"}
SANITY = ("goodreads", "p01")  # uniform synth sweep: GLS must be ~1


def blobs(gt, sweep):
    out = {}
    for f in glob.glob(f"{gt}/**/oracle_v4_{sweep}_*.pt", recursive=True):
        out.setdefault(os.path.basename(f), f)  # one file per fingerprint
    return list(out.values())


def load_blob(f):
    b = torch.load(f, map_location="cpu", weights_only=False)
    return {k: b[k] for k in ("topk", "pass_counts", "n_items", "k_gt")}


def local_pass(unf, flt, k=K):
    """|unfiltered top-k ∩ filtered top-k| / k per row (numpy, rows aligned)."""
    u, f = unf[:, :k], flt[:, :k]
    out = np.empty(len(u))
    for i in range(len(u)):
        out[i] = np.intersect1d(u[i], f[i][f[i] >= 0], assume_unique=False).size / k
    return out


def pick_pair(gt, sweep):
    """The filtered blob (k_gt >= K) and unfiltered `p1` blob with the highest top-1 agreement on rows
    where the unfiltered top-1 passes: a same-encoder pair agrees ~always, a mixed one ~never."""
    best = None
    for fu in blobs(gt, "p1"):
        u = load_blob(fu)
        for ff in blobs(gt, sweep):
            b = load_blob(ff)
            if b["k_gt"] < K or b["n_items"] != u["n_items"] or len(b["topk"]) != len(u["topk"]):
                continue
            ut, ft = u["topk"].numpy(), b["topk"].numpy()
            inF = np.array([ut[i, 0] in set(ft[i, :K]) for i in range(len(ut))])
            agree = float((ft[inF, 0] == ut[inF, 0]).mean()) if inF.any() else 0.0
            if best is None or agree > best[0]:
                best = (agree, fu, ff, ut, ft, b["pass_counts"].numpy(), b["n_items"])
    return best


def spearman(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    trees = [Path(t) for t in sys.argv[2:]]
    recs = load(trees)
    pq = per_query(recs, trees)
    pq = {
        k: v for k, v in pq.items() if k[0] in GT and not (k[0] == "goodreads" and k[1] == "bloom")
    }
    rows, prov, feats = [], [], {}
    # sanity: a uniform synth sweep
    sp = pick_pair(GT[SANITY[0]], SANITY[1])
    lq = local_pass(sp[3], sp[4])
    p = sp[5] / sp[6]
    prov.append(
        (SANITY[0], SANITY[1], os.path.basename(sp[1]), os.path.basename(sp[2]), round(sp[0], 4))
    )
    sanity = (
        float(np.median(lq / p)),
        float(np.percentile(lq / p, 5)),
        float(np.percentile(lq / p, 95)),
    )
    for (ds, fk, sw), d in sorted(pq.items()):
        pair = pick_pair(GT[ds], sw)
        if pair is None:
            continue
        agree, fu, ff, ut, ft, pc, n = pair
        prov.append((ds, sw, os.path.basename(fu), os.path.basename(ff), round(agree, 4)))
        lq = local_pass(ut, ft)
        # sidecar rows are the kept query indices; check alignment via the pass counts
        z_rows = None
        for r in recs:
            if (
                r["dataset"] == ds
                and r.get("per_query")
                and r["sweep"] == sw
                and r["algo"] == "silvertorch"
            ):
                z_rows = np.load(next(t for t in trees if t.name == r["_tree"]) / r["per_query"])[
                    "rows"
                ]
                break
        assert np.allclose(pc[z_rows] / n, d["p"]), (
            ds,
            sw,
            "pass counts disagree with the sidecar",
        )
        l_rows = lq[z_rows]
        gls = l_rows / np.maximum(d["p"], 1e-12)
        feats[(ds, fk, sw)] = l_rows
        rec = d["ivf"]
        rows.append(
            (
                ds,
                fk,
                sw,
                len(rec),
                round(float(np.median(d["p"])), 5),
                round(float(np.median(l_rows)), 4),
                round(float(np.median(gls)), 3),
                round(float(np.nanmean(rec)), 4),
                round(spearman(rec, np.log10(np.maximum(d["p"], 1e-9))), 3),
                round(spearman(rec, np.log10(np.maximum(gls, 1e-9))), 3),
                round(spearman(rec, l_rows), 3),
            )
        )
    # pooled per dataset + router with l_q as the feature
    pooled, router_rows = [], []
    cur, ivl = curves(recs), ivf_latency(recs)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, ds in enumerate(sorted({k[0] for k in feats})):
        keys = [k for k in sorted(pq) if k[0] == ds and k in feats]
        rec = np.concatenate([pq[k]["ivf"] for k in keys])
        p = np.concatenate([pq[k]["p"] for k in keys])
        lq = np.concatenate([feats[k] for k in keys])
        gls = lq / np.maximum(p, 1e-12)
        pooled.append(
            (
                ds,
                len(rec),
                round(spearman(rec, np.log10(np.maximum(p, 1e-9))), 3),
                round(spearman(rec, np.log10(np.maximum(gls, 1e-9))), 3),
                round(spearman(rec, lq), 3),
            )
        )
        for j, (x, name) in enumerate(
            ((p, "pass rate p_q"), (lq, "local pass rate l_q (share of 100 NN passing)"))
        ):
            ax = axes[i][j]
            edges = np.unique(np.quantile(x, np.linspace(0, 1, 21)))
            idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
            mx = [np.nanmean(x[idx == b]) for b in range(len(edges) - 1)]
            my = [np.nanmean(rec[idx == b]) for b in range(len(edges) - 1)]
            ax.scatter(x[::7], rec[::7], s=2, alpha=0.15)
            ax.plot(mx, my, "o-", color="k", ms=3, label="ventile means")
            ax.set_xscale("log" if j == 0 else "linear")
            ax.set_xlabel(name)
            ax.set_ylabel("IVF recall_oracle@100 (n_probe 24)")
            ax.set_title(
                f"{ds}: Spearman {spearman(rec, np.log10(np.maximum(x, 1e-9))) if j == 0 else spearman(rec, x):.2f}",
                fontsize=9,
            )
            ax.legend(fontsize=6)
        for bs in (1, 16):
            if (ds, bs) not in ivl:
                continue
            lst = [(k[1], k[2], pq[k]) for k in keys]
            t = table(lst, cur[ds], ivl[(ds, bs)][0], bs)
            t["l"] = np.concatenate(
                [feats[k][~np.isnan(pq[k]["ivf"])] for k in keys]
            )  # table()'s row filter
            for feat in ("p", "l"):
                best = None
                for thr in sorted(set(np.quantile(t[feat], np.linspace(0, 1, 201)))) + [2.0]:
                    ex = t[feat] < thr
                    r = float(np.where(ex, t["re"], t["ri"]).mean())
                    lat = float(np.where(ex, t["le"], t["li"]).mean())
                    if r >= TARGET and (best is None or lat < best[2]):
                        best = (thr, r, lat, float(ex.mean()))
                router_rows.append(
                    (
                        ds,
                        bs,
                        f"route exact if {feat}_q < t",
                        f"{best[0]:.3g}",
                        round(best[1], 4),
                        round(best[2], 4),
                        round(best[3], 3),
                    )
                )
            o = oracle(t, TARGET)
            router_rows.append(
                (ds, bs, "oracle", "", round(o[0], 4), round(o[1], 4), round(o[2], 3))
            )
            router_rows.append(
                (
                    ds,
                    bs,
                    "all-exact",
                    "",
                    round(float(t["re"].mean()), 4),
                    round(float(t["le"].mean()), 4),
                    1.0,
                )
            )
    fig.suptitle("IVF recall vs pass rate and vs local pass rate (real sweeps, NOT CITABLE)")
    fig.tight_layout()
    fig.savefig(out / "gls-recall.png", dpi=120)
    for name, hdr, data in (
        (
            "gls-sweeps.csv",
            [
                "dataset",
                "filter_kind",
                "sweep",
                "queries",
                "median_p",
                "median_l",
                "median_gls",
                "ivf_recall",
                "spearman_recall_logp",
                "spearman_recall_loggls",
                "spearman_recall_l",
            ],
            rows,
        ),
        (
            "gls-pooled.csv",
            ["dataset", "queries", "spearman_logp", "spearman_loggls", "spearman_l"],
            pooled,
        ),
        (
            "gls-router.csv",
            ["dataset", "bs", "policy", "t", "recall@100", "latency_ms", "share_exact"],
            router_rows,
        ),
        (
            "gls-blobs.csv",
            ["dataset", "sweep", "unfiltered_blob", "filtered_blob", "top1_agreement"],
            prov,
        ),
    ):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hdr)
            w.writerows(data)
    print("sanity (uniform synth p01) GLS median / 5-95 %:", [round(x, 3) for x in sanity])
    print(collections.Counter(r[0] for r in rows))


if __name__ == "__main__":
    main()
