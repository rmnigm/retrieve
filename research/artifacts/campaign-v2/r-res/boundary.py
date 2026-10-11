"""R-RES, CPU: where the freeze's golden cells could differ from their golden JSONs.

    python boundary.py FREEZE_DIR

FREEZE_DIR is the gate-3 output (``filter/*.jsonl``, ``filter/*.perquery/``, ``_parity/``).
Per cell (seed 0, the golden's params): the spilled top-``k_max`` ids/scores are paired with
the cached v4 oracle whose per-query ``recall_oracle@k`` they reproduce exactly, then, at each
k, the rows whose rank-k boundary sits on a score tie (``scores[k-1] == scores[k]``) and the
recall@k those ties could swing (members of the tie class inside vs outside the oracle top-k)."""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

CELLS = {  # (dataset, algo, params) -> oracle glob
    (
        "arxiv",
        "silvertorch",
        '{"n_probe": 24}',
    ): "/data/arxiv-papers/gt_d128/oracle_v4_c0_maincat_*.pt",
    (
        "goodreads",
        "linr_v2",
        "{}",
    ): "/data/goodreads-work-id/gt_d128/oracle_v4_c0_genre_*.pt",
    (
        "goodreads",
        "linr_v3",
        "{}",
    ): "/data/goodreads-work-id/gt_d128/oracle_v4_c0_genre_*.pt",
    (
        "goodreads",
        "linr_v1_filter_mask",
        "{}",
    ): "/data/goodreads-work-id/gt_d128/oracle_v4_c0_genre_*.pt",
}


def per_row_recall(ids: np.ndarray, gt: np.ndarray, k: int) -> np.ndarray:
    t = gt[:, :k]
    hits = np.array([np.isin(a[:k], b[b >= 0]).sum() for a, b in zip(ids, t)])
    return hits / np.maximum(np.minimum((t >= 0).sum(1), k), 1)


def main(root: Path) -> None:
    recs = {}
    for f in (root / "filter").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["seed"] == 0 and r["backend"] == "triton":
                recs[
                    (r["dataset"], r["algo"], json.dumps(r["params"], sort_keys=True))
                ] = r
    for key, oglob in CELLS.items():
        r = recs[key]
        pq = np.load(root / r["per_query"])
        spills = glob.glob(str(root / "_parity" / f"{key[0]}-d128_{key[1]}" / "*.npz"))
        oracles = [torch.load(p, weights_only=False) for p in sorted(glob.glob(oglob))]
        oracles = [o for o in oracles if o["k_gt"] >= r["k_max"]]
        rows = pq["rows"]
        match = None
        for sp in spills:
            z = np.load(sp)
            for o in oracles:
                gt = o["topk"].numpy()[rows]
                if np.array_equal(
                    per_row_recall(z["ids"], gt, 100).astype(np.float32),
                    pq["recall_oracle@100"],
                ):
                    match = (sp, z, o, gt)
        assert match, key
        sp, z, o, gt = match
        ids, sc = z["ids"], z["scores"]
        print(
            f"\n## {key[0]} {key[1]} {key[2]}  spill {Path(sp).name}  oracle fp {o['fingerprint'][:16]}"
        )
        for k in r["ks"]:
            rec = per_row_recall(ids, gt, k)
            assert np.array_equal(rec.astype(np.float32), pq[f"recall_oracle@{k}"]), k
            line = f"k={k}: mean recall {rec.mean():.10f} (record {r['quality']['oracle'][f'recall@{k}']:.10f})"
            if k < ids.shape[1]:
                tie = sc[:, k - 1] == sc[:, k]
                swing_lo = swing_hi = 0
                for i in np.nonzero(tie)[0]:
                    s = sc[i, k - 1]
                    cls = np.nonzero(sc[i] == s)[0]
                    inside = cls[cls < k]
                    member = np.isin(ids[i, cls], gt[i, :k])
                    m_in, n_in = member[cls < k].sum(), len(inside)
                    best, worst = (
                        min(member.sum(), n_in),
                        max(0, n_in - (~member).sum()),
                    )
                    swing_lo += worst - m_in
                    swing_hi += best - m_in
                    if cls[-1] == ids.shape[1] - 1:
                        line += f" [row {i}: tie class runs past k_max]"
                line += (
                    f"; rows with a tie at the k boundary {int(tie.sum())}, recall@{k} those ties"
                    f" can move: {swing_lo:+d}..{swing_hi:+d} hits"
                )
            print(line)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
