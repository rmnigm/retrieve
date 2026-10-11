"""D1-G's fairer batching companion: per failing `bs16 < 16 × bs1` entry, each timed bs 16 batch against the sum of
the bs 1 times of the same 16 queries, from the stored per-call samples (no rerun).

Why it can be rebuilt: `inputs.query_pool` draws `[4096, bs]` rows with one `manual_seed(seed)` CPU generator per bs, and
a CPU `randint` draw's prefix is the smaller draw, so bs 16 pool batch j holds bs 1 pool entries 16j..16j+15 (j < 256).
`measure.latency_group` advances each arm's round-robin counter by 1 (eager: the sync-debug call) + 50 warmup + 20
calibration calls, then n per window, and keeps the median window's per-call ms. So sample i of the kept window ran
pool entry (pre + pick·n + i) mod 4096. The check: bs 1 per-call ms against the query's pass count correlates at that
offset only (`offset_corr` vs `shifted_corr_max_abs`). Only the median window's calls are stored, so a bs 16 batch is
matched only where the kept bs 1 window holds all 16 of its pool entries. For every entry, the expectation form: both
pools are uniform draws from the same kept queries, so E[sum of 16 bs 1 calls] = 16 × mean bs 1 (`bs16_mean_over_16_bs1_mean`).

usage: batch_vs_sum.py MERGED_TREE CHECKS.csv OUT.csv
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "evaluation"))
from bench import records  # noqa: E402

N_POOL, WARMUP, CALIB = 4096, 50, 20


def pool_rows(n_kept_idx: np.ndarray, bs: int, seed: int) -> np.ndarray:
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randint(0, len(n_kept_idx), (N_POOL, bs), generator=g).reshape(-1).numpy()
    return n_kept_idx[r].reshape(N_POOL, bs)


def window_start(entry: dict, n: int) -> int:
    med = entry["window_medians_ms"]
    pick = sorted(range(len(med)), key=med.__getitem__)[len(med) // 2]
    pre = (1 if entry["mode"] == "eager" else 0) + WARMUP + CALIB
    return pre + pick * n


def samples_for(
    tree: Path, r: dict, k: int, bs: int, mode: str, median: float
) -> list[float]:
    key = {f: r[f] for f in ("dataset", "dim", "inputs", "suite", "filter_kind", "sweep", "algo", "backend", "params",
                             "seed")}  # fmt: skip
    path = records.samples_path(tree / r["suite"] / f"{r['dataset']}-d{r['dim']}.jsonl")
    hits = []
    for line in path.read_text().splitlines():
        s = json.loads(line)
        if (
            s["k"] == k
            and s["bs"] == bs
            and s["mode"] == mode
            and all(s.get(f) == v for f, v in key.items())
        ):
            hits.append(s["ms"])
    hits = [
        h
        for h in hits
        if abs(float(np.median(h)) - median) < 1e-9 * max(1.0, median) + 1e-6
    ]
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} sample lines match {key} k{k} bs{bs} {mode}")
    return hits[0]


def main():
    tree, checks, out = map(Path, sys.argv[1:4])
    failing = [
        row["where"]
        for row in csv.DictReader(open(checks))
        if row["check"] == "batching"
    ]
    recs = records.latest(tree)
    rows = []
    for where in failing:
        head, k, mode = where.rsplit(" ", 2)
        k = int(k[1:])
        ds, suite, fk, sweep, arm, backend_params, seed = head.split(":", 1)[1].split(
            "/"
        )
        seed = int(seed[1:])
        algo = {"V1": "linr_v1_filter_mask", "V2": "linr_v2", "V3": "linr_v3"}[arm]
        backend = backend_params.split(" ")[0]
        (r,) = [x for x in recs if (x["dataset"], x["suite"], x["filter_kind"], x["sweep"], x["algo"], x["backend"],
                                    x["seed"]) == (ds, suite, fk, sweep, algo, backend, seed)
                and (" " not in backend_params) == (not x["params"])]  # fmt: skip
        e = {(x["bs"], x["k"], x["mode"]): x for x in r["perf"]}
        e1, e16 = e[(1, k, mode)], e[(16, k, mode)]
        s1 = samples_for(tree, r, k, 1, mode, e1["median_ms"])
        s16 = samples_for(tree, r, k, 16, mode, e16["median_ms"])
        a1, a16 = window_start(e1, len(s1)), window_start(e16, len(s16))
        by_pool1 = {(a1 + i) % N_POOL: t for i, t in enumerate(s1)}
        # offset check: bs 1 ms vs the query's pass count, at the reconstructed offset and at shifted ones
        pq = np.load(tree / r["per_query"])
        pass_of = dict(zip(pq["rows"].tolist(), pq["pass_count"].tolist(), strict=True))
        keep = np.array(sorted(pass_of))
        p1 = pool_rows(keep, 1, seed)[:, 0]

        def corr(shift: int) -> float:
            x = [pass_of[int(p1[(a1 + shift + i) % N_POOL])] for i in range(len(s1))]
            return float(np.corrcoef(x, s1)[0, 1]) if np.std(x) > 0 else float("nan")

        c0 = corr(0)
        c_shift = max((abs(corr(s)) for s in (1, 7, 100, 1000)), default=float("nan"))
        ratios, n_match = [], 0
        for i, t16 in enumerate(s16):
            j = (a16 + i) % N_POOL
            if j >= N_POOL // 16:
                continue
            members = [by_pool1.get(16 * j + m) for m in range(16)]
            if None in members:
                continue
            n_match += 1
            ratios.append(t16 / sum(members))
        q = np.quantile(ratios, [0.0, 0.5, 1.0]) if ratios else [float("nan")] * 3
        rows.append({
            "entry": where, "bs1_median_ms": round(e1["median_ms"], 4), "bs16_median_ms": round(e16["median_ms"], 4),
            "bs16_over_bs1_median": round(e16["median_ms"] / e1["median_ms"], 2),
            "bs16_mean_over_16_bs1_mean": round(float(np.mean(s16)) / (16 * float(np.mean(s1))), 3),
            "bs1_window_pool": f"{a1 % N_POOL}..{(a1 + len(s1) - 1) % N_POOL}",
            "bs16_window_batches": f"{a16 % N_POOL}..{(a16 + len(s16) - 1) % N_POOL}",
            "matched_batches": n_match, "batch_over_sum16_min": round(float(q[0]), 3),
            "batch_over_sum16_median": round(float(q[1]), 3), "batch_over_sum16_max": round(float(q[2]), 3), "offset_corr": round(c0, 3), "shifted_corr_max_abs": round(c_shift, 3),
        })  # fmt: skip
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
