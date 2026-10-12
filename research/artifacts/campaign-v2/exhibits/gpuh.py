"""V-YFCC GPU-h projection from measured per-variant latencies and the timing protocol (README.md § gpuh).

usage: gpuh.py SYNTH_ARMS_CSV D1_YFCC_JSONL

Cost of one cell = overhead (build, quality, process; measured at 10 M on d1/yfcc10m) + Σ over its timed
variants (k × bs × mode) of measure.latency's windows: 50 warm-up + 3 × clamp(2 s / t, 1000, 5000) calls of t.
"""

import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "evaluation"))
from bench import config  # noqa: E402

CFG = Path(__file__).resolve().parents[4] / "evaluation" / "config"


def window_s(t_ms):
    t = t_ms / 1000
    return 50 * t + 3 * min(max(2.0 / t, 1000), 5000) * t


# 3 M arXiv-synth at v2.1 (pod c, .chains/v-ax-synth/2026-10-09-073000000-axsynth-rate.md), graph p50 ms
V1_3M = {1: 1.28, 16: 3.26}
V2_OVER_V1_3M_BS16 = {
    0.001: 0.69,
    0.01: 0.69,
    0.03: 0.75,
    0.1: 0.9,
    1.0: 1.83,
}  # 0.69 measured p <= 0.01, 1.83 at p 1
# official bloom eager / triton bloom graph, n_probe 24, arXiv c0_maincat k 100 (H2H-FINAL, 408b1188, T3):
# 1.586 / 0.198 at bs 1, 1.798 / 0.259 at bs 16. d1/yfcc10m ran official on clause (the pack_mask adapter) only.
OFF_RATIO = {1: 1.586 / 0.198, 16: 1.798 / 0.259}
N_3M, N_10M, D_3M, D_10M = 2_988_996, 10_000_000, 128, 192


def pilot_ratios(path):
    """arm / V1 latency per (p, bs) on goodreads-synth (408b1188, graph; official eager)."""
    g = {}
    for r in csv.DictReader(open(path)):
        if r["filter_kind"] == "clause":
            g[(r["arm"], float(r["p"]), int(r["bs"]), r["mode"])] = float(r["p50_ms"])
    v1 = {
        (p, bs): t
        for (a, p, bs, m), t in g.items()
        if a == "linr_v1_filter_mask/triton" and m == "graph"
    }
    return {(a, p, bs): t / v1[(p, bs)] for (a, p, bs, m), t in g.items() if m == "graph"}


def synth_latency(algo, params, p, bs, ratios, scale):
    """ms at 10 M d192 for one synth cell (graph): V1 from 3 M v2.1 × scale; the rest as ratios to V1."""
    v1 = V1_3M[bs] * scale
    near = min(V2_OVER_V1_3M_BS16, key=lambda q: abs(q - p))
    pr = min((0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0), key=lambda q: abs(q - p))
    if algo == "linr_v1_filter_mask":
        return v1
    if algo == "linr_v2":
        return v1 * (V2_OVER_V1_3M_BS16[near] if bs == 16 else ratios[("linr_v2/triton", pr, 1)])
    if algo == "linr_v3":
        return v1 * ratios[(f"linr_v3/triton/pool={params['candidate_pool_frac']}", pr, bs)]
    if algo == "postfilter":
        return v1 * ratios[(f"postfilter/torch/a={params['alpha']}", pr, bs)]
    if algo == "silvertorch":  # n_probe 24 as measured, plus the scanned share of a V1 scan
        return v1 * (
            ratios[("silvertorch/triton/np=24", pr, bs)] + params["n_probe"] / params["n_lists"]
        )
    raise KeyError(algo)


def d1_yfcc(path):
    """Per-arm graph (or eager) p50 by (bs, k) and the non-timed overhead seconds, d1/yfcc10m 72e5a90."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["status"] != "ok":
            continue
        t = {(e["bs"], e["k"], e["mode"]): e["median_ms"] for e in r["perf"] if e.get("median_ms")}
        timed = sum(window_s(v) for v in t.values())
        out[(r["algo"], r["backend"], r["params"].get("n_probe"))] = (
            t,
            r["elapsed_s"] - timed,
        )
    return out


def jobs(ds, suite):
    js = config.load_matrix(CFG / f"{ds}.yaml", CFG / "suites.yaml", suite)
    return [(j, p) for j in js for p in j.cells()]


def cell_s(j, lat, overhead):
    modes = ("eager",) if j.backend == "official" else ("eager", "graph")
    return overhead + sum(window_s(lat(bs)) for _ in j.ks for bs in j.batch_sizes for _ in modes)


QUALITY_COPY_S = (
    30  # seeds 1-2 of a seed-free arm copy seed 0's quality (evaluation.md § Quality cache)
)
SEED_FREE = ("linr_v1_filter_mask", "linr_v2", "postfilter")


def synth_lat(j, p, ratios, scale):
    pp = 1.0 if j.sweep == "p1" else float("0." + j.sweep[2:])
    off = OFF_RATIO if j.backend == "official" else {1: 1.0, 16: 1.0}
    params = {**p, "n_lists": p.get("n_lists", 4096)}

    def lat(bs):
        return synth_latency(j.algo, params, pp, bs, ratios, scale) * off[bs]

    return lat


def real_lat(j, p, d1, ratios, speed):
    """d1/yfcc10m's latency × the d1 → v2.1 speedup; SilverTorch scaled by its scanned share (d1 ran
    n_lists 1024 at n_probe 24); postfilter (not run at 10 M) as the pilot's postfilter / V1 ratio."""
    if j.algo == "postfilter":
        t = d1[("linr_v1_filter_mask", "triton", None)][0]
        f = {bs: ratios[(f"postfilter/torch/a={p['alpha']}", 0.01, bs)] for bs in (1, 16)}
    else:
        t = d1[(j.algo, j.backend, 24 if j.algo == "silvertorch" else None)][0]
        share = p["n_probe"] / p.get("n_lists", 4096) / (24 / 1024) if "n_probe" in p else 1.0
        f = {bs: max(1.0, share) for bs in (1, 16)}

    def lat(bs):
        return t.get((bs, 100, "graph"), t[(bs, 100, "eager")]) * f[bs] * speed

    return lat


def main():
    ratios = pilot_ratios(sys.argv[1])
    d1 = d1_yfcc(sys.argv[2])
    ov = {a: o for (a, b, n), (t, o) in d1.items() if b == "triton"}
    ov["postfilter"] = ov["linr_v1_filter_mask"]
    ov_off = max(o for (a, b, n), (t, o) in d1.items() if b == "official")
    rows, tot = [], collections.defaultdict(float)
    # low: V1 scales with N only, d1's overhead and the 3 M d1 -> v2.1 speedup (V1 bs 16 7.59 -> 3.26 ms)
    # hold; high: N x d, 1.5 x overhead, no speedup on the real-filter suites
    for case, scale, ovf, speed in (
        ("low", N_10M / N_3M, 1.0, 3.26 / 7.59),
        ("high", N_10M / N_3M * D_10M / D_3M, 1.5, 1.0),
    ):
        per = collections.defaultdict(lambda: [0, 0.0])
        for suite, ds in (("synth", "yfcc10m-synth"), ("filter", "yfcc10m"), ("deep", "yfcc10m")):
            for j, p in jobs(ds, suite):
                o = ov_off if j.backend == "official" else ov.get(j.algo, ov["silvertorch"])
                if j.seed > 0 and j.algo in SEED_FREE:
                    o = QUALITY_COPY_S
                lat = (
                    synth_lat(j, p, ratios, scale)
                    if suite == "synth"
                    else real_lat(j, p, d1, ratios, speed)
                )
                per[(suite, f"{j.algo}/{j.backend}")][0] += 1
                per[(suite, f"{j.algo}/{j.backend}")][1] += cell_s(j, lat, o * ovf)
        for (suite, a), (n, s) in sorted(per.items()):
            rows.append((case, suite, a, n, round(s / 3600, 2)))
            tot[(case, suite)] += s / 3600
    oracle_h = 0.3  # synth 5 sweeps + tags_and at 10 M, not measured here
    w = csv.writer(sys.stdout)
    w.writerow(["case", "suite", "arm", "cells", "GPU-h"])
    w.writerows(rows)
    for (case, suite), h in sorted(tot.items()):
        w.writerow([case, suite, "TOTAL", "", round(h, 1)])
    for case in ("low", "high"):
        h = sum(h for (c, _), h in tot.items() if c == case) + oracle_h
        w.writerow([case, "all", "TOTAL + oracle", "", round(h, 1)])


if __name__ == "__main__":
    main()
