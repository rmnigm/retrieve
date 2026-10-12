"""V-PILOT pilot gate and prediction tables from the goodreads-synth synth records.

python3 gate.py /scratch/campaign-v2/results/synth/goodreads-synth-d128.jsonl /data/goodreads-work-id/synth_filter.json
"""

import json
import statistics as st
import sys
from collections import Counter, defaultdict

SWEEPS = ["p0001", "p0003", "p001", "p003", "p01", "p03", "p1"]
TARGET = dict(zip(SWEEPS, [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]))


def arm(r):
    p = dict(r["params"])
    q = ",".join(f"{k}={p[k]}" for k in sorted(p) if k in ("compile", "alpha", "n_probe", "candidate_pool_frac"))
    return f"{r['algo']}/{r['backend']}" + (f"[{q}]" if q else "")


def main(path, synth_json):
    recs = [json.loads(line) for line in open(path)]
    last = {}
    for r in recs:  # latest record per key wins
        last[(arm(r), r["filter_kind"], r["sweep"], r["seed"])] = r
    recs = list(last.values())
    print("records", len(recs), Counter(r["status"] for r in recs))
    for r in recs:
        if r["status"] != "ok":
            print("  NOT OK", arm(r), r["filter_kind"], r["sweep"], r["seed"], r["status"], r.get("stage"))
    by_arm = Counter((arm(r), r["filter_kind"]) for r in recs)
    print("\ncells per arm"), [print(f"  {a:60s} {k:6s} {n}") for (a, k), n in sorted(by_arm.items())]

    sj = json.load(open(synth_json))
    print("\npass rates (synth_filter.json achieved vs target; records' pass_rate over kept queries)")
    rec_pr = defaultdict(set)
    for r in recs:
        rec_pr[r["sweep"]].add(round(r["pass_rate"], 9))
    for s, tgt, ach in zip(SWEEPS, sj["rates"], sj["pass_rates"]):
        print(f"  {s:6s} target {tgt:<6} achieved {ach:.7f} rel {(ach - tgt) / tgt:+.3%} "
              f"{'OK' if abs(ach - tgt) / tgt <= 0.01 else 'MISS'}  records {sorted(rec_pr[s])}")

    def recall(r, k):
        q = r.get("quality") or {}
        return (q.get("oracle") or {}).get(f"recall@{k}")

    print("\nrecall_oracle@k, mean over seeds (min..max)")
    tab = defaultdict(list)
    for r in recs:
        if r["status"] == "ok":
            for k in r["ks"]:
                v = recall(r, k)
                if v is not None:
                    tab[(arm(r), r["filter_kind"], k, r["sweep"])].append(v)
    rows = sorted({(a, fk, k) for a, fk, k, _ in tab})
    print(f"  {'arm':52s} {'kind':6s} {'k':>5s} " + " ".join(f"{s:>15s}" for s in SWEEPS))
    for a, fk, k in rows:
        cells = []
        for s in SWEEPS:
            v = tab.get((a, fk, k, s))
            cells.append(f"{st.mean(v):.4f}({min(v):.4f})" if v else "-")
        print(f"  {a:52s} {fk:6s} {k:5d} " + " ".join(f"{c:>15s}" for c in cells))

    print("\nmedian_ms, median over seeds, per (arm, kind, bs, k, mode)")
    lat = defaultdict(list)
    for r in recs:
        for e in r.get("perf") or []:
            if e.get("median_ms") is not None:
                lat[(arm(r), r["filter_kind"], e["bs"], e["k"], e["mode"], r["sweep"])].append(e["median_ms"])
    lrows = sorted({x[:5] for x in lat})
    print(f"  {'arm':52s} {'kind':6s} {'bs':>3s} {'k':>5s} {'mode':6s} " + " ".join(f"{s:>8s}" for s in SWEEPS))
    for key in lrows:
        cells = [lat.get((*key, s)) for s in SWEEPS]
        print(f"  {key[0]:52s} {key[1]:6s} {key[2]:3d} {key[3]:5d} {key[4]:6s} "
              + " ".join(f"{st.median(v):8.3f}" if v else f"{'-':>8s}" for v in cells))

    def ratio(num, den, kind):
        print(f"\n{num} / {den} ({kind}), median over seeds of per-seed ratios")
        for bs in (1, 16):
            for k in (100, 1000):
                for mode in ("eager", "graph"):
                    out = []
                    for s in SWEEPS:
                        a, b = lat.get((num, kind, bs, k, mode, s)), lat.get((den, kind, bs, k, mode, s))
                        out.append(f"{st.median([x / y for x, y in zip(a, b)]):8.3f}" if a and b else f"{'-':>8s}")
                    print(f"  bs {bs:2d} k {k:4d} {mode:6s} " + " ".join(out))

    for kind in ("clause", "bloom"):
        ratio("linr_v2/triton", "linr_v1_filter_mask/triton", kind)
        for f in ("0.01", "0.05"):
            ratio(f"linr_v3/triton[candidate_pool_frac={f}]", "linr_v2/triton", kind)

    print("\nV1 triton latency spread across p: (max - min) / min of the per-p medians")
    for key in lrows:
        if key[0] == "linr_v1_filter_mask/triton":
            v = [st.median(lat[(*key, s)]) for s in SWEEPS if (*key, s) in lat]
            print(f"  {key[1]:6s} bs {key[2]:2d} k {key[3]:4d} {key[4]:6s} {(max(v) - min(v)) / min(v):+.3f} over {len(v)} p")

    print("\nclocks")
    fr = [r["env"].get("frac_windows_below_max") for r in recs if r.get("perf")]
    fr = [x for x in fr if x is not None]
    print(f"  records with perf {len(fr)}; frac_windows_below_max: mean {st.mean(fr):.3f}, "
          f"records with any window below max {sum(x > 0 for x in fr)}, with >= 10 % {sum(x >= 0.1 for x in fr)}")
    print(f"  unstable records {sum(bool(r.get('unstable')) for r in recs)} / {len(recs)}")
    sm = Counter(m for r in recs for e in r.get("perf") or [] for m in (e.get("window_sm_mhz") or []) if m)
    print("  window sm_mhz histogram", dict(sorted(sm.items())))

    print("\nwall time per arm (s; an interleaved group's elapsed_s split evenly across its arms)")
    wall = Counter()
    for r in recs:
        n = len(r["interleave"]["arms"]) if r.get("interleave") else 1
        wall[arm(r)] += r["elapsed_s"] / n
    for a, s in sorted(wall.items()):
        print(f"  {a:52s} {s:8.0f}")
    print(f"  total {sum(wall.values()):.0f} s = {sum(wall.values()) / 3600:.2f} h")


if __name__ == "__main__":
    main(*sys.argv[1:])
