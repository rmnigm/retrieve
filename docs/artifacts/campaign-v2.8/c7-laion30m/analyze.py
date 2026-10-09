"""C7 at 30 M: ours (triton) / Meta (official -O3) per interleaved cell. Ratio of medians with a 95 % bootstrap CI over the per-call vectors
(2,000 resamples), the per-round ratios (window i of both arms is round i), and the --profile kernel split (kernel_scopes: scorer / topk /
epilogue / other device µs and launches) with the host share (median − kernels_us). Run: python analyze.py RESULTS_DIR > c7.md"""

import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
recs = [json.loads(line) for line in open(root / "c7-laion30m/laion30m-d256.jsonl")]
samples = {}
for line in open(root / "c7-laion30m/laion30m-d256.samples.jsonl"):
    s = json.loads(line)
    samples[s["backend"], s["sweep"], s["params"]["n_probe"], s["bs"]] = np.asarray(
        s["ms"]
    )
ok = {
    (r["backend"], r["sweep"], r["params"]["n_probe"]): r
    for r in recs
    if r["status"] in ("ok", "partial")
}
rng = np.random.default_rng(0)


def split(p):
    ks = p.get("kernel_scopes") or {}
    us = " / ".join(
        f"{ks.get(s, {}).get('us', 0):.0f}"
        for s in ("scorer", "topk", "epilogue", "other")
    )
    return f"{us} ({p.get('kernels_calls')})"


def host(p):
    return p["median_ms"] - (p.get("kernels_us") or 0) / 1000


print(
    "| sweep | n_probe | bs | ours ms | Meta ms | ours/Meta | 95 % CI | per-round | ours scorer / topk / epi / other µs (launches) | Meta scorer / topk / epi / other µs (launches) | host ms ours / Meta |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|")
for sw in ("c0_domain", "tags4"):
    for npb in (32, 128):
        t, o = ok["triton", sw, npb], ok["official", sw, npb]
        for bs in (16, 64):
            pt = next(p for p in t["perf"] if p["bs"] == bs and p["mode"] == "eager")
            po = next(p for p in o["perf"] if p["bs"] == bs and p["mode"] == "eager")
            a, b = samples["triton", sw, npb, bs], samples["official", sw, npb, bs]
            boot = [
                np.median(rng.choice(a, a.size)) / np.median(rng.choice(b, b.size))
                for _ in range(2000)
            ]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rounds = [
                x / y for x, y in zip(pt["window_medians_ms"], po["window_medians_ms"])
            ]

            print(
                f"| {sw} | {npb} | {bs} | {pt['median_ms']:.3f} | {po['median_ms']:.3f} | **{pt['median_ms'] / po['median_ms']:.2f}** | {lo:.2f}-{hi:.2f} | "
                f"{' '.join(f'{x:.2f}' for x in rounds)} | {split(pt)} | {split(po)} | {host(pt):.3f} / {host(po):.3f} |"
            )
print()
print(
    "recall_oracle@100 ours / Meta:",
    {
        f"{sw} {npb}": (
            round(ok["triton", sw, npb]["quality"]["oracle"]["recall@100"], 4),
            round(ok["official", sw, npb]["quality"]["oracle"]["recall@100"], 4),
        )
        for sw in ("c0_domain", "tags4")
        for npb in (32, 128)
    },
)
print(
    "official so_sha256:",
    {
        (r["env"].get("official_build") or {}).get("so_sha256", "")[:12]
        for r in recs
        if r["backend"] == "official"
    },
    "code_version:",
    {r["env"]["code_version"][:8] for r in recs},
    "unstable:",
    sum(r["unstable"] for r in recs),
)
