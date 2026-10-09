"""Chart data for the paper-materials draft (NOT CITABLE). Reads Hub legs fetched under HUB, writes data.json."""
import csv
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HUB = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/paper-draft/hub")
OUT = Path(__file__).with_name("data.json")


def recs(leg):
    for f in sorted(glob.glob(str(HUB / leg / "**/*.jsonl"), recursive=True)):
        for line in open(f):
            r = json.loads(line)
            if r["status"] in ("ok", "partial"):
                yield r


def perf(r, bs, mode="graph", k=100):
    for p in r.get("perf") or []:
        if p["bs"] == bs and p["mode"] == mode and p["k"] == k:
            return p["median_ms"]
    return None


def recall(r):
    return r["quality"]["oracle"]["recall@100"]


def p_of(sweep):
    return float("0." + sweep[2:]) if sweep.startswith("p0") else float(sweep[1:])


def st_recall(leg, dataset=None):
    """SilverTorch Triton clause recall@100 by (sweep, n_probe), median over seeds."""
    acc = defaultdict(list)
    for r in recs(leg):
        if r["algo"] == "silvertorch" and r["backend"] == "triton" and r["filter_kind"] == "clause":
            acc[(r["sweep"], r["params"]["n_probe"])].append(recall(r))
    return {k: st.median(v) for k, v in acc.items()}


def v2_over_v1(leg):
    """V2 / V1 ratio of graph p50 inside each interleave group, k 100."""
    by = defaultdict(dict)
    for r in recs(leg):
        if r["algo"] in ("linr_v1_filter_mask", "linr_v2") and r["backend"] == "triton" and r["filter_kind"] == "clause":
            by[(r["sweep"], r["seed"])][r["algo"]] = r
    out = {}
    for (sweep, _), d in by.items():
        if len(d) < 2:
            continue
        for bs in (1, 16):
            a, b = perf(d["linr_v1_filter_mask"], bs), perf(d["linr_v2"], bs)
            if a and b:
                out.setdefault(bs, []).append({"p": d["linr_v2"]["pass_rate"], "sweep": sweep, "ratio": round(b / a, 3),
                                               "v1_ms": round(a, 3), "v2_ms": round(b, 3)})
    for v in out.values():
        v.sort(key=lambda x: x["p"])
    return out


data = {}

# Local pass rate: uniform vs correlated filters, arXiv 3 M, n_lists 2048.
uni = st_recall("campaign-v2.5/arxiv-synth-synth")
cor = st_recall("campaign-v2.5/arxiv-corr-synth-synth")
corr_p = {r["sweep"]: r["pass_rate"] for r in recs("campaign-v2.5/arxiv-corr-synth-synth")}
data["f2"] = {
    "uniform": {npb: sorted([[p_of(s), round(v, 4)] for (s, n), v in uni.items() if n == npb]) for npb in (24, 256, 1024)},
    "correlated": {npb: sorted([[corr_p[s], round(v, 4)] for (s, n), v in cor.items() if n == npb]) for npb in (24, 256, 1024)},
}
data["gls"] = list(csv.DictReader(open(HUB / "artifacts/exhibits/20261009-1057-gls/gls-sweeps.csv")))

# IVF recall at 24 probes vs N, uniform synth (each N on its own leg).
scal = []
for label, n, leg, nl in [("0.8 M · d128", 0.8, "campaign-v2/goodreads-synth-synth", 1024),
                          ("3 M · d128", 3, "campaign-v2.5/arxiv-synth-synth", 2048),
                          ("10 M · d192", 10, "campaign-v2.5/yfcc10m-synth-synth", 4096),
                          ("30 M · d256", 30, "campaign-v2.5/laion30m-synth-synth", 16384)]:
    rr = st_recall(leg)
    scal.append({"label": label, "n": n, "n_lists": nl,
                 "p01": round(rr.get(("p01", 24), float("nan")), 3), "p1": round(rr.get(("p1", 24), float("nan")), 3)})
data["scaling"] = scal

# Real-filter recall at 24 probes and best measured (sources: V-GR-DEEP, arXiv IVF-TUNE, YFCC deep, LAION grid).
lai = st_recall("campaign-v2.5/laion30m-filter")
yf = st_recall("campaign-v2.5/yfcc10m-deep")
data["laion"] = {f"{s}@{n}": round(v, 3) for (s, n), v in lai.items()}
data["yfcc_real"] = {f"{s}@{n}": round(v, 3) for (s, n), v in yf.items()}

# V1 vs V2.
data["c1"] = {
    "3 M": v2_over_v1("campaign-v2.7/arxiv-synth-synth"),
    "10 M YFCC": v2_over_v1("campaign-v2.7/yfcc10m-synth-synth"),
    "30 M LAION": v2_over_v1("campaign-v2.7/laion30m-synth-laion30m-synth"),
    "30 M LAION, real filters": v2_over_v1("campaign-v2.5/laion30m-filter"),
}

# int8 mechanism on YFCC.
data["int8"] = {s: json.load(open(HUB / f"artifacts/yfcc-int8/int8-{s}.json"))["recall_oracle@100"] for s in ("p1", "p001")}

# Co-design: full / partial, > 1 = co-design faster.
c5 = list(csv.DictReader(open(HUB / "artifacts/exhibits/20261009-1324-c5/c5.csv")))
pts = [{"who": "ours (Triton)", "dataset": r["dataset"], "sweep": r["sweep"],
        "n_probe": int(r["n_probe"]), "bs": int(r["bs"]), "mode": r["mode"], "ratio": float(r["full_over_partial"])}
       for r in c5 if r["backend"] != "official" and r["mode"] == "graph"]
meta = defaultdict(dict)
for leg in ("campaign-v2.8/arxiv-codesign", "campaign-v2.8/goodreads-codesign"):
    for r in recs(leg):
        for p in r["perf"]:
            if p["mode"] == "eager" and p["k"] == 100:
                meta[(r["dataset"], r["sweep"], r["params"]["n_probe"], p["bs"])][r["params"]["bloom_path"]] = p["median_ms"]
for (ds, sw, npb, bs), d in meta.items():
    if len(d) == 2 and None not in d.values():
        pts.append({"who": "Meta's code", "dataset": ds, "sweep": sw, "n_probe": npb, "bs": bs, "mode": "eager",
                    "ratio": round(d["full"] / d["partial"], 3)})
lai30 = defaultdict(dict)
for r in recs("campaign-v2.8/laion30m-codesign-laion30m"):
    for p in r["perf"]:
        if p["mode"] == "eager" and p["k"] == 100:
            lai30[(r["sweep"], r["params"]["n_probe"], p["bs"])][r["params"]["bloom_path"]] = p["median_ms"]
for (sw, npb, bs), d in lai30.items():
    if len(d) == 2 and None not in d.values():
        pts.append({"who": "Meta's code", "dataset": "laion30m", "sweep": sw, "n_probe": npb, "bs": bs, "mode": "eager",
                    "ratio": round(d["full"] / d["partial"], 3)})
ours30 = defaultdict(dict)
for r in recs("campaign-v2.7/laion30m-codesign-laion30m"):
    for p in r["perf"]:
        if p["mode"] == "graph" and p["k"] == 100:
            ours30[(r["sweep"], r["params"]["n_probe"], p["bs"])][r["params"]["bloom_path"]] = p["median_ms"]
for (sw, npb, bs), d in ours30.items():
    if len(d) == 2 and None not in d.values():
        pts.append({"who": "ours (Triton)", "dataset": "laion30m", "sweep": sw, "n_probe": npb, "bs": bs, "mode": "graph",
                    "ratio": round(d["full"] / d["partial"], 3)})
data["c5"] = pts

# C7: official / Triton eager end to end and device time, v2.1 h2h (d128).
t3 = list(csv.DictReader(open(HUB / "artifacts/exhibits/20261009-0843-c7/t3x.csv")))
c7 = defaultdict(dict)
for r in t3:
    if r["code_version"] != "v2.1":
        continue
    c7[(r["dataset"], r["filter"], r["k"], r["bs"])][r["arm"]] = r
rows = []
for (ds, fl, k, bs), d in c7.items():
    te, of = d.get("triton eager"), d.get("official/fp16 eager")
    if te and of and k == "100":
        dev = (float(of["device_ms"]) / float(te["device_ms"])) if of.get("device_ms") and te.get("device_ms") else None
        rows.append({"cell": f"{ds} {fl} bs {bs}", "e2e": round(float(of["p50_ms"]) / float(te["p50_ms"]), 2),
                     "device": round(dev, 2) if dev else None,
                     "graph_vs_official": round(float(of["p50_ms"]) / float(d["triton graph"]["p50_ms"]), 2) if "triton graph" in d else None})
data["c7"] = sorted(rows, key=lambda x: x["cell"])

# C7 at d768: official (-O3 build) / Triton eager, final adapter, one interleave group per n_probe.
# Until campaign-v2.8/pubmed-filter is on the Hub, the ratios come from pod b's c7-pubmed-v28 control note.
c7p = defaultdict(dict)
for r in recs("campaign-v2.8/pubmed-filter"):
    if r["algo"] == "silvertorch":
        c7p[r["params"]["n_probe"]][r["backend"]] = r
if c7p:
    data["c7_d768"] = {npb: {bs: round(perf(d["official"], bs, "eager") / perf(d["triton"], bs, "eager"), 3) for bs in (1, 16)}
                       for npb, d in sorted(c7p.items())}
else:
    data["c7_d768"] = {24: {1: round(0.822 / 0.412, 3), 16: round(1.161 / 0.533, 3)}, 1024: {1: round(1.039 / 0.799, 3), 16: round(2.605 / 8.837, 3)}}

# V3 bits on PubMed d768.
v3 = defaultdict(dict)
for r in recs("campaign-v2.5/pubmed-v3bits"):
    v3[(r["sweep"], r["params"]["candidate_pool_frac"])][r["params"]["k_bits"]] = {"recall": round(recall(r), 4), "ms16": round(perf(r, 16), 2)}
data["v3"] = {f"{s}|{pool}": d for (s, pool), d in v3.items()}
g3 = defaultdict(list)
for r in recs("campaign-v2.2/goodreads-synth-v3bits"):
    if r["filter_kind"] == "clause" and r["params"]["candidate_pool_frac"] == 0.01:
        g3[(r["params"]["k_bits"], r["pass_rate"])].append(recall(r))
data["v3_goodreads"] = sorted([[kb, round(p, 4), round(st.median(v), 4)] for (kb, p), v in g3.items()])

# C4: bloom false-positive rate vs width (Triton, quality legs).
fpr = defaultdict(list)
for ds in ("arxiv", "goodreads", "pubmed"):
    for r in recs(f"campaign-v2/{ds}-bloomwidth"):
        if r["backend"] == "triton":
            fpr[(ds, r["params"]["k_hash"], r["params"]["m_bits"])].append(r["bloom_fp_rate"] or 0.0)
data["fpr"] = sorted([[ds, kh, mb, st.mean(v)] for (ds, kh, mb), v in fpr.items()])

OUT.write_text(json.dumps(data, indent=1, default=float))
print("wrote", OUT)
