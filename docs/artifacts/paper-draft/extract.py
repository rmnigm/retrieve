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


def v2_over_v1(leg, sweeps=None):
    """V2 / V1 ratio of graph p50 inside each interleave group, k 100; mean over seeds per sweep."""
    by = defaultdict(dict)
    for r in recs(leg):
        if r["algo"] in ("linr_v1_filter_mask", "linr_v2") and r["backend"] == "triton" and r["filter_kind"] == "clause":
            if sweeps is None or r["sweep"] in sweeps:
                by[(r["sweep"], r["seed"])][r["algo"]] = r
    acc = defaultdict(list)
    for (sweep, _), d in by.items():
        if len(d) < 2:
            continue
        for bs in (1, 16):
            a, b = perf(d["linr_v1_filter_mask"], bs), perf(d["linr_v2"], bs)
            if a and b:
                acc[(bs, sweep)].append((d["linr_v2"]["pass_rate"], b / a, a, b))
    out = {}
    for (bs, sweep), v in acc.items():
        out.setdefault(bs, []).append({"p": v[0][0], "sweep": sweep, "ratio": round(st.mean(x[1] for x in v), 3),
                                       "v1_ms": round(st.mean(x[2] for x in v), 3), "v2_ms": round(st.mean(x[3] for x in v), 3)})
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
# 30 M at matched recall, same leg / box / code (v2.9): exact V1, V2 and SilverTorch at the 25 % cap, graph.
data["laion_x"] = {f"{r['algo']}|{r['sweep']}": {bs: round(perf(r, bs), 3) for bs in (1, 16)} for r in recs("campaign-v2.9/laion30m-x")}
data["yfcc_real"] = {f"{s}@{n}": round(v, 3) for (s, n), v in yf.items()}

# V1 vs V2.
data["c1"] = {
    "0.8 M": v2_over_v1("campaign-v2.9/goodreads-synth-v1v2"),
    "3 M": v2_over_v1("campaign-v2.9/arxiv-synth-v1v2-pod1"),
    "10 M": v2_over_v1("campaign-v2.9/yfcc10m-synth-synth"),
    "30 M": v2_over_v1("campaign-v2.9/laion30m-synth-v1v2"),
}
# Real single- and multi-clause filters at v2.9 (same code as the synth curves): one point per sweep, aggregate pass rate.
real = []
for leg, label in (("campaign-v2.9/goodreads-filter", "goodreads"), ("campaign-v2.9/arxiv-filter", "arXiv"),
                   ("campaign-v2.9/yfcc10m-filter", "YFCC"), ("campaign-v2.9/laion30m-x", "LAION")):
    for bs, rows in v2_over_v1(leg).items():
        for r in rows:
            real.append(dict(r, bs=bs, label=f"{label} {r['sweep']}"))
data["c1_real"] = real

# int8 mechanism on YFCC.
data["int8"] = {s: json.load(open(HUB / f"artifacts/yfcc-int8/int8-{s}.json"))["recall_oracle@100"] for s in ("p1", "p001")}

# Co-design: full / partial, > 1 = co-design faster.
# Below 30 M on current code: exhibits' paired full / partial (ours v2.9 graph, Meta v2.8 -O3 eager).
pts = []
for r in csv.DictReader(open(HUB / "artifacts/exhibits/20261009-2125-c5/c5-below30m.csv")):
    if (r["code_version"], r["backend"], r["mode"]) in (("v2.8", "official", "eager"), ("v2.9", "triton", "graph")):
        pts.append({"who": "Meta's code" if r["backend"] == "official" else "ours (Triton)", "dataset": r["dataset"], "sweep": r["sweep"],
                    "n_probe": int(r["n_probe"]), "bs": int(r["bs"]), "mode": r["mode"], "ratio": float(r["full_over_partial"])})
# Ours at 10 M PubMed, v2.9 (graph p50 per interleaved pair).
pm = defaultdict(dict)
for r in recs("campaign-v2.9/pubmed-codesign-ours"):
    for p in r["perf"]:
        if p["mode"] == "graph" and p["k"] == 100:
            pm[(r["params"]["n_probe"], p["bs"])][r["params"]["bloom_path"]] = p["median_ms"]
for (npb, bs), d in pm.items():
    if len(d) == 2 and None not in d.values():
        pts.append({"who": "ours (Triton)", "dataset": "pubmed", "sweep": "c0_mesh", "n_probe": npb, "bs": bs, "mode": "graph",
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
for r in recs("campaign-v2.9/laion30m-codesign-laion30m"):
    for p in r["perf"]:
        if p["mode"] == "graph" and p["k"] == 100:
            ours30[(r["sweep"], r["params"]["n_probe"], p["bs"])][r["params"]["bloom_path"]] = p["median_ms"]
for (sw, npb, bs), d in ours30.items():
    if len(d) == 2 and None not in d.values():
        pts.append({"who": "ours (Triton)", "dataset": "laion30m", "sweep": sw, "n_probe": npb, "bs": bs, "mode": "graph",
                    "ratio": round(d["full"] / d["partial"], 3)})
data["c5"] = pts

# C7 at d128: official (fp16, -O3 build) / Triton, final adapter, h2h at v2.8 (one interleave group per cell), k 100.
h2h = defaultdict(dict)
for leg in ("campaign-v2.8/goodreads-h2h", "campaign-v2.8/arxiv-h2h"):
    for r in recs(leg):
        arm = r["backend"] + ("/" + r["params"]["score_path"] if "score_path" in r["params"] else "")
        for p in r["perf"]:
            if p["k"] == 100 and p["median_ms"]:
                h2h[(r["dataset"], r["filter_kind"], p["bs"])][(arm, p["mode"])] = p
rows = []
for (ds, fl, bs), d in sorted(h2h.items()):
    te, tg, of = d[("triton", "eager")], d[("triton", "graph")], d[("official/fp16", "eager")]
    rows.append({"cell": f"{ds} {fl} bs {bs}", "e2e": round(of["median_ms"] / te["median_ms"], 2),
                 "device": round(of["kernels_us"] / te["kernels_us"], 2),
                 "graph_vs_official": round(of["median_ms"] / tg["median_ms"], 2)})
data["c7"] = rows

# C7 at d768: official (-O3 build) / Triton eager, final adapter, one interleave group per n_probe.
# Until campaign-v2.9/pubmed-filter is on the Hub, the ratios come from pod b's c7-pubmed-v29 control note.
c7p = defaultdict(dict)
for r in recs("campaign-v2.9/pubmed-filter"):
    if r["algo"] == "silvertorch":
        c7p[r["params"]["n_probe"]][r["backend"]] = r
if c7p:
    data["c7_d768"] = {npb: {bs: round(perf(d["official"], bs, "eager") / perf(d["triton"], bs, "eager"), 3) for bs in (1, 16)}
                       for npb, d in sorted(c7p.items())}
else:
    data["c7_d768"] = {24: {1: round(0.822 / 0.413, 3), 16: round(1.155 / 0.529, 3)}, 1024: {1: round(1.056 / 0.622, 3), 16: round(2.614 / 2.442, 3)}}

# C7 at 30 M LAION d256, n_probe 128, bloom partial path: ours at v2.10 (ST-WIDE-2) vs Meta fp16 -O3, eager (repo table of ST-WIDE-2's gate).
c7l = {}
for line in (Path(__file__).parents[1] / "campaign-v2.9/st-wide-2-30m/st-wide-2-30m.md").read_text().splitlines():
    c = [x.strip() for x in line.strip("|").split("|")]
    if len(c) > 7 and c[2] == "eager" and c[1].isdigit():
        c7l[f"{c[0]}|{c[1]}"] = round(float(c[5]) / float(c[4]), 3)
data["c7_30m"] = c7l

# V3 bits on PubMed d768.
v3 = defaultdict(dict)
for r in recs("campaign-v2.5/pubmed-v3bits"):
    v3[(r["sweep"], r["params"]["candidate_pool_frac"])][r["params"]["k_bits"]] = {"recall": round(recall(r), 4), "ms16": round(perf(r, 16), 2)}
data["v3"] = {f"{s}|{pool}": d for (s, pool), d in v3.items()}
# V3 (pool 1 %, 768 bits) against the cheaper exact arm per sweep and batch (V1 / V2 from the same box and code, separate leg).
ex = defaultdict(dict)
for r in recs("campaign-v2.5/pubmed-router"):
    if r["status"] == "ok" and r["algo"] in ("linr_v1_filter_mask", "linr_v2"):
        for bs in (1, 16):
            ex[(r["sweep"], bs)][r["algo"]] = perf(r, bs)
v3r = {(r["sweep"], bs): perf(r, bs) for r in recs("campaign-v2.5/pubmed-v3bits")
       if r["params"]["candidate_pool_frac"] == 0.01 and r["params"]["k_bits"] == 768 for bs in (1, 16)}
data["v3_vs_exact"] = {f"{sw}|{bs}": {"exact": min(d, key=d.get), "speedup": round(min(d.values()) / v3r[(sw, bs)], 3)}
                       for (sw, bs), d in ex.items() if (sw, bs) in v3r}
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
