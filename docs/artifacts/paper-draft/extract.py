"""Chart data for the paper-materials page (NOT CITABLE). Reads Hub legs and exhibit tables fetched under HUB, writes data.json.

Every number comes from a record or an exhibits table; nothing is typed in from notes.
"""
import csv
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HUB = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/paper-draft/hub")
HERE = Path(__file__).parent
EXHIBITS = HUB / "artifacts/exhibits"


def recs(leg):
    for f in sorted(glob.glob(str(HUB / leg / "**/*.jsonl"), recursive=True)):
        if f.endswith(".samples.jsonl"):
            continue
        for line in open(f):
            r = json.loads(line)
            if r["status"] in ("ok", "partial"):
                yield r


def perf(r, bs, mode="graph", k=100):
    for p in r.get("perf") or []:
        if p["bs"] == bs and p["mode"] == mode and p["k"] == k and p["median_ms"]:
            return p["median_ms"]
    return None


def recall(r):
    return r["quality"]["oracle"]["recall@100"]


def table(path):
    """Rows of a markdown table in the repo, as lists of stripped cells."""
    for line in (HERE.parent / path).read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) > 3 and not c[0].startswith("-"):
            yield c


data = {}


# C1: V2 / V1 inside each interleave group (graph, k 100), mean over seeds per sweep.
def v2_over_v1(leg):
    by = defaultdict(dict)
    for r in recs(leg):
        if r["algo"] in ("linr_v1_filter_mask", "linr_v2") and r["backend"] == "triton" and r["filter_kind"] == "clause":
            by[(r["sweep"], r["seed"])][r["algo"]] = r
    acc = defaultdict(list)
    for (sweep, _), d in by.items():
        if len(d) == 2:
            for bs in (1, 16):
                a, b = perf(d["linr_v1_filter_mask"], bs), perf(d["linr_v2"], bs)
                if a and b:
                    acc[(bs, sweep)].append((d["linr_v2"]["pass_rate"], b / a))
    out = defaultdict(list)
    for (bs, sweep), v in acc.items():
        out[bs].append({"p": v[0][0], "sweep": sweep, "ratio": round(st.mean(x[1] for x in v), 3)})
    return {bs: sorted(v, key=lambda x: x["p"]) for bs, v in out.items()}


data["c1"] = {
    "0.8 M": v2_over_v1("campaign-v2.9/goodreads-synth-v1v2"),
    "3 M": v2_over_v1("campaign-v2.9/arxiv-synth-v1v2-pod1"),
    "10 M": v2_over_v1("campaign-v2.9/yfcc10m-synth-synth"),
    "30 M": v2_over_v1("campaign-v2.9/laion30m-synth-v1v2"),
}
data["c1_real"] = [dict(r, bs=bs, label=f"{name} {r['sweep']}")
                   for leg, name in (("campaign-v2.9/goodreads-filter", "goodreads"), ("campaign-v2.9/arxiv-filter", "arXiv"),
                                     ("campaign-v2.9/yfcc10m-filter", "YFCC"), ("campaign-v2.9/laion30m-x", "LAION"))
                   for bs, rows in v2_over_v1(leg).items() for r in rows]

# C2: V3 recall by bit budget (goodreads d128, pool 1 %) and V3 against the cheaper exact arm (PubMed d768, same box and code).
g3 = defaultdict(list)
for r in recs("campaign-v2.2/goodreads-synth-v3bits"):
    if r["filter_kind"] == "clause" and r["params"]["candidate_pool_frac"] == 0.01:
        g3[(r["params"]["k_bits"], round(r["pass_rate"], 4))].append(recall(r))
data["c2_bits"] = sorted([kb, p, round(st.median(v), 4)] for (kb, p), v in g3.items())
exact = defaultdict(dict)
for r in recs("campaign-v2.5/pubmed-router"):
    if r["status"] == "ok" and r["algo"] in ("linr_v1_filter_mask", "linr_v2"):
        for bs in (1, 16):
            exact[(r["sweep"], bs)]["V1" if r["algo"] == "linr_v1_filter_mask" else "V2"] = perf(r, bs)
v3 = {(r["sweep"], bs): (perf(r, bs), recall(r)) for r in recs("campaign-v2.5/pubmed-v3bits")
      if r["params"]["candidate_pool_frac"] == 0.01 and r["params"]["k_bits"] == 768 for bs in (1, 16)}
data["c2_v3"] = {f"{sw}|{bs}": {"vs": min(d, key=d.get), "speedup": round(min(d.values()) / v3[(sw, bs)][0], 3),
                                "recall": round(v3[(sw, bs)][1], 4)}
                 for (sw, bs), d in exact.items() if (sw, bs) in v3}

# C3: torch.compile(max-autotune) V1 / our Triton V1 on real filters, eager over eager.
c3 = defaultdict(dict)
for ds in ("goodreads", "arxiv", "yfcc10m", "pubmed"):
    for r in recs(f"campaign-v2.10/{ds}-c3-real"):
        if r["status"] == "ok":
            c3[(ds, r["sweep"])]["compiled" if r["backend"] == "torch" else "triton"] = r
data["c3"] = [{"dataset": ds, "sweep": sw, "p": round(d["triton"]["pass_rate"], 4),
               **{f"bs{bs}": round(perf(d["compiled"], bs, "eager") / perf(d["triton"], bs, "eager"), 3) for bs in (1, 16)}}
              for (ds, sw), d in sorted(c3.items()) if len(d) == 2]

# C4: bloom false-positive rate by width (Triton, quality legs, mean over seeds).
fpr = defaultdict(list)
for ds in ("arxiv", "goodreads", "pubmed"):
    for r in recs(f"campaign-v2/{ds}-bloomwidth"):
        if r["backend"] == "triton":
            fpr[(ds, r["params"]["k_hash"], r["params"]["m_bits"])].append(r["bloom_fp_rate"] or 0.0)
data["c4"] = sorted([ds, kh, mb, st.mean(v)] for (ds, kh, mb), v in fpr.items())

# C5: full / partial (> 1 = co-design faster). Below 30 M: exhibits' paired ratios (ours v2.9 graph, Meta v2.8 -O3 eager).
c5 = []
for r in csv.DictReader(open(EXHIBITS / "20261009-2125-c5/c5-below30m.csv")):
    if (r["code_version"], r["backend"], r["mode"]) in (("v2.8", "official", "eager"), ("v2.9", "triton", "graph")):
        c5.append({"who": "Meta" if r["backend"] == "official" else "ours", "dataset": r["dataset"], "sweep": r["sweep"],
                   "n_probe": int(r["n_probe"]), "bs": int(r["bs"]), "ratio": float(r["full_over_partial"])})
for leg, who, mode in (("campaign-v2.9/pubmed-codesign-ours", "ours", "graph"),
                       ("campaign-v2.9/laion30m-codesign-laion30m", "ours", "graph"),
                       ("campaign-v2.8/laion30m-codesign-laion30m", "Meta", "eager")):
    pair = defaultdict(dict)
    for r in recs(leg):
        for p in r["perf"]:
            if p["mode"] == mode and p["k"] == 100 and p["median_ms"]:
                pair[(r["dataset"], r["sweep"], r["params"]["n_probe"], p["bs"])][r["params"]["bloom_path"]] = p["median_ms"]
    for (ds, sw, npb, bs), d in pair.items():
        if len(d) == 2:
            c5.append({"who": who, "dataset": ds, "sweep": sw, "n_probe": npb, "bs": bs, "ratio": round(d["full"] / d["partial"], 3)})
data["c5"] = c5


# C6: recall vs pass rate at 3 M, uniform and cluster-correlated synth (SilverTorch triton clause, 2048 lists).
def st_recall(leg):
    acc = defaultdict(list)
    for r in recs(leg):
        if r["algo"] == "silvertorch" and r["backend"] == "triton" and r["filter_kind"] == "clause":
            acc[(r["sweep"], r["params"]["n_probe"], r["pass_rate"])].append(recall(r))
    return {k: st.median(v) for k, v in acc.items()}


uni, cor = st_recall("campaign-v2.5/arxiv-synth-synth"), st_recall("campaign-v2.5/arxiv-corr-synth-synth")
data["c6_f2"] = {name: {n: sorted([round(p, 4), round(v, 4)] for (_, npb, p), v in src.items() if npb == n) for n in (24, 256)}
                 for name, src in (("uniform", uni), ("correlated", cor))}
data["c6_gls"] = list(csv.DictReader(open(EXHIBITS / "20261009-1057-gls/gls-sweeps.csv")))

# C6: the recall ceiling by item-score precision on the same probes, each bench at its best tuned point (YFCC at 256 probes).
data["c6_ceiling"] = []
for label, f in (("goodreads 0.8 M", "ceiling-int8-gr-ax/ceiling-goodreads-c0_genre.json"),
                 ("arXiv 3 M", "ceiling-int8-gr-ax/ceiling-arxiv-c0_maincat.json"),
                 ("YFCC 10 M", "yfcc-int8/int8-p1.json"),
                 ("PubMed 10 M", "ceiling-int8-pubmed/ceiling-pubmed-c3_journal_reverse.json"),
                 ("PubMed 10 M · 0.02 % pass", "ceiling-int8-pubmed/ceiling-pubmed-c0_mesh.json")):
    r = json.load(open(HUB / "artifacts" / f))
    data["c6_ceiling"].append({"label": label, "p": round(r["pass_rate"], 4),
                               **{k: round(r["recall_oracle@100"][k], 4) for k in ("shipped", "per_row", "fp16")}})

# Scale: IVF recall at the paper's 24 probes vs N; uniform synth at p 0.1 and p 1, and each dataset's broadest real filter.
synth = []
for n, leg in ((0.8, "campaign-v2/goodreads-synth-synth"), (3, "campaign-v2.5/arxiv-synth-synth"),
               (10, "campaign-v2.9/yfcc10m-synth-synth"), (30, "campaign-v2.5/laion30m-synth-synth")):
    rr = {(sw, npb): v for (sw, npb, _), v in st_recall(leg).items()}
    synth.append({"n": n, "p01": round(rr[("p01", 24)], 3), "p1": round(rr[("p1", 24)], 3)})
real24 = []
for n, leg, sweep, label in ((0.8, "campaign-v2.9/goodreads-filter", "c0_genre", "goodreads genre (33 %)"),
                             (3, "campaign-v2.9/arxiv-filter", "c0_maincat", "arXiv main category (14 %)"),
                             (10, "campaign-v2.9/yfcc10m-filter", "tags_and", "YFCC tags (1.9 %)"),
                             (10, "campaign-v2.9/pubmed-deep", "c0_mesh", "PubMed MeSH (0.02 %)"),
                             (30, "campaign-v2.9/laion30m-filter", "c0_domain", "LAION domain (0.95 %)")):
    for r in recs(leg):
        if r["algo"] == "silvertorch" and r["filter_kind"] == "clause" and r["sweep"] == sweep and r["params"]["n_probe"] == 24:
            real24.append({"n": n, "r": round(recall(r), 3), "label": label})
            break
data["scale_at24"] = {"synth": synth, "real": real24}

# Scale: IVF against the cheaper exact arm at recall 0.95, same leg / box / code (v2.9). IVF at the dataset's tuned probe count
# (PubMed: the 25 % cap from its deep sweep); a sweep whose IVF recall stays below 0.95 is reported with its best recall.
matched = []
for leg, name, ivf_leg in (("campaign-v2.9/goodreads-filter", "0.8 M goodreads", None), ("campaign-v2.9/arxiv-filter", "3 M arXiv", None),
                           ("campaign-v2.9/yfcc10m-filter", "10 M YFCC", None), ("campaign-v2.9/pubmed-filter", "10 M PubMed", "campaign-v2.9/pubmed-deep"),
                           ("campaign-v2.9/laion30m-x", "30 M LAION", None)):
    ex, ivf = defaultdict(dict), {}
    for r in recs(leg):
        if r["backend"] == "triton" and r["filter_kind"] == "clause" and r["algo"] in ("linr_v1_filter_mask", "linr_v2"):
            ex[r["sweep"]][r["algo"]] = r
    for r in recs(ivf_leg or leg):
        if r["algo"] == "silvertorch" and r["backend"] == "triton" and r["filter_kind"] == "clause":
            if r["sweep"] not in ivf or r["params"]["n_probe"] > ivf[r["sweep"]]["params"]["n_probe"]:
                ivf[r["sweep"]] = r
    for sw, d in sorted(ex.items()):
        if sw not in ivf or len(d) < 2:
            continue
        iv = ivf[sw]
        row = {"label": f"{name} · {sw.replace('_', ' ')}", "p": round(iv["pass_rate"], 4), "recall": round(recall(iv), 3)}
        for bs in (1, 16):
            row[f"bs{bs}"] = round(min(perf(d[a], bs) for a in d) / perf(iv, bs), 3) if recall(iv) >= 0.95 else None
        matched.append(row)
data["scale_matched"] = matched

# C7 at d128: Meta (-O3, fp16) / ours, h2h at v2.8, eager end to end and device time, k 100.
h2h = defaultdict(dict)
for leg in ("campaign-v2.8/goodreads-h2h", "campaign-v2.8/arxiv-h2h"):
    for r in recs(leg):
        arm = r["backend"] + ("/" + r["params"]["score_path"] if "score_path" in r["params"] else "")
        for p in r["perf"]:
            if p["k"] == 100 and p["median_ms"]:
                h2h[(r["dataset"], r["filter_kind"], p["bs"])][(arm, p["mode"])] = p
data["c7_d128"] = [{"cell": f"{ds} {'no filter' if fl == 'none' else fl} · bs {bs}",
                    "e2e": round(d[("official/fp16", "eager")]["median_ms"] / d[("triton", "eager")]["median_ms"], 2),
                    "device": round(d[("official/fp16", "eager")]["kernels_us"] / d[("triton", "eager")]["kernels_us"], 2)}
                   for (ds, fl, bs), d in sorted(h2h.items())]

# C7 at d768: Meta (-O3, fp16) / ours v2.9, PubMed bloom, interleaved per n_probe, eager.
d768 = defaultdict(dict)
for r in recs("campaign-v2.9/pubmed-filter"):
    if r["algo"] == "silvertorch":
        d768[(r["params"]["n_probe"], r["interleave"]["group"])][r["backend"]] = r
acc = defaultdict(list)
for (npb, _), d in d768.items():
    if len(d) == 2:
        for bs in (1, 16):
            acc[(npb, bs)].append(perf(d["official"], bs, "eager") / perf(d["triton"], bs, "eager"))
data["c7_d768"] = {npb: {bs: round(st.mean(acc[(npb, bs)]), 3) for bs in (1, 16)} for npb in sorted({k[0] for k in acc})}

# C7 at 30 M: ours with ST-TOPK (v2.11 library) vs Meta -O3 fp16 and int32, n_probe 128, bloom partial, eager, one process.
data["c7_30m"] = {}
for c in table("campaign-v2.10/st-topk-30m/st-topk-30m.md"):
    if len(c) > 11 and c[2] == "eager" and c[1].isdigit():
        data["c7_30m"][f"{c[0]}|{c[1]}|fp16"] = round(float(c[8]) / float(c[4]), 3)
        data["c7_30m"][f"{c[0]}|{c[1]}|int32"] = round(float(c[10]) / float(c[4]), 3)

(HERE / "data.json").write_text(json.dumps(data, indent=1, default=float))
print("wrote", HERE / "data.json")
