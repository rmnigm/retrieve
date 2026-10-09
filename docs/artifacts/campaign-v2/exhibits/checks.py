"""Bug and sanity checks over every fetched record (README.md § checks): writes checks.md + checks.csv.

usage: checks.py OUT TREE [TREE ...]
"""

import collections
import csv
import json
import statistics as st
import sys
from pathlib import Path

from load import EXACT, arm, box, clock_unknown, cv, key, load, pass_p, perf, recall, short

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "evaluation"))
import yaml  # noqa: E402
from bench import config  # noqa: E402
from bench.run import exact_gate  # noqa: E402

CFG = Path(__file__).resolve().parents[4] / "evaluation" / "config"
rows = []  # (check, severity, where, numbers)


KNOWN = [  # (check, substring of where, reason): documented in validation / hub-index
    ("status", "d1:pubmed/filter", "V3 build OOM at 10 M x 768 (hub-index d1/pubmed)"),
    (
        "exact_recall",
        "d1:yfcc10m",
        "fp16 quanta: the YFCC gate is on @1000 (0.9944 since L1, validation)",
    ),
    (
        "triton_vs_official_recall",
        "yfcc10m",
        "official ~0.0105 below Triton on yfcc10m (validation)",
    ),
]


def flag(check, sev, where, numbers):
    why = [w for c, s, w in KNOWN if c == check and s in where]
    rows.append(
        (
            check,
            "known" if why else sev,
            where,
            numbers + (f" [known: {why[0]}]" if why else ""),
        )
    )


def med(x):
    x = [v for v in x if v is not None]
    return st.median(x) if x else None


def status(recs, md):
    md.append("## Status, partial, unstable per tree × suite × dataset\n")
    md.append(
        "| tree | suite | dataset | records | ok | failed | partial | unstable | clock-unknown | quality copies |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    c = collections.defaultdict(collections.Counter)
    for r in recs:
        g = c[(r["_tree"], r["suite"], r["dataset"])]
        g["n"] += 1
        g[r["status"]] += 1
        if clock_unknown(r):
            g["clock_unknown"] += 1
        else:
            g["unstable"] += bool(r.get("unstable"))
        g["copies"] += r.get("quality_source") is not None
        if r["status"] != "ok":
            flag(
                "status",
                "high" if r["status"] == "failed" else "mid",
                short(r),
                f"{r['status']} {r.get('partial_reasons') or r.get('stage')}",
            )
    for k, g in sorted(c.items()):
        md.append(
            f"| {' | '.join(map(str, k))} | {g['n']} | {g['ok']} | {g['failed']} | {g['partial']} "
            f"| {g['unstable']} | {g['clock_unknown']} | {g['copies']} |"
        )


def exact(recs, md):
    md.append("\n## Exact arms (V1, V2): recall_oracle ≥ the harness gate (`run.exact_gate`)\n")
    md.append("| tree | dataset | suite | arm | k | records | min recall | gate | worst record |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    g = collections.defaultdict(list)
    for r in recs:
        if r["algo"] in EXACT and r["status"] == "ok":
            for k in r.get("ks") or []:
                if recall(r, k) is not None:
                    g[
                        (
                            r["_tree"],
                            r["dataset"],
                            r["suite"],
                            f"{r['algo']}/{r['backend']}",
                            k,
                        )
                    ].append((recall(r, k), r))
    for k, v in sorted(g.items()):
        lo, r = min(v, key=lambda x: x[0])
        gate = exact_gate(config.load_dataset(CFG / f"{r['dataset']}.yaml", r["dim"]), r["k_max"])
        md.append(f"| {' | '.join(map(str, k))} | {len(v)} | {lo:.4f} | {gate} | {short(r)} |")
        if lo < gate:
            flag(
                "exact_recall",
                "high",
                short(r),
                f"recall_oracle@{k[-1]} {lo:.4f} < {gate}",
            )


def ids(recs, md):
    md.append("\n## Eager vs graph `ids_sha256` per (bs, k)\n")
    md.append(
        "Same ids from the eager module and the graph replay on the first 8 batches (D1-G's identity gate). "
        "Known: SilverTorch at `408b1188` differs in graph `quantize_int8` (redo ledger).\n"
    )
    md.append("| tree | suite | algo/backend | pairs | equal | differ | no graph |")
    md.append("|---|---|---|---|---|---|---|")
    g = collections.defaultdict(collections.Counter)
    for r in recs:
        for e in r.get("perf") or []:
            if e["mode"] != "eager" or not e.get("ids_sha256"):
                continue
            gr = [
                x
                for x in r["perf"]
                if x["mode"] == "graph" and x["bs"] == e["bs"] and x["k"] == e["k"]
            ]
            c = g[(r["_tree"], r["suite"], f"{r['algo']}/{r['backend']}")]
            c["pairs"] += 1
            if not gr or not gr[0].get("ids_sha256"):
                c["nograph"] += 1
            elif gr[0]["ids_sha256"] == e["ids_sha256"]:
                c["eq"] += 1
            else:
                c["ne"] += 1
                known = r["algo"] == "silvertorch" and cv(r) == "v2"
                flag(
                    "ids_eager_graph",
                    "known" if known else "high",
                    short(r) + f" bs{e['bs']} k{e['k']}",
                    f"{e['ids_sha256'][:10]} vs {gr[0]['ids_sha256'][:10]}"
                    + (" [known: graph quantize_int8 at 408b1188, V-GRAPH-IDS]" if known else ""),
                )
    for k, c in sorted(g.items()):
        md.append(f"| {' | '.join(k)} | {c['pairs']} | {c['eq']} | {c['ne']} | {c['nograph']} |")


def batching(recs, md):
    md.append("\n## Batching: median_ms(bs 16) < 16 × median_ms(bs 1)\n")
    n = bad = 0
    worst = []
    for r in recs:
        for k in r.get("ks") or []:
            for mode in ("eager", "graph"):
                a, b = perf(r, 1, k, mode), perf(r, 16, k, mode)
                if a and b:
                    n += 1
                    q = b["median_ms"] / a["median_ms"]
                    worst.append((q, short(r), k, mode, a["median_ms"], b["median_ms"]))
                    if q >= 16:
                        bad += 1
                        flag(
                            "batching",
                            "high",
                            short(r) + f" k{k} {mode}",
                            f"bs16/bs1 {q:.2f}",
                        )
    worst.sort(reverse=True)
    md.append(f"{n} (record, k, mode) pairs, **{bad} violate**. Largest bs16/bs1 ratios:\n")
    md.append("| bs16/bs1 | record | k | mode | bs1 ms | bs16 ms |")
    md.append("|---|---|---|---|---|---|")
    for w in worst[:8]:
        md.append(f"| {w[0]:.2f} | {w[1]} | {w[2]} | {w[3]} | {w[4]:.3f} | {w[5]:.3f} |")


def graph_vs_eager(recs, md):
    md.append("\n## Graph vs eager latency (graph / eager median, per record and (bs, k))\n")
    md.append(
        "A ratio > 1 means the CUDA-graph replay is slower than eager. `clk Δ` = graph `sm_mhz` − eager "
        "`sm_mhz` (MHz); `unst` = either entry `unstable`.\n"
    )
    md.append(
        "| tree | dataset | suite | algo/backend | bs | k | n | median g/e | min | max | n g>e+3 % | "
        "of which clk Δ≠0 or unst |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    g = collections.defaultdict(list)
    for r in recs:
        for e in r.get("perf") or []:
            if e["mode"] != "eager" or e.get("median_ms") is None:
                continue
            x = perf(r, e["bs"], e["k"], "graph")
            if x:
                odd = (x.get("sm_mhz") != e.get("sm_mhz")) or e.get("unstable") or x.get("unstable")
                g[
                    (
                        r["_tree"],
                        r["dataset"],
                        r["suite"],
                        f"{r['algo']}/{r['backend']}",
                        e["bs"],
                        e["k"],
                    )
                ].append((x["median_ms"] / e["median_ms"], odd))
    for k, v in sorted(g.items()):
        q = [a for a, _ in v]
        slow = [(a, o) for a, o in v if a > 1.03]
        md.append(
            f"| {' | '.join(map(str, k))} | {len(v)} | {st.median(q):.3f} | {min(q):.3f} | {max(q):.3f} "
            f"| {len(slow)} | {sum(o for _, o in slow)} |"
        )
        if st.median(q) > 1.03:
            flag(
                "graph_slower",
                "mid",
                " ".join(map(str, k)),
                f"median graph/eager {st.median(q):.3f} over {len(v)}",
            )


def monotone(recs, md):
    md.append("\n## Monotone where it must be\n")
    # V1 latency flat in p (synth)
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    pf = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        if r["suite"] != "synth" or r["status"] != "ok":
            continue
        if r["algo"] == "linr_v1_filter_mask" and r["backend"] == "triton":
            for e in r["perf"] or []:
                if e.get("median_ms"):
                    g[
                        (
                            r["_tree"],
                            r["dataset"],
                            r["filter_kind"],
                            e["bs"],
                            e["k"],
                            e["mode"],
                        )
                    ][pass_p(r)].append(e["median_ms"])
        if r["algo"] == "postfilter":
            for k in r.get("ks") or []:
                pf[
                    (
                        r["_tree"],
                        r["dataset"],
                        r["filter_kind"],
                        r["params"]["alpha"],
                        k,
                    )
                ][pass_p(r)].append(recall(r, k))
    md.append("**V1 triton latency vs p** (max / min of the per-p seed medians; > 1.10 flagged)\n")
    md.append("| tree | dataset | filter | bs | k | mode | max/min | at p (max, min) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for k, by in sorted(g.items()):
        m = {p: st.median(v) for p, v in by.items()}
        hi, lo = max(m, key=m.get), min(m, key=m.get)
        q = m[hi] / m[lo]
        md.append(f"| {' | '.join(map(str, k))} | {q:.3f} | {hi}, {lo} |")
        if q > 1.10:
            flag(
                "v1_flat",
                "mid",
                " ".join(map(str, k)),
                f"V1 max/min over p {q:.3f} (p {hi} vs {lo})",
            )
    md.append("\n**Postfilter recall non-decreasing in p** (seed medians)\n")
    md.append(
        "| tree | dataset | filter | alpha | k | recall from lowest to highest p | monotone |"
    )
    md.append("|---|---|---|---|---|---|---|")
    for k, by in sorted(pf.items()):
        ps = sorted(by)
        m = [med(by[p]) for p in ps]
        ok = all(b >= a - 1e-4 for a, b in zip(m, m[1:]))
        md.append(
            f"| {' | '.join(map(str, k))} | {' '.join(f'{x:.4f}' for x in m)} | {'yes' if ok else '**no**'} |"
        )
        if not ok:
            flag(
                "postfilter_monotone",
                "mid",
                " ".join(map(str, k)),
                " ".join(f"{x:.4f}" for x in m),
            )
    # SilverTorch recall rising with n_probe, per curve and seed
    curves = collections.defaultdict(list)
    for r in recs:
        if r["algo"] == "silvertorch" and "n_probe" in r["params"] and r["status"] == "ok":
            for k in r.get("ks") or []:
                if recall(r, k) is not None:
                    curves[key(r, drop=("n_probe",)) + (k, r["_tree"])].append(
                        (r["params"]["n_probe"], recall(r, k), r)
                    )
    n = drops = 0
    md.append(
        "\n**SilverTorch recall non-decreasing in n_probe** (per curve, seed and k; drops > 0.001 listed)\n"
    )
    lst = []
    for c, pts in curves.items():
        pts.sort(key=lambda x: x[0])
        if len(pts) < 2:
            continue
        n += 1
        for (a, ra, _), (b, rb, r) in zip(pts, pts[1:]):
            if rb < ra - 1e-4:
                drops += 1
                lst.append((ra - rb, short(r), c[-2], a, b, ra, rb))
    lst.sort(reverse=True)
    md.append(f"{n} curves, {drops} adjacent drops > 1e-4.\n")
    if lst:
        md.append("| drop | record (at the larger n_probe) | k | n_probe a → b | recall a → b |")
        md.append("|---|---|---|---|---|")
        for d in lst[:15]:
            md.append(
                f"| {d[0]:.4f} | {d[1]} | {d[2]} | {d[3]} → {d[4]} | {d[5]:.4f} → {d[6]:.4f} |"
            )
            if d[0] > 0.001:
                flag(
                    "st_nprobe_monotone",
                    "mid",
                    d[1],
                    f"k{d[2]} n_probe {d[3]}→{d[4]} recall {d[5]:.4f}→{d[6]:.4f}",
                )


def backends(recs, md):
    md.append("\n## Triton vs official: recall per matching cell\n")
    md.append(
        "Pairs differ only in backend (official's `score_path` dropped; bloomwidth not paired: official has "
        "no m_bits).\n"
    )
    by = collections.defaultdict(dict)
    for r in recs:
        if r["algo"] != "silvertorch" or r["suite"].startswith("bloomwidth") or r["status"] != "ok":
            continue
        k = list(key(r, drop=("score_path",)))
        k[6] = "*"
        by[tuple(k) + (r["_tree"],)][(r["backend"], r["params"].get("score_path"))] = r
    md.append("| tree | suite | dataset | pairs | k | max |Δ recall| | worst |")
    md.append("|---|---|---|---|---|---|---|")
    agg = collections.defaultdict(list)
    for k, d in by.items():
        t = d.get(("triton", None))
        if not t:
            continue
        for (b, sp), o in d.items():
            if b != "official":
                continue
            for kk in t.get("ks") or []:
                if recall(t, kk) is not None and recall(o, kk) is not None:
                    agg[(k[-1], k[2], k[0], kk)].append(
                        (
                            abs(recall(t, kk) - recall(o, kk)),
                            short(o),
                            recall(t, kk),
                            recall(o, kk),
                        )
                    )
    for k, v in sorted(agg.items()):
        w = max(v)
        md.append(
            f"| {k[0]} | {k[1]} | {k[2]} | {len(v)} | {k[3]} | {w[0]:.4f} | {w[1]} ({w[2]:.4f} vs {w[3]:.4f}) |"
        )
        if w[0] > 0.01:
            flag(
                "triton_vs_official_recall",
                "mid",
                w[1],
                f"k{k[3]} triton {w[2]:.4f} vs official {w[3]:.4f}",
            )


def cross(recs, md):
    md.append("\n## The same cell in two places (code_versions, or quality-only vs timed suite)\n")
    by = collections.defaultdict(list)
    for r in recs:
        if r["status"] != "ok":
            continue
        k = list(key(r))
        k[2] = "bloomwidth*" if k[2].startswith("bloomwidth") else k[2]
        by[tuple(k)].append(r)
    md.append(
        "| cell | a | b | k | recall a | recall b | eager bs16 ms a | b | graph bs16 ms a | b |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    summ = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, v in sorted(by.items()):
        if len(v) < 2:
            continue
        v.sort(key=lambda r: (r["env"]["started"], r["suite"]))
        for a, b, kk in [
            (a, b, kk) for a, b in zip(v, v[1:]) for kk in sorted(set(a["ks"]) & set(b["ks"]))
        ]:
            ra, rb = recall(a, kk), recall(b, kk)
            pe = [perf(x, 16, kk, "eager") for x in (a, b)]
            pg = [perf(x, 16, kk, "graph") for x in (a, b)]
            g = summ[(cv(a), a["suite"], cv(b), b["suite"], k[0], a["backend"], box(a), box(b))]
            g["n"].append(1)
            if ra is not None and rb is not None:
                g["dr"].append(abs(ra - rb))
            for m, pp in (("e", pe), ("g", pg)):
                if pp[0] and pp[1]:
                    g[m].append(pp[1]["median_ms"] / pp[0]["median_ms"])
            f = lambda e: f"{e['median_ms']:.3f}" if e else "-"  # noqa: E731
            md.append(
                f"| {k[0]}/{k[3]}/{k[4]}/{arm(a)}/s{k[8]} | {cv(a)}:{a['suite']} | {cv(b)}:{b['suite']} | {kk} "
                f"| {ra if ra is None else f'{ra:.4f}'} | {rb if rb is None else f'{rb:.4f}'} "
                f"| {f(pe[0])} | {f(pe[1])} | {f(pg[0])} | {f(pg[1])} |"
            )
            if ra is not None and rb is not None and abs(ra - rb) > 1e-6 and cv(a) == cv(b):
                flag(
                    "same_cell_recall",
                    "high",
                    f"{short(a)} vs {short(b)}",
                    f"k{kk} {ra:.5f} vs {rb:.5f}",
                )
    md.append("\n**Summary: b / a per pair of places** (bs 16 p50 medians; max |Δ recall|)\n")
    md.append(
        "| a | b | dataset | backend | box a / b | pairs | eager b/a | graph b/a | max abs Δ recall |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|")
    for (ca, sa, cb, sb, ds, be, xa, xb), g in sorted(summ.items()):
        f = lambda x: f"{st.median(x):.3f}" if x else "-"  # noqa: E731
        # timings compare only within one box (decisions, 2026-10-10); recall compares anywhere
        tm = (
            (f(g["e"]), f(g["g"]))
            if xa == xb
            else ("cross-box, not compared", "cross-box, not compared")
        )
        md.append(
            f"| {ca}:{sa} | {cb}:{sb} | {ds} | {be} | {xa} / {xb} | {len(g['n'])} | {tm[0]} | {tm[1]} "
            f"| {max(g['dr']) if g['dr'] else '-'} |"
        )
    md.append("\n**Co-design: partial and full recall must be equal (S-13)**\n")
    cd = collections.defaultdict(dict)
    for r in recs:
        if r["suite"] == "codesign":
            cd[key(r, drop=("bloom_path",)) + (r["_tree"],)][r["params"]["bloom_path"]] = r
    n = bad = 0
    for k, d in cd.items():
        if len(d) == 2:
            for kk in d["full"]["ks"]:
                n += 1
                if recall(d["full"], kk) != recall(d["partial"], kk):
                    bad += 1
                    flag(
                        "codesign_recall",
                        "high",
                        short(d["full"]),
                        f"full {recall(d['full'], kk)} vs partial {recall(d['partial'], kk)}",
                    )
    md.append(f"{n} pairs, {bad} unequal.")


def planned(recs, md):
    md.append("\n## Cells planned in suites.yaml (staging) but absent from every fetched tree\n")
    md.append(
        "Matched on dataset, suite, filter kind, sweep, algo, backend, params and seed (any code_version, "
        "inputs ignored).\n"
    )
    have = {key(r) for r in recs}
    md.append("| suite | dataset | planned | present | absent | absent by arm (first 6) |")
    md.append("|---|---|---|---|---|---|")
    suites = yaml.safe_load(open(CFG / "suites.yaml"))
    for suite, s in suites.items():
        if not isinstance(s, dict) or "datasets" not in s:
            continue
        for ds in s["datasets"]:
            jobs = config.load_matrix(CFG / f"{ds}.yaml", CFG / "suites.yaml", suite)
            keys = []
            for j in jobs:
                for p in j.cells():
                    kb = j.key(p)
                    keys.append(
                        (
                            kb["dataset"],
                            kb["dim"],
                            kb["suite"],
                            kb["filter_kind"],
                            kb["sweep"],
                            kb["algo"],
                            kb["backend"],
                            json.dumps(kb["params"], sort_keys=True),
                            kb["seed"],
                        )
                    )
            miss = [k for k in keys if k not in have]
            c = collections.Counter(
                f"{k[5]}/{k[6]}/{k[3]}" + (" " + k[7] if k[7] != "{}" else "") for k in miss
            )
            top = "; ".join(f"{a} ×{n}" for a, n in c.most_common(6))
            md.append(
                f"| {suite} | {ds} | {len(keys)} | {len(keys) - len(miss)} | {len(miss)} | {top} |"
            )


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    recs = load(sys.argv[2:])
    md = [
        "# Bug and sanity checks (NOT CITABLE)\n",
        f"{len(recs)} records from {', '.join(Path(t).name for t in sys.argv[2:])}; the latest record per resume "
        "key in each tree; trees are one code_version each (v2 = `408b1188`, v2.1 = `f01255f1`, d1 = `72e5a90` "
        "with 70 `c0e42d1` V3 bloom records in `deep`).\n",
    ]
    for f in (
        status,
        exact,
        ids,
        batching,
        graph_vs_eager,
        monotone,
        backends,
        cross,
        planned,
    ):
        f(recs, md)
    sev = {"high": 0, "mid": 1, "known": 2}
    rows.sort(key=lambda x: sev[x[1]])
    md.insert(
        2,
        f"\n**Findings: {sum(r[1] == 'high' for r in rows)} high, {sum(r[1] == 'mid' for r in rows)} mid, "
        f"{sum(r[1] == 'known' for r in rows)} known** (checks.csv lists each with its record).\n",
    )
    (out / "checks.md").write_text("\n".join(md) + "\n")
    with open(out / "checks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["check", "severity", "where", "numbers"])
        w.writerows(rows)
    print(md[2])


if __name__ == "__main__":
    main()
