"""H2H-FINAL summary from the h2h JSONL: per (kind, k, bs, mode) the median over seeds of each arm's
median_ms, the median over seeds of official/triton paired by interleave group, parity fields,
and the unstable / below-max-clock counts. Usage: python h2h_summary.py <h2h/DATASET-d128.jsonl>"""
import json, statistics as st, sys
from collections import defaultdict

recs = [json.loads(l) for l in open(sys.argv[1])]
recs = [r for r in recs if r["status"] == "ok"]
arm = lambda r: r["backend"] + ("/" + r["params"]["score_path"] if "score_path" in r["params"] else "")
print("records ok:", len(recs), "unstable:", sum(r["unstable"] for r in recs))
groups = defaultdict(dict)
for r in recs:
    for e in r["perf"] or []:
        if e.get("median_ms") is not None:
            groups[(r["filter_kind"], e["k"], e["bs"], e["mode"], r["seed"])][arm(r)] = e["median_ms"]
cells = defaultdict(lambda: defaultdict(list))
for (fk, k, bs, mode, seed), arms in groups.items():
    for a, v in arms.items():
        cells[(fk, k, bs, mode)][a].append(v)
    t = arms.get("triton")
    for a in ("official/fp16", "official/int32"):
        if t and a in arms:
            cells[(fk, k, bs, mode)]["r_" + a].append(arms[a] / t)
print(f"{'kind':5} {'k':>4} {'bs':>2} {'mode':5} {'triton':>8} {'off_fp16':>8} {'off_i32':>8} {'fp16/tr':>12} {'i32/tr':>12}")
for key in sorted(cells):
    c = cells[key]
    m = lambda a: f"{st.median(c[a]):.3f}" if c.get(a) else "-"
    rr = lambda a: (f"{st.median(c[a]):.2f} [{min(c[a]):.2f},{max(c[a]):.2f}]" if c.get(a) else "-")
    print(f"{key[0]:5} {key[1]:>4} {key[2]:>2} {key[3]:5} {m('triton'):>8} {m('official/fp16'):>8} "
          f"{m('official/int32'):>8} {rr('r_official/fp16'):>12} {rr('r_official/int32'):>12}")
print("parity (official arms):")
par = defaultdict(list)
for r in recs:
    q = r["quality"]
    if r["backend"] == "official":
        par[(r["filter_kind"], arm(r))].append((q.get("parity"), q.get("jaccard_vs_first@100"),
                                                q.get("jaccard_vs_first@1000"), q.get("score_max_abs_diff")))
for key, v in sorted(par.items()):
    print(" ", key, "parity", sorted({x[0] for x in v}), "j@100 min", min(x[1] for x in v),
          "j@1000 min", min(x[2] for x in v), "max|dscore|", max(x[3] for x in v))
rec_q = defaultdict(list)
for r in recs:
    rec_q[(r["filter_kind"], arm(r))].append(r["quality"]["heldout"]["recall@100"])
print("heldout recall@100:", {k: round(st.median(v), 4) for k, v in sorted(rec_q.items())})
win = [s for r in recs for e in r["perf"] or [] for s in (e.get("window_sm_mhz") or [])]
print(f"windows {len(win)}, below 1410 MHz {sum(s < 1410 for s in win)}, min {min(win)}")
