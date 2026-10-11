import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "/workspace/retrieve/evaluation")
from bench.records import latest
UP = Path("/scratch/campaigns/upload")
def perf(r, bs, k=100):
    return next(p for p in r["perf"] if p["k"] == k and p["mode"] == "eager" and p["bs"] == bs)
for ds in ["goodreads", "arxiv", "yfcc10m", "pubmed"]:
    cells = defaultdict(dict)
    for r in latest(UP / ds):
        if r["status"] != "ok" or r["algo"] != "silvertorch": continue
        cells[(r["filter_kind"], r["sweep"], r["seed"], r["params"].get("n_probe"))][r["backend"]] = r
    print(f"## {ds}")
    print("fk sweep seed np | off_ms1 tri_ms1 | off_qps16 tri_qps16 ratio | off_rec tri_rec | off_mib tri_mib")
    for key in sorted(cells, key=str):
        c = cells[key]
        o, t = c.get("official"), c.get("triton")
        f = lambda r, fn: fn(r) if r else float("nan")
        oq, tq = f(o, lambda r: perf(r,16)["qps"]), f(t, lambda r: perf(r,16)["qps"])
        print(key, "| %.3f %.3f | %.0f %.0f %.3f | %.4f %.4f | %.0f %.0f" % (
            f(o, lambda r: perf(r,1)["median_ms"]), f(t, lambda r: perf(r,1)["median_ms"]),
            oq, tq, oq/tq, f(o, lambda r: r["quality"]["oracle"]["recall@100"]), f(t, lambda r: r["quality"]["oracle"]["recall@100"]),
            f(o, lambda r: perf(r,16)["peak_fwd_mib"]), f(t, lambda r: perf(r,16)["peak_fwd_mib"])))
