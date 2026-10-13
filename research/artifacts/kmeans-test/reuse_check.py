"""km-sanity vs the existing random-10 records (handoff reuse rule): oracle recall@100 / @1000 equal, no tolerance.
clause: numbers/A1-laion30m (official-fork); bloom: numbers/A1-bloom n-bloom-curve-d256 laion30m (recheck false)."""
import glob, json, sys

def recs(pattern):
    out = []
    for f in glob.glob(pattern, recursive=True):
        out += [json.loads(l) for l in open(f)]
    return out

def key(r):
    p = r["params"]
    return (r["filter_kind"], r["sweep"], p.get("n_lists"), p.get("n_probe"))

km = [r for r in recs("/scratch/kmeans-test/out/km-sanity/*.jsonl") if r["status"] in ("ok", "partial")]
ref = [r for r in recs("/scratch/kmeans-test/ref/A1-laion30m/**/laion30m-d256.jsonl") if r["algo"] == "silvertorch" and r["backend"] == "official-fork" and r["filter_kind"] == "clause"]
ref += [r for r in recs("/scratch/kmeans-test/ref/A1-bloom/n-bloom-curve-d256/laion30m-d256.jsonl") if r["filter_kind"] == "bloom" and r["params"].get("recheck") is False]
latest = {}
for r in ref:
    latest[key(r)] = r  # newest per key wins, as aggregation does
ok = True
rows = []
for r in km:
    k = key(r)
    a = r["quality"]["oracle"]
    b = latest.get(k, {}).get("quality", {}).get("oracle")
    for kk in ("recall@100", "recall@1000"):
        eq = b is not None and a.get(kk) == b.get(kk)
        ok &= eq
        rows.append({"filter_kind": k[0], "sweep": k[1], "n_lists": k[2], "n_probe": k[3], "metric": kk,
                     "km_sanity": a.get(kk), "reference": None if b is None else b.get(kk), "equal": eq})
for x in rows:
    print(x)
print("REUSE", "PASS" if ok and rows else "FAIL", f"({sum(x['equal'] for x in rows)} / {len(rows)} equal)")
json.dump({"pass": bool(ok and rows), "rows": rows}, open(sys.argv[1], "w"), indent=1)
