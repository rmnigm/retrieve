import json, sys, polars as pl
def load(path):
    cells, perf = [], []
    for i, line in enumerate(open(path)):
        r = json.loads(line)
        q = r.get("quality") or {}
        base = {k: r.get(k) for k in ["status","dataset","dim","suite","filter_kind","sweep","algo","backend","seed","path","n_items","n_queries","n_kept","pass_rate","build_s","index_mib","filter_mib","bloom_fp_rate"]}
        base["line"] = i
        base["params"] = json.dumps(r.get("params"), sort_keys=True)
        for kk, v in (r.get("params") or {}).items(): base["p_"+kk] = v
        base["partial_reasons"] = json.dumps(r.get("partial_reasons"))
        for sect in ["heldout","oracle"]:
            for kk, v in (q.get(sect) or {}).items(): base[f"{sect}_{kk}"] = v
        for kk in ["score_max_abs_diff","parity"] + [f"jaccard_vs_first@{k}" for k in (100,200,400)]:
            base[kk] = q.get(kk)
        base["error"] = str(r.get("error"))[:200] if r.get("error") else None
        cells.append(base)
        for p in r.get("perf") or []:
            d = dict(base); d.update({k: v for k, v in p.items() if not isinstance(v, list)})
            d["wmed"] = p.get("window_medians_ms"); d["wmhz"] = p.get("window_sm_mhz")
            perf.append(d)
    return pl.DataFrame(cells, infer_schema_length=None), pl.DataFrame(perf, infer_schema_length=None)
