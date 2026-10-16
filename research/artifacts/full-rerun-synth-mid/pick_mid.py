"""SYNTH-MID stage 2 picks, pick.py's rule at k 100 only (trimmed GO): per (dataset, sweep, n_lists) SilverTorch bloom curve the smallest
n_probe reaching oracle recall@100 0.90 and 0.95, plus the papers' 24 at the smallest n_lists (4096); a curve that stops short of
n_lists / 4 is skipped. LiNR V3 at matched recall: the smallest pool of fr-synth-mid-v3pool-d256 reaching 0.90 / 0.95 (a target no
pool reaches gets no V3 point). Writes fr-synth-mid-timed-d256 in pick.py's timed-suite shape: LiNR V1 / V2 exact and V3 interleaved
with SilverTorch bloom (recheck false), k 100, bs 1 / 16 / 64, seed 0, interleave by (algo, n_lists).
usage: pick_mid.py STAGE1_RESULTS_PARQUET > suites-synth-mid-timed.yaml"""
import json, sys
import polars as pl
import yaml

class NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

TARGETS = [0.90, 0.95]
df = pl.read_parquet(sys.argv[1]).filter((pl.col("status") == "ok") & (pl.col("seed") == 0))
p = lambda name, dt: pl.col("params").map_elements(lambda s: dt.to_python()(json.loads(s)[name]), return_dtype=dt)
v3 = (df.filter((pl.col("algo") == "linr_v3") & (pl.col("suite") == "fr-synth-mid-v3pool-d256"))
      .with_columns(pool=p("candidate_pool_frac", pl.Float64)).unique(["dataset", "sweep", "pool"], keep="first").sort("pool"))
v3_pools = {}
for (ds, sw), g in v3.group_by(["dataset", "sweep"]):
    pick = sorted({g.filter(pl.col("oracle_recall@100") >= t)["pool"][0] for t in TARGETS if (g["oracle_recall@100"] >= t).any()})
    if pick:
        v3_pools[ds, sw] = pick
st = (df.filter((pl.col("algo") == "silvertorch") & (pl.col("backend") == "official-fork") & (pl.col("filter_kind") == "bloom")
                & (pl.col("suite") == "fr-synth-mid-curve-d256"))
      .with_columns(n_lists=p("n_lists", pl.Int64), n_probe=p("n_probe", pl.Int64))
      .unique(["dataset", "sweep", "n_lists", "n_probe"], keep="first").sort("n_probe"))
smallest = {ds: g["n_lists"].min() for (ds,), g in st.group_by(["dataset"])}
pts = {}
for (ds, sw, nl), g in st.group_by(["dataset", "sweep", "n_lists"]):
    if g["n_probe"].max() < nl // 4:
        print(f"# skipped {ds} {sw} n_lists {nl}: curve ends at {g['n_probe'].max()}", file=sys.stderr); continue
    pick = {24} if nl == smallest[ds] else set()
    for t in TARGETS:
        ok = g.filter(pl.col("oracle_recall@100") >= t)
        if len(ok):
            pick.add(int(ok["n_probe"][0]))
    pts[ds, sw, int(nl)] = sorted(pick)
dss = sorted({k[0] for k in pts})
arms = [{"algo": "linr_v1_filter_mask", "backends": ["triton"], "filter_kinds": ["clause"]},
        {"algo": "linr_v2", "backends": ["triton"], "filter_kinds": ["clause"]}]
for ds, sw in sorted({(k[0], k[1]) for k in pts}):
    if (ds, sw) in v3_pools:
        arms.append({"algo": "linr_v3", "backends": ["triton"], "filter_kinds": ["clause"],
                     "datasets": {ds: {"sweeps": [sw], "query": {"candidate_pool_frac": v3_pools[ds, sw]}}}})
for (ds, sw, nl), probes in sorted(pts.items()):
    arms.append({"algo": "silvertorch", "backends": ["official-fork"], "filter_kinds": ["bloom"], "build": {"n_lists": [nl], "recheck": [False]},
                 "datasets": {ds: {"sweeps": [sw], "query": {"n_probe": probes}}}})
out = {"fr-synth-mid-timed-d256": {"datasets": dss, "dims": [256], "filter_kinds": ["bloom", "clause"], "ks": [100], "batch_sizes": [1, 16, 64],
       "seeds": [0], "sweeps": {d: sorted({k[1] for k in pts if k[0] == d}) for d in dss}, "interleave": [{"by": ["algo", "n_lists"]}], "arms": arms}}
yaml.dump(out, sys.stdout, sort_keys=False, width=140, Dumper=NoAlias)
print(f"# SilverTorch points {sum(map(len, pts.values()))} over {len(pts)} curves; V3 sweeps {len(v3_pools)}: {v3_pools}", file=sys.stderr)
