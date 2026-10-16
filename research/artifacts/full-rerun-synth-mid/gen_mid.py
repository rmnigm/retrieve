"""SYNTH-MID suites (user 2026-10-15, trimmed GO), derived from the FR tag's suites-lean.yaml so they cannot drift:
- fr-synth-mid-curve-d256 (stage 1): fr-curve-synth-s0-d256's SilverTorch bloom arms (recheck false) at n_lists 4096 and 16384 with
  their n_probe grids, laion10m-synth / laion30m-synth, sweeps p02 / p03 / p05, k 100, bs 16, quality only (trim: no clause, no k 1000,
  no 65536 lists).
- fr-synth-mid-v3pool-d256 (stage 1): fr-v3pool-laion's LiNR V3 pools on the same datasets / sweeps, k 100, bs 16, quality only, so the
  timed suite can run V3 at pick.py's matched-recall pools.
- fr-synth-mid-probesweep-d256 (stage 3): laion30m-synth, all seven synth filters, SilverTorch bloom (recheck false) at n_lists 4096
  (n_probe 24 ... 1024) and 16384 (24 ... 8192), interleaved with LiNR V1 (exact) as the time anchor, k 100, bs 16.
usage: gen_mid.py SUITES_LEAN_YAML > suites-synth-mid.yaml"""
import copy, sys
import yaml

class NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

src = yaml.safe_load(open(sys.argv[1]))
DS, MID = ["laion10m-synth", "laion30m-synth"], ["p02", "p03", "p05"]
curve = src["fr-curve-synth-s0-d256"]
arms = []
for a in curve["arms"]:
    nl = (a.get("build") or {}).get("n_lists", [None])[0]
    if a.get("filter_kinds") != ["bloom"] or nl not in (4096, 16384):
        continue
    b = copy.deepcopy(a)
    b["datasets"] = {d: {"sweeps": MID, "query": a["datasets"][d]["query"]} for d in DS}
    arms.append(b)
assert len(arms) == 2 and all(x["build"]["recheck"] == [False] for x in arms), arms
out = {"fr-synth-mid-curve-d256": {
    "perf": False, "datasets": DS, "dims": [256], "filter_kinds": ["bloom"], "ks": [100], "batch_sizes": [16], "seeds": [0],
    "sweeps": {d: MID for d in DS}, "arms": arms}}
v3 = [copy.deepcopy(a) for a in src["fr-v3pool-laion"]["arms"] if a["algo"] == "linr_v3"]
assert len(v3) == 1
out["fr-synth-mid-v3pool-d256"] = {
    "perf": False, "datasets": DS, "dims": [256], "filter_kinds": ["clause"], "ks": [100], "batch_sizes": [16], "seeds": [0],
    "sweeps": {d: MID for d in DS}, "arms": [{**v3[0], "filter_kinds": ["clause"]}]}
ALL = ["p0001", "p001", "p01", "p1"] + MID
GRID = {4096: [24, 32, 64, 128, 256, 512, 1024], 16384: [24, 64, 256, 1024, 2048, 4096, 8192]}
out["fr-synth-mid-probesweep-d256"] = {
    "datasets": ["laion30m-synth"], "dims": [256], "filter_kinds": ["bloom", "clause"], "ks": [100], "batch_sizes": [16], "seeds": [0],
    "sweeps": {"laion30m-synth": ALL}, "interleave": [{"by": ["algo", "n_lists"]}],
    "arms": [{"algo": "linr_v1_filter_mask", "backends": ["triton"], "filter_kinds": ["clause"]}] + [
        {"algo": "silvertorch", "backends": ["official-fork"], "filter_kinds": ["bloom"], "build": {"n_lists": [nl], "recheck": [False]},
         "datasets": {"laion30m-synth": {"sweeps": ALL, "query": {"n_probe": np}}}} for nl, np in GRID.items()]}
yaml.dump(out, sys.stdout, sort_keys=False, width=140, Dumper=NoAlias)
