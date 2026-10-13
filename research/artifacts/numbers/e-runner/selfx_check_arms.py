import sys
from collections import Counter
from bench.config import load_matrix
CFG = "/scratch/e/selfx/config"
LEGS = [("n-curve-q", ["arxiv", "arxiv-synth", "arxiv-corr-synth"]), ("n-bloom-curve-d128", ["arxiv"]),
        ("n-bloom-curve-synth-d128", ["arxiv-synth"]), ("n-st-k2048-d128", ["arxiv"]), ("n-bloomwidth", ["arxiv"]),
        ("n-v3pool", ["arxiv"])]
bad = []
for suite, dss in LEGS:
    for ds in dss:
        jobs = load_matrix(f"{CFG}/{ds}.yaml", f"{CFG}/suites.yaml", suite, dims=[128])
        c = Counter()
        for j in jobs:
            n = sum(1 for _ in j.params) if hasattr(j, "params") else 1
            c[(j.algo, j.backend, j.filter_kind, str(j.build.get("recheck")))] += 1
            if j.algo.startswith("linr") and j.filter_kind != "clause":
                bad.append((suite, ds, j.algo, j.filter_kind))
            if j.algo == "silvertorch" and j.filter_kind == "bloom" and j.build.get("recheck") is not False:
                bad.append((suite, ds, j.algo, "bloom recheck", j.build.get("recheck")))
        print(suite, ds, dict(c))
print("BAD", bad)
sys.exit(1 if bad else 0)
