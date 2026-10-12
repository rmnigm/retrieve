"""V-PUBMED identity: the restaged V1 clause c0_mesh seed-0 cell against d1/pubmed's.

Every slice fact and every quality metric at the ks both records carry must be equal (exact
V1 on the same embeddings is deterministic). Exit 1 on any difference.
"""

import json
import sys

FACTS = ("n_items", "n_queries", "n_kept", "n_queries_heldout", "n_targets_in_filter",
         "n_queries_oracle", "pass_rate")  # fmt: skip


def cell(path):
    for line in open(path):
        r = json.loads(line)
        if (r["algo"], r["backend"], r["filter_kind"], r["sweep"], r["seed"]) == (
            "linr_v1_filter_mask", "triton", "clause", "c0_mesh", 0):  # fmt: skip
            return r
    raise SystemExit(f"no V1 c0_mesh seed 0 cell in {path}")


old, new = cell(sys.argv[1]), cell(sys.argv[2])
diffs = [f"{f}: {old[f]} vs {new[f]}" for f in FACTS if old[f] != new[f]]
n = 0
for side in ("heldout", "oracle"):
    for m, v in new["quality"][side].items():
        if m in old["quality"][side]:
            n += 1
            print(f"{side}.{m}: d1 {old['quality'][side][m]!r} new {v!r}")
            if old["quality"][side][m] != v:
                diffs.append(f"{side}.{m}: {old['quality'][side][m]} vs {v}")
print(f"code_version d1 {old.get('code_version')} new {new.get('code_version')}; {n} metrics compared")
print("EQUAL" if not diffs else "DIFFERENT:\n" + "\n".join(diffs))
sys.exit(1 if diffs else 0)
