"""FR-GATE step 4 (D1-G b): every rerun record of quality_legs.py against the FR records with the same key block, in
every merged tree that holds one (the accepted record and any superseded `ok` line): the `quality` block equal (without
the parity-group fields, which depend on which backend of the group ran first: D1-G's own rule) and the per-query sidecar
arrays equal. A chosen cell the
driver skipped (its oracle is not cached on this pod) is counted as skipped; any failure or difference is a mismatch. Prints a per-leg table; exit 1 on any mismatch.

usage: compare_legs.py TREES_DIR RERUN_DIR CHOSEN_JSON [CHOSEN_JSON...] [--out gate_b.json]
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench import records

MARK = "FR-GATE: oracle not cached on this pod; cell skipped"
args = sys.argv[1:]
trees, rerun = Path(args[0]), Path(args[1])
out = Path(args[args.index("--out") + 1]) if "--out" in args else None
chosen_files = [
    a for a in args[2:] if not a.startswith("--") and (out is None or Path(a) != out)
]
chosen = [c for a in chosen_files for c in json.loads(Path(a).read_text())]
skipped = Counter(c["leg"] for c in chosen if c.get("skipped"))


def key(rec):
    return json.dumps(records.key_block(rec), sort_keys=True)


PARITY_GROUP = (
    "parity",
    "jaccard_vs_first@100",
    "jaccard_vs_first@1000",
    "jaccard_vs_first@2048",
    "score_max_abs_diff",
)


def quality(rec):
    """The quality block without the parity-group fields: a rerun of one arm is its parity group's first backend and
    has nothing to compare against (the D1-G byte-identical-rerun clause, research/artifacts/evaluation-harness-v2/d1a)."""
    return {
        k: v for k, v in (rec.get("quality") or {}).items() if k not in PARITY_GROUP
    }


leg_of = {json.dumps(c["key"], sort_keys=True): c["leg"] for c in chosen}
fr = defaultdict(list)
for sub in sorted(p for p in trees.iterdir() if p.is_dir()):
    for p in records.record_files(sub):
        for r in records.read_records(p):
            if r["status"] in ("ok", "partial") and r.get("quality"):
                fr[key(r)].append((sub, r))
new = [
    r
    for t in rerun.iterdir()
    if t.is_dir()
    for p in records.record_files(t)
    for r in records.read_records(p)
]
per_leg, bad = defaultdict(Counter), []
for r in new:
    k = key(r)
    leg = leg_of.get(k, "?")
    if r["status"] == "failed":
        if MARK in (r.get("error") or ""):
            per_leg[leg]["skipped_no_oracle"] += 1
        else:
            per_leg[leg]["failed"] += 1
            bad.append(
                f"FAILED [{leg}] {k[:150]}: {(r.get('error') or '').strip().splitlines()[-1][:160]}"
            )
        continue
    if k not in fr:
        per_leg[leg]["no_fr_record"] += 1
        bad.append(f"NO FR RECORD [{leg}] {k[:150]}")
        continue
    per_leg[leg]["compared"] += 1
    for sub, f in fr[k]:
        if quality(r) != quality(f):
            per_leg[leg]["quality_differs"] += 1
            bad.append(
                f"QUALITY DIFFERS [{leg}] vs {sub.name}: {k[:150]}\n  fr  {quality(f)}\n  new {quality(r)}"
            )
            continue
        src = next(t for t in rerun.iterdir() if (t / r["per_query"]).exists())
        a, b = np.load(src / r["per_query"]), np.load(sub / f["per_query"])
        if set(a.files) != set(b.files) or any(
            not np.array_equal(a[n], b[n], equal_nan=True) for n in a.files
        ):
            per_leg[leg]["sidecar_differs"] += 1
            bad.append(f"SIDECAR DIFFERS [{leg}] vs {sub.name}: {k[:150]}")
        else:
            per_leg[leg]["equal"] += 1
chosen_legs = Counter(c["leg"] for c in chosen)
for leg in sorted(chosen_legs):
    c = per_leg[leg]
    c["skipped_no_oracle"] += skipped[leg]
    print(
        f"{leg:48s} chosen {chosen_legs[leg]:3d}  compared {c['compared']:3d}  equal {c['equal']:3d}  skipped {c['skipped_no_oracle']:3d}  "
        f"bad {c['failed'] + c['no_fr_record'] + c['quality_differs'] + c['sidecar_differs']}"
    )
tot = sum((per_leg[lg] for lg in per_leg), Counter())
print(f"{len(new)} rerun records: {dict(tot)}; {len(bad)} mismatch(es)")
for b in bad:
    print(b)
if out:
    out.write_text(
        json.dumps(
            {
                "per_leg": {k: dict(v) for k, v in per_leg.items()},
                "chosen": dict(chosen_legs),
                "totals": dict(tot),
                "bad": bad,
            },
            indent=1,
        )
    )
sys.exit(1 if bad else 0)
