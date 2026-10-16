"""FR-GATE steps 2, 3a, 3c, 3d: the plan against `bench merge`'s tables (README.md).

- Coverage: every record-bearing subtree of `manifest.yaml` is claimed by exactly one leg of `legs.yaml` (as the leg or
  one of its `-rerun` supplements); every `files` leg's listed files exist in its tree.
- Cells: every planned cell has exactly one accepted record, `ok`, in one of its leg's subtrees, at the leg's tag; every
  accepted record is a planned cell. Anything else is a gap, listed by cell. Superseded `failed` lines (the OOM
  fallbacks) and accepted `unstable` cells are counted, not gaps (FR policy).
- D1-G (a): on every accepted record timed in both modes, the eager and graph `ids_sha256` are equal per (k, bs).
- D1-G (c): on every accepted record with bs 1 and bs 16 timed, `median_ms(bs 16) < 16 x median_ms(bs 1)` per (k, mode); a
  failure the user ruled a known exception (`legs.yaml` `batch_scaling_exceptions`) is listed with its ruling, never dropped.
- D1-G (d): `bench report --manifest` is gone; `manifest.yaml` stands in: every subtree's `manifest_sha256` is the
  hub-index's (verify_trees.py), and every (dataset, suite, algo, backend) group's subtrees are the plan's.

usage: gate.py LEGS_YAML TABLES_DIR TREES_DIR OUT_DIR PLAN_JSONL [PLAN_JSONL...]   (writes gate.json, gate.md)
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml

legs_yaml, tables, trees, out = (Path(a) for a in sys.argv[1:5])
plans = [Path(a) for a in sys.argv[5:]]
out.mkdir(parents=True, exist_ok=True)
legs = yaml.safe_load(legs_yaml.read_text())
manifest = yaml.safe_load((tables / "manifest.yaml").read_text())
records = pq.read_table(tables / "records.parquet").to_pylist()
timings = pq.read_table(tables / "timings.parquet").to_pylist()
KEY = (
    "dataset",
    "dim",
    "inputs",
    "suite",
    "filter_kind",
    "sweep",
    "algo",
    "backend",
    "params",
    "seed",
)


def ckey(d):
    k = {f: d[f] for f in KEY}
    if d.get("self_excluded"):
        k["self_excluded"] = True
    return json.dumps(k, sort_keys=True)


def cell_label(k):
    d = json.loads(k)
    return f"{d['dataset']} d{d['dim']} {d['suite']} {d['algo']}/{d['backend']} {d['filter_kind']}/{d['sweep']} seed {d['seed']} {d['params']}"


gaps, notes = [], []
# ---- coverage
claim = {}
for leg in legs["legs"]:
    for s in [leg["name"], *leg.get("supplements", [])]:
        if s in claim:
            gaps.append(f"subtree {s} claimed by two legs: {claim[s]}, {leg['name']}")
        claim[s] = leg["name"]
record_subtrees = {s for s, m in manifest["subtrees"].items() if m["records"]}
for s in sorted(record_subtrees - set(claim)):
    gaps.append(f"record subtree {s} is not claimed by any leg of legs.yaml")
for s in sorted(set(claim) - set(manifest["subtrees"])):
    leg = next(lg for lg in legs["legs"] if lg["name"] == claim[s])
    if not leg.get("never_run"):
        gaps.append(f"planned subtree {s} is not among the merged trees")
files_state = {}
for leg in legs["legs"]:
    if "files" not in leg:
        continue
    tree = trees / leg["name"]
    missing = [f for f in leg["files"] if not (tree / f).exists()]
    short = {
        d: len(list((tree / d).iterdir()))
        for d, n in (leg.get("min_files_in") or {}).items()
        if (tree / d).is_dir() and len(list((tree / d).iterdir())) < n
    }
    files_state[leg["name"]] = {
        "missing": missing,
        "short": short,
        "note": leg.get("note"),
    }
    gaps += [f"{leg['name']}: file {f} missing" for f in missing]
    gaps += [
        f"{leg['name']}: {d} holds {n} files, fewer than the {leg['min_files_in'][d]} planned"
        for d, n in short.items()
    ]

# ---- planned cells vs accepted records
plan = {}
for p in plans:
    for line in p.read_text().splitlines():
        r = json.loads(line)
        k = ckey(r["key"])
        if k in plan:
            gaps.append(
                f"planned twice ({plan[k]['leg']}, {r['leg']}): {cell_label(k)}"
            )
        plan[k] = r
accepted = defaultdict(list)
for r in records:
    if r["accepted"]:
        accepted[ckey(r)].append(r)
status_acc = Counter(r["status"] for r in records if r["accepted"])
unstable_acc = sum(1 for r in records if r["accepted"] and r["unstable"])
superseded = Counter(r["status"] for r in records if not r["accepted"])
leg_rows = defaultdict(lambda: Counter())
never_run = []
for k, pr in plan.items():
    recs = accepted.get(k, [])
    leg_rows[pr["leg"]]["planned"] += 1
    if pr["never_run"]:
        never_run.append(cell_label(k))
        if recs:
            gaps.append(f"never-run cell has a record: {cell_label(k)}")
        continue
    if not recs:
        gaps.append(
            f"planned cell without an accepted record [{pr['leg']}]: {cell_label(k)}"
        )
        continue
    if len(recs) > 1:
        gaps.append(
            f"two accepted records for one cell: {cell_label(k)} in {[r['subtree'] for r in recs]}"
        )
    r = recs[0]
    leg_rows[pr["leg"]][r["status"]] += 1
    if r["unstable"]:
        leg_rows[pr["leg"]]["unstable"] += 1
    if r["status"] != "ok":
        gaps.append(
            f"accepted record is {r['status']} ({r.get('error') or r.get('partial_reasons')}) [{r['subtree']}]: {cell_label(k)}"
        )
    if r["subtree"] not in pr["subtrees"]:
        gaps.append(
            f"accepted record from {r['subtree']}, the plan's leg is {pr['leg']}: {cell_label(k)}"
        )
    if r["tag"] != pr["tag"]:
        gaps.append(
            f"accepted record at tag {r['tag']} ({r['env_commit']}), the plan's is {pr['tag']} [{r['subtree']}]: {cell_label(k)}"
        )
for k, recs in accepted.items():
    if k not in plan:
        gaps.append(
            f"accepted record no plan lists [{recs[0]['subtree']}]: {cell_label(k)}"
        )

# ---- D1-G (a) eager vs graph ids, (c) batch scaling, on accepted records
acc_cells = {r["cell"] for r in records if r["accepted"]}
by_cell = defaultdict(list)
for t in timings:
    if t["accepted"] and t["cell"] in acc_cells and t.get("perf_n") is not None:
        by_cell[t["cell"]].append(t)
ids_pairs = ids_bad = 0
ids_bad_list = []
scale_pairs = 0
scale_worst = 0.0
scale_bad, scale_known = [], []
for cell, ts in by_cell.items():
    by_kb = defaultdict(dict)
    by_km = defaultdict(dict)
    for t in ts:
        by_kb[t["perf_k"], t["perf_bs"]][t["perf_mode"]] = t
        by_km[t["perf_k"], t["perf_mode"]][t["perf_bs"]] = t
    for (k, bs), modes in by_kb.items():
        if "eager" in modes and "graph" in modes:
            ids_pairs += 1
            if modes["eager"]["perf_ids_sha256"] != modes["graph"]["perf_ids_sha256"]:
                ids_bad += 1
                ids_bad_list.append(f"{cell_label(ckey(ts[0]))} k {k} bs {bs}")
    for (k, mode), bss in by_km.items():
        if 1 in bss and 16 in bss:
            scale_pairs += 1
            ratio = bss[16]["perf_median_ms"] / bss[1]["perf_median_ms"]
            scale_worst = max(scale_worst, ratio)
            if not ratio < 16:
                t = ts[0]
                exc = next(
                    (
                        e
                        for e in legs.get("batch_scaling_exceptions") or []
                        if all(
                            str(t[f]) == str(e[f])
                            for f in (
                                "subtree",
                                "dataset",
                                "suite",
                                "algo",
                                "backend",
                                "sweep",
                                "params",
                            )
                        )
                        and e["k"] == k
                        and e["mode"] == mode
                    ),
                    None,
                )
                line = (
                    f"{t['subtree']}: {cell_label(ckey(t))} k {k} {mode}: {bss[16]['perf_median_ms']:.3f} / {bss[1]['perf_median_ms']:.3f} ms = {ratio:.2f}x"
                    + (" (unstable)" if t["unstable"] else "")
                )
                (scale_known if exc else scale_bad).append(
                    line + (f" -- known exception: {exc['ruling']}" if exc else "")
                )

# ---- D1-G (d) manifest
verify = json.loads((trees / "verify_trees.json").read_text())
man_bad = []
for s, m in manifest["subtrees"].items():
    v = verify["trees"].get(f"fr/{s}")
    if v is None or not v["hub_index_sha256_matches"] or v["file_mismatches"]:
        man_bad.append(f"{s}: not verified against the Hub")
plan_groups = defaultdict(set)
for k, pr in plan.items():
    d = json.loads(k)
    plan_groups[d["dataset"], d["suite"], d["algo"], d["backend"]].update(
        pr["subtrees"]
    )
for g in manifest["cells"]:
    key = (g["dataset"], g["suite"], g["algo"], g["backend"])
    extra = set(g["subtrees"]) - plan_groups.get(key, set())
    if extra:
        man_bad.append(
            f"group {key}: accepted subtrees {sorted(extra)} outside the plan"
        )
gaps += man_bad

summary = {
    "planned_cells": len(plan),
    "planned_never_run": never_run,
    "legs": len(leg_rows),
    "subtrees": len(manifest["subtrees"]),
    "record_subtrees": len(record_subtrees),
    "records_total": len(records),
    "accepted_by_status": dict(status_acc),
    "accepted_unstable": unstable_acc,
    "superseded_by_status": dict(superseded),
    "d1g_a_ids_pairs": ids_pairs,
    "d1g_a_ids_mismatches": ids_bad,
    "d1g_a_mismatch_list": ids_bad_list,
    "d1g_c_pairs": scale_pairs,
    "d1g_c_worst_ratio": round(scale_worst, 3),
    "d1g_c_failures": scale_bad,
    "d1g_c_known_exceptions": scale_known,
    "d1g_d_manifest_problems": man_bad,
    "files_legs": files_state,
    "gaps": gaps,
    "per_leg": {k: dict(v) for k, v in sorted(leg_rows.items())},
}
(out / "gate.json").write_text(json.dumps(summary, indent=1))
md = [
    "# FR gate (gate.py)",
    "",
    f"- planned cells {len(plan)} in {len(leg_rows)} legs ({len(never_run)} planned, never run); merged subtrees {len(manifest['subtrees'])}, "
    f"{len(record_subtrees)} with records, {len(records)} record lines",
    f"- accepted records by status: {dict(status_acc)}; accepted unstable: {unstable_acc}; superseded lines by status: {dict(superseded)}",
    f"- D1-G (a) eager vs graph ids: {ids_pairs} (cell, k, bs) pairs, {ids_bad} mismatches",
    f"- D1-G (c) batch scaling bs 16 < 16 x bs 1: {scale_pairs} (cell, k, mode) pairs, worst {scale_worst:.2f}x, {len(scale_bad)} failures, {len(scale_known)} listed known exceptions",
    f"- D1-G (d) manifest: {len(man_bad)} problems",
    f"- gaps: {len(gaps)}",
    "",
    "## Gaps",
    "",
    *([f"- {g}" for g in gaps] or ["none"]),
    "",
    "## D1-G (c) failures",
    "",
    *([f"- {g}" for g in scale_bad] or ["none"]),
    "",
    "## D1-G (c) known exceptions (user rulings)",
    "",
    *([f"- {g}" for g in scale_known] or ["none"]),
    "",
    "## Per leg (planned / ok / unstable)",
    "",
    "| leg | planned | ok | partial | unstable |",
    "|---|---|---|---|---|",
    *[
        f"| {k} | {v['planned']} | {v.get('ok', 0)} | {v.get('partial', 0)} | {v.get('unstable', 0)} |"
        for k, v in sorted(leg_rows.items())
    ],
]
(out / "gate.md").write_text("\n".join(md) + "\n")
print("\n".join(md[2:9]))
print(
    f"{len(gaps)} gap(s); {len(scale_bad)} batch-scaling failure(s) ({len(scale_known)} known exceptions); {ids_bad} ids mismatch(es)"
)
for g in gaps[:60]:
    print("GAP", g)
sys.exit(1 if gaps or ids_bad or scale_bad else 0)
