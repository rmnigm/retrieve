"""The final pass's one table (README.md): every record under the fetched `campaign-final/*` subtrees, deduplicated by
record key, into one merged results tree and its `results.parquet` (`records.aggregate`).

Within one subtree the last line per key wins (`records.latest`'s rule: a later run of a cell supersedes an earlier one in
the same results tree, e.g. M1's gate runs before d0-03's interleaved V1 / V2 leg); those superseded lines are counted. A
leg's upload carries the earlier legs' records of the same suite file (upload_command), so a key appears in several
subtrees: every copy must be the same JSON line, byte for byte; any that is not is listed and fails the run. Every record
must carry the tag's library code_version. Per-query sidecars are hardlinked into the merged tree (same name = same key;
equal content asserted). The recipe's legs are checked against what arrived: each upload path present, its record count
equal to `expect.records`.

A record from a PREFER subtree (comma-separated) wins over other copies of its key that differ; those are listed as
superseded. It is for subtrees that restore a leg's own records after another upload replaced them at the leg's path.

usage: aggregate.py LEGS_DIR OUT_DIR RECIPE.yaml CODE_VERSION [PREFER]
"""

import collections
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "evaluation"))
from bench import records  # noqa: E402


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    legs, out, recipe, cv = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        sys.argv[4],
    )
    prefer = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else set()
    by_prefer = []  # (key, kept subtree, superseded subtree)
    tree = out / "tree"
    if tree.exists():
        raise SystemExit(f"{tree} exists: start from an empty OUT_DIR")
    tree.mkdir(parents=True)
    seen = {}  # key -> (line, first subtree, (suite, file name))
    per_leg = collections.Counter()
    dup, bad, wrong_cv = collections.Counter(), [], []
    status, superseded = collections.Counter(), collections.Counter()
    for sub in sorted(p for p in legs.iterdir() if p.is_dir()):
        last = {}  # key -> (line, record, (suite, file)): the subtree's latest record per key
        for f in sorted(sub.glob("*/*.jsonl")):
            if f.name.endswith(".samples.jsonl"):
                continue
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                k = records.record_key(r)
                superseded[sub.name] += k in last
                last[k] = (line, r, (f.parent.name, f.name))
        for k, (line, r, where) in last.items():
            per_leg[sub.name] += 1
            if not r["env"]["code_version"].startswith(cv):
                wrong_cv.append((sub.name, k))
            if k in seen:
                dup[sub.name] += 1
                if line != seen[k][0]:
                    if sub.name in prefer and seen[k][1] not in prefer:
                        by_prefer.append((k, sub.name, seen[k][1]))
                        seen[k] = (line, sub.name, where)
                    elif seen[k][1] in prefer and sub.name not in prefer:
                        by_prefer.append((k, seen[k][1], sub.name))
                    else:
                        bad.append((k, seen[k][1], sub.name))
                continue
            seen[k] = (line, sub.name, where)
            status[r["status"]] += 1
    by_file = collections.defaultdict(list)
    for line, _, where in seen.values():
        by_file[where].append(line)
    for (suite, name), lines in by_file.items():
        (tree / suite).mkdir(exist_ok=True)
        (tree / suite / name).write_text("\n".join(lines) + "\n")
    # sidecars and samples: per-query files hardlinked (equal content asserted), sample lines unioned
    clash = []
    for sub in sorted(p for p in legs.iterdir() if p.is_dir()):
        for pq in sub.glob("*/*.perquery"):
            dst = tree / pq.parent.name / pq.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in pq.iterdir():
                t = dst / f.name
                if t.exists():
                    if digest(t) != digest(f):
                        clash.append(str(f.relative_to(legs)))
                else:
                    os.link(f, t)
        for s in sub.glob("*/*.samples.jsonl"):
            dst = tree / s.parent.name / s.name
            have = set(dst.read_text().splitlines()) if dst.exists() else set()
            new = [x for x in s.read_text().splitlines() if x.strip() and x not in have]
            if new:
                with open(dst, "a") as fh:
                    fh.write("\n".join(new) + "\n")
    parquet = records.aggregate(tree)

    # the recipe against what arrived
    rec = yaml.safe_load(recipe.read_text())
    want = collections.defaultdict(int)
    for lg in rec["legs"]:
        sub = lg["upload"].split("/", 1)[1]
        want[sub] = max(want[sub], lg["expect"]["records"])
    missing = sorted(s for s in want if not (legs / s).is_dir())
    count_off = sorted(
        (s, per_leg[s], n)
        for s, n in want.items()
        if (legs / s).is_dir() and per_leg[s] != n
    )

    out.mkdir(exist_ok=True)
    with open(out / "dedup.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subtree", "records", "superseded_in_subtree", "duplicates_of_earlier_subtrees",
                    "recipe_expect_records"])  # fmt: skip
        for s in sorted(per_leg):
            w.writerow([s, per_leg[s], superseded[s], dup[s], want.get(s, "")])
    summary = {
        "code_version": cv, "subtrees": len(per_leg), "records_read": sum(per_leg.values()),
        "unique_records": len(seen), "superseded_in_subtree": sum(superseded.values()), "duplicates": sum(dup.values()), "duplicates_not_byte_identical": len(bad),
        "superseded_by_prefer": collections.Counter(f"{a} over {b}" for _, a, b in by_prefer),
        "wrong_code_version": len(wrong_cv), "status": dict(status), "sidecar_clashes": len(clash),
        "recipe_paths_missing": missing, "recipe_record_count_off": count_off,
        "results_parquet_sha256": digest(parquet),
    }  # fmt: skip
    (out / "dedup.json").write_text(json.dumps(summary | {"not_identical": bad[:50], "wrong_cv": wrong_cv[:50],
                                                          "clashes": clash[:50]}, indent=1))  # fmt: skip
    print(json.dumps(summary, indent=1))
    if bad or wrong_cv or clash:
        raise SystemExit(
            "duplicates differ, a record has another code_version, or sidecars clash"
        )


if __name__ == "__main__":
    main()
