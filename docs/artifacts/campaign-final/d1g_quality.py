"""D1-G's byte-identical quality subset (README.md): per recipe grid leg, one cheap job of the final pass re-run quality
only (`--skip-perf --force`) into a scratch tree at the same library, and its `quality` compared with the final record's,
byte for byte (canonical JSON). Two modes:

  plan    MERGED_TREE RECIPE.yaml OUT.json   one job per leg: the (filter kind, sweep, seed) selection of the leg's
                                             arms whose recorded builds are cheapest, as `bench run` filters
  compare MERGED_TREE RERUN_TREE OUT.csv     every rerun record against the final record of the same key
"""

import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "evaluation"))
import yaml  # noqa: E402
from bench import records  # noqa: E402


def plan(tree: Path, recipe: Path, out: Path) -> None:
    recs = records.latest(tree)
    legs = [
        lg
        for lg in yaml.safe_load(recipe.read_text())["legs"]
        if lg["suite"] != "ceiling-script"
    ]
    picks = []
    for lg in legs:
        cmd = lg["command"]
        algos = [a for a in cmd.split("--algo ")[1:]] and [
            c.split()[0] for c in cmd.split("--algo ")[1:]
        ]
        backend = cmd.split("--backend ")[1].split()[0] if "--backend " in cmd else None
        mine = [r for r in recs if r["suite"] == lg["suite"] and r["dataset"] == lg["dataset"]
                and (not algos or r["algo"] in algos) and (backend is None or r["backend"] == backend)]  # fmt: skip
        if not mine:
            picks.append({"leg": lg["id"], "error": "no final records"})
            continue
        groups = collections.defaultdict(list)
        for r in mine:
            groups[(r["filter_kind"], r["sweep"], r["seed"])].append(r)
        builds = {
            g: sum(
                {
                    (r["algo"], r["backend"], r.get("build_s")): r.get("build_s") or 0
                    for r in rs
                }.values()
            )
            + len(rs)
            for g, rs in groups.items()
        }
        (fk, sw, seed) = min(builds, key=lambda g: (builds[g], g[2], str(g)))
        picks.append({
            "leg": lg["id"], "index": lg["index"], "suite": lg["suite"], "dataset": lg["dataset"],
            "dim": groups[(fk, sw, seed)][0]["dim"], "algos": algos or sorted({r["algo"] for r in mine}),
            "backend": backend, "filter_kind": fk, "sweep": sw, "seed": seed,
            "official": any(r["backend"] == "official" for r in groups[(fk, sw, seed)]),
            "records": len(groups[(fk, sw, seed)]), "build_s": round(builds[(fk, sw, seed)], 1),
        })  # fmt: skip
    out.write_text(json.dumps(picks, indent=1))
    print(f"{len(picks)} picks, {sum(p.get('records', 0) for p in picks)} records, "
          f"{sum(p.get('build_s', 0) for p in picks) / 3600:.2f} h of builds")  # fmt: skip


def canon(q) -> str:
    return json.dumps(q, sort_keys=True, separators=(",", ":"))


def compare(tree: Path, rerun: Path, out: Path) -> None:
    final = {records.record_key(r): r for r in records.latest(tree)}
    rows, bad = [], 0
    for r in records.latest(rerun):
        k = records.record_key(r)
        f = final.get(k)
        same = f is not None and canon(f["quality"]) == canon(r["quality"])
        bad += not same
        rows.append([r["suite"], r["dataset"], r["algo"], r["backend"], r["filter_kind"], r["sweep"],
                     json.dumps(r["params"], sort_keys=True), r["seed"], f is not None, same])  # fmt: skip
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["suite", "dataset", "algo", "backend", "filter_kind", "sweep", "params", "seed",
                    "final_record", "quality_byte_identical"])  # fmt: skip
        w.writerows(rows)
    print(
        f"{len(rows)} rerun records, {len(rows) - bad} byte-identical quality, {bad} not"
    )
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    mode = sys.argv[1]
    {"plan": plan, "compare": compare}[mode](*map(Path, sys.argv[2:5]))
