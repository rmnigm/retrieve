"""Copy the final run's per-run results trees (Hub fr/<leg>) into the archive under neutral names,
scrubbed, with results.parquet and MANIFEST.json rebuilt by the export's own bench code.

Usage: scrub_runs.py SRC_FR OUT_RUNS MAP_CSV
"""

import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/wt/export-inputs/evaluation")
import yaml  # noqa: E402

from bench import records, upload  # noqa: E402

SRC, OUT, MAP = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

POD = r"(?:pod1|pode|podf)"
PART = {"pod1": "a", "pode": "b", "podf": "c"}  # one letter per partition of the datasets
PATH_PREFIXES = [
    ("/scratch/wt/fr-clause/", ""),
    ("/scratch/wt/fr/", ""),
    ("/venvs/fr-clause/", "<venv>/"),
    ("/venvs/fr/", "<venv>/"),
]
DROP_DIRS = {"logs"}
DROP_NAMES = {"MANIFEST.json", "results.parquet", "marker.json", "commit.txt", ".done", ".repaired"}
DROP_SUFFIXES = (".sh", ".log")
LEG = re.compile(rf"(?P<base>.+?)(?:-(?P<pod>{POD}))?-s(?P<seed>\d)(?:-on{POD})?(?P<rerun>-rerun)?$")
SKIP_LEGS = {"marks"}  # progress markers of the run driver, no outputs


def suite(name: str) -> str:
    s = re.sub(r"^fr-", "", name)
    s = re.sub(rf"-{POD}(?=-|$)", "", s)
    s = re.sub(r"^ext-", "followup-", s)
    if s == "synth-mid-probesweep-d256":
        return "probe-sweep-d256"  # the name in results/final-run-config/suites.yaml
    return re.sub(r"^synth-mid-", "synth-broad-", s)


def leg_name(leg: str, multi_part: set[str]) -> str:
    if leg == "picks":
        return "timed-point-picks"
    m = LEG.fullmatch(leg)
    assert m, leg
    base = re.sub(r"^ext-", "followup-", m["base"])
    base = "probe-sweep" if m["base"] == "synth-mid-probesweep" else re.sub(r"^synth-mid-", "synth-broad-", base)
    part = f"-{PART[m['pod']]}" if m["base"] in multi_part else ""
    return f"{base}{part}-s{m['seed']}{m['rerun'] or ''}"


def text(s: str) -> str:
    for a, b in PATH_PREFIXES:
        s = s.replace(a, b)
    return s


hosts: dict[str, str] = {}


def host(h: str) -> str:
    return hosts.setdefault(h, f"gpu-host-{len(hosts) + 1}")


def scrub(o, key: str = ""):
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            out[k] = scrub(v, k)
        return out
    if isinstance(o, list):
        return [scrub(v, key) for v in o]
    if isinstance(o, str):
        if key == "suite":
            return suite(o)
        if key == "per_query":
            head, _, rest = o.partition("/")
            return f"{suite(head)}/{rest}"
        if key == "host":
            return host(o)
        return text(o)
    return o


def rel_out(rel: Path) -> Path:
    parts = list(rel.parts)
    if len(parts) > 1:
        parts[0] = suite(parts[0])
    parts[-1] = re.sub(rf"-{POD}(?=-|\.)", "", parts[-1])
    return Path(*parts)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    name = src.name
    if name.endswith(".jsonl"):
        with src.open() as fi, dst.open("w") as fo:
            for line in fi:
                rec = scrub(json.loads(line))
                fo.write(json.dumps(rec, separators=(",", ":"), allow_nan=False) + "\n")
    elif name.endswith(".json"):
        dst.write_text(json.dumps(scrub(json.loads(src.read_text())), indent=1) + "\n")
    elif name.endswith(".yaml"):
        doc = yaml.safe_load(src.read_text())
        doc = {suite(k): scrub(v) for k, v in doc.items()}
        dst.write_text(yaml.safe_dump(doc, sort_keys=False))
    elif name.endswith(".csv"):
        dst.write_text(text(src.read_text()))
    else:
        os.link(src, dst)


def main() -> None:
    legs = sorted(d.name for d in SRC.iterdir() if d.is_dir() and d.name not in SKIP_LEGS)
    pods_per_base: dict[str, set[str]] = {}
    for leg in legs:
        m = LEG.fullmatch(leg)
        if m and m["pod"]:
            pods_per_base.setdefault(m["base"], set()).add(m["pod"])
    multi = {b for b, p in pods_per_base.items() if len(p) > 1}
    names = {leg: leg_name(leg, multi) for leg in legs}
    assert len(set(names.values())) == len(names), "name collision"
    if OUT.exists():
        shutil.rmtree(OUT)
    rows = []
    for leg in legs:
        src, dst = SRC / leg, OUT / names[leg]
        for p in sorted(src.rglob("*")):
            rel = p.relative_to(src)
            if not p.is_file() or rel.parts[0] in DROP_DIRS or p.name in DROP_NAMES:
                continue
            if p.name.endswith(DROP_SUFFIXES) or p.name.startswith("FAILED"):
                continue
            copy_file(p, dst / rel_out(rel))
        if records.record_files(dst):
            records.aggregate(dst)
        prefix = f"runs/{names[leg]}"
        man = upload.manifest(dst, upload.files(dst), prefix, None)
        man["source"] = prefix
        (dst / "MANIFEST.json").write_text(json.dumps(man, indent=2))
        rows.append({"archive_run": names[leg], "hub_subtree": f"fr/{leg}", "n_records": man["n_records"],
                     "n_files": man["n_files"], "bytes": man["bytes"]})
        print(f"{leg} -> {names[leg]}: {man['n_records']} records, {man['n_files']} files", flush=True)
    MAP.parent.mkdir(parents=True, exist_ok=True)
    with MAP.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (MAP.parent / "host_map.json").write_text(json.dumps(hosts, indent=1) + "\n")
    print(f"{len(rows)} runs; hosts {len(hosts)}")


main()
