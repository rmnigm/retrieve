"""The archive's scrubbed runs against the Hub originals: every record line present, equal on every
field the scrub does not touch; every npz byte-identical; every manifest sha256 true."""

import csv
import hashlib
import json
import sys
from pathlib import Path

SRC = Path("/data/work/hub/eval-results/fr")
OUT = Path("/data/anon-export/eval-results/runs")
MAP = Path("/data/work/private/run_name_map.csv")
TOUCHED = {"suite", "per_query", "backend", "path", "env", "error", "interleave"}


def lines(root: Path, samples: bool):
    out = []
    for p in sorted(root.rglob("*.jsonl")):
        if p.name.endswith(".samples.jsonl") != samples:
            continue
        out += [json.loads(x) for x in p.open()]
    return out


def strip(r):
    keep = {k: v for k, v in r.items() if k not in TOUCHED}
    if isinstance(r.get("interleave"), dict):
        keep["interleave"] = {k: v for k, v in r["interleave"].items() if k != "arms"}
    env = r.get("env") or {}
    keep["_env"] = {k: v for k, v in env.items() if k not in
                    ("host", "lib_dir", "official_build", "official_fork_build", "official_commit",
                     "published_build", "cuda_build", "published_commit")}
    return json.dumps(keep, sort_keys=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


bad = 0
n_rec = n_npz = n_files = 0
for row in csv.DictReader(MAP.open()):
    src, dst = SRC / row["hub_subtree"].removeprefix("fr/"), OUT / row["archive_run"]
    for samples in (False, True):
        a = sorted(strip(r) for r in lines(src, samples))
        b = sorted(strip(r) for r in lines(dst, samples))
        n_rec += len(a)
        if a != b:
            bad += 1
            print("DIFF", row["archive_run"], "samples" if samples else "records", len(a), len(b))
    a_npz = sorted(sha(p) for p in src.rglob("*.npz"))
    b_npz = sorted(sha(p) for p in dst.rglob("*.npz"))
    n_npz += len(a_npz)
    if a_npz != b_npz:
        bad += 1
        print("NPZ DIFF", row["archive_run"])
    man = json.loads((dst / "MANIFEST.json").read_text())
    for f in man["files"]:
        n_files += 1
        p = dst / f["path"]
        if not p.is_file() or p.stat().st_size != f["bytes"] or sha(p) != f["sha256"]:
            bad += 1
            print("MANIFEST", row["archive_run"], f["path"])
    for r in lines(dst, False):
        if r.get("per_query") and not (dst / r["per_query"]).is_file():
            bad += 1
            print("PER_QUERY MISSING", row["archive_run"], r["per_query"])
print(f"{n_rec} record + sample lines, {n_npz} npz, {n_files} manifest entries checked; {bad} problems")
sys.exit(1 if bad else 0)
