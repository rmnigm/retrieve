"""FR-GATE step 0: every fetched `fr/*` tree against the Hub. For each local tree: download its `MANIFEST.json`
from the Hub, check its sha256 against `research/artifacts/hub-index.md`, check every file the manifest lists
against the local copy (size and sha256, `bench.upload.verify`), list local files the manifest does not name,
then hard-link the tree into OUT/<name> with the manifest beside it (the input of `bench merge`). Also reports
hub-index `fr/` rows with no local tree and local trees with no row. Exit 1 on any mismatch.

usage: verify_trees.py HUB_INDEX_MD LOCAL_ROOT OUT_ROOT [--repo pinkmeme/eval-results]
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from bench.upload import verify

args = sys.argv[1:]
repo = args[args.index("--repo") + 1] if "--repo" in args else "pinkmeme/eval-results"
hub_index, local_root, out_root = Path(args[0]), Path(args[1]), Path(args[2])
out_root.mkdir(parents=True, exist_ok=True)

rows = {}
for line in hub_index.read_text().splitlines():
    m = re.match(r"\| `(fr/[^`]+)` \|.*\| `([0-9a-f]{64})` \|\s*$", line)
    if m:
        rows[m.group(1)] = m.group(2)
local = sorted(p.name for p in local_root.iterdir() if p.is_dir())
problems = []
missing_local = sorted(set(rows) - {f"fr/{n}" for n in local})
missing_row = sorted(n for n in local if f"fr/{n}" not in rows)
problems += [f"hub-index row without a local tree: {s}" for s in missing_local]
problems += [f"local tree without a hub-index row: {n}" for n in missing_row]

summary = {}
cache = Path(os.environ.get("FR_GATE_HF_CACHE", "/scratch/hf/fr-gate-manifests"))
for name in local:
    sub = f"fr/{name}"
    if sub not in rows:
        continue
    got = Path(
        hf_hub_download(
            repo, f"{sub}/MANIFEST.json", repo_type="dataset", cache_dir=str(cache)
        )
    )
    man_bytes = got.read_bytes()
    sha = hashlib.sha256(man_bytes).hexdigest()
    man = json.loads(man_bytes)
    tree = local_root / name
    bad = verify(tree, man)
    listed = {f["path"] for f in man["files"]}
    extra = sorted(
        p.relative_to(tree).as_posix()
        for p in tree.rglob("*")
        if p.is_file()
        and p.relative_to(tree).as_posix() not in listed | {"MANIFEST.json"}
    )
    entry = {
        "hub_index_sha256_matches": sha == rows[sub],
        "n_files": man["n_files"],
        "n_records": man.get("n_records"),
        "file_mismatches": bad,
        "local_files_not_in_manifest": extra,
    }
    if sha != rows[sub]:
        problems.append(
            f"{sub}: MANIFEST.json sha256 {sha[:12]} != hub-index {rows[sub][:12]}"
        )
    problems += [f"{sub}: {b}" for b in bad]
    if extra:
        problems.append(f"{sub}: local files not in the manifest: {extra}")
    dest = out_root / name
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["cp", "-al", str(tree), str(dest)], check=True)
    (dest / "MANIFEST.json").write_bytes(man_bytes)
    summary[sub] = entry
    print(
        f"{sub:50s} sha {'ok' if entry['hub_index_sha256_matches'] else 'MISMATCH'}  files {man['n_files']:4d}  "
        f"mismatches {len(bad)}  extra {len(extra)}",
        flush=True,
    )

(out_root / "verify_trees.json").write_text(
    json.dumps({"problems": problems, "trees": summary}, indent=1)
)
print(f"{len(summary)} trees verified; {len(problems)} problem(s)")
for p in problems:
    print("PROBLEM", p)
sys.exit(1 if problems else 0)
