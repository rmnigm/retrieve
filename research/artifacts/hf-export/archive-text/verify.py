"""Check a copy of this archive (or any part of it) against MANIFEST.json: python3 verify.py [ROOT] [PREFIX ...]

ROOT defaults to this file's directory; each PREFIX (e.g. eval-pubmed/) restricts the check to the
entries under it, for a partial download. Standard library only."""

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
prefixes = tuple(sys.argv[2:])
manifest = json.loads((root / "MANIFEST.json").read_text())
bad = n = 0
for f in manifest["files"]:
    if prefixes and not f["path"].startswith(prefixes):
        continue
    n += 1
    p = root / f["path"]
    if not p.is_file():
        print(f"missing: {f['path']}")
        bad += 1
        continue
    if p.stat().st_size != f["bytes"]:
        print(f"size {p.stat().st_size} != {f['bytes']}: {f['path']}")
        bad += 1
        continue
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != f["sha256"]:
        print(f"sha256 differs: {f['path']}")
        bad += 1
print(f"{n} files checked, {bad} problems")
sys.exit(1 if bad else 0)
