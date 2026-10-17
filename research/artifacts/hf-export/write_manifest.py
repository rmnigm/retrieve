"""MANIFEST.json of the archive: every file but the manifest, with bytes and sha256, plus per-part totals."""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(sys.argv[1])


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "MANIFEST.json" or (p.is_file() and p.parent != ROOT and p.name == "MANIFEST.json"))
files = [p for p in files if ".cache" not in p.relative_to(ROOT).parts]
with ThreadPoolExecutor(64) as ex:
    shas = list(ex.map(sha, files))
entries = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": s} for p, s in zip(files, shas)]
parts = {}
for e in entries:
    top = e["path"].split("/")[0] if "/" in e["path"] else "(root)"
    parts.setdefault(top, {"files": 0, "bytes": 0})
    parts[top]["files"] += 1
    parts[top]["bytes"] += e["bytes"]
man = {"manifest_version": 1, "n_files": len(entries), "bytes": sum(e["bytes"] for e in entries), "parts": parts, "files": entries}
(ROOT / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=1))
