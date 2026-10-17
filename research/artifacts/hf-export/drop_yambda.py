"""Drop Yambda from an archive tree (user decision 2026-10-17: four datasets): delete eval-yambda-500m/ and
eval-results/runs/yambda-s0/ (moved under /data/work/dropped-yambda/<tree>/), rewrite README.md and LICENSES.md, and drop their entries from MANIFEST.json (the
two texts re-hashed, the part totals recounted). python -I drop_yambda.py ROOT [ROOT ...]"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

GONE = ("eval-yambda-500m/", "eval-results/runs/yambda-s0/")
KEEP = Path("/data/work/dropped-yambda")  # moved aside, not deleted
README = [
    ("synthetic filters, the sequential encoder checkpoints)", "synthetic filters, the Goodreads sequential encoder checkpoint)"),
    ("| `eval-yambda-500m/` | `<owner>/eval-yambda-500m` | `data/yambda-500m` | `yambda-500m` | 10.23 GB |\n", ""),
    ("| `eval-results/` | `<owner>/eval-results` | (results trees) | | 0.61 GB |", "| `eval-results/` | `<owner>/eval-results` | (results trees) | | 0.60 GB |"),
    ("Checkpoints (under `<dataset>/checkpoints/<ckpt-id>/`): `eval-goodreads-work-id/checkpoints/sasrec-ssm-logq-d128`\n"
     "(the run read Goodreads at d128 only) and `eval-yambda-500m/checkpoints/sasrec-ssm-logq-d{64,128,256}`. arXiv,\n"
     "PubMed and LAION are pre-encoded and need no checkpoint.",
     "The checkpoint: `eval-goodreads-work-id/checkpoints/sasrec-ssm-logq-d128` (the run read Goodreads at d128 only).\n"
     "arXiv, PubMed and LAION are pre-encoded and need no checkpoint."),
    ("uv run --directory evaluation eval-data fetch yambda-500m --include-checkpoints\n", ""),
    ("The query encode cache of the sequential datasets is rebuilt", "Goodreads' query encode cache is rebuilt"),
    ("106 runs, one directory", "105 runs, one directory"),
    ("3,827 records and 9,543 sample lines", "3,752 records and 9,543 sample lines"),
    ("| `bloomwidth-s0`, `yambda-s0` | the Bloom width grid (quality only); Yambda, unfiltered |", "| `bloomwidth-s0` | the Bloom width grid (quality only) |"),
]
LICENSES = [
    ("## Yambda (`eval-yambda-500m/`)\n\n"
     "- **Source:** `yandex/yambda` on the Hugging Face Hub (Apache-2.0), the 500M `listens` table, Listen+ events.\n"
     "- **Shipped (derived):** the test split (dense track ids, anonymised-epoch timestamps, targets), the track-id map,\n"
     "  the three sequential encoders trained on the derived training split.\n"
     "- **Licence of the shipped part:** Apache-2.0, as the source.\n\n", ""),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rewrite(p: Path, edits) -> None:
    assert p.stat().st_nlink == 1, p
    t = p.read_text()
    for old, new in edits:
        assert t.count(old) == 1, (p, old[:60])
        t = t.replace(old, new)
    assert "yambda" not in t.lower(), p
    p.write_text(t)


for root in map(Path, sys.argv[1:]):
    rewrite(root / "README.md", README)
    rewrite(root / "LICENSES.md", LICENSES)
    for g in GONE:
        aside = KEEP / root.name / g
        aside.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(root / g, aside)
    man = json.loads((root / "MANIFEST.json").read_text())
    entries = [e for e in man["files"] if not e["path"].startswith(GONE)]
    for e in entries:
        if e["path"] in ("README.md", "LICENSES.md"):
            e["bytes"], e["sha256"] = (root / e["path"]).stat().st_size, sha(root / e["path"])
    parts = {}
    for e in entries:
        top = e["path"].split("/")[0] if "/" in e["path"] else "(root)"
        parts.setdefault(top, {"files": 0, "bytes": 0})
        parts[top]["files"] += 1
        parts[top]["bytes"] += e["bytes"]
    man = {"manifest_version": 1, "n_files": len(entries), "bytes": sum(e["bytes"] for e in entries), "parts": parts, "files": entries}
    (root / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    print(root, json.dumps({k: v for k, v in man.items() if k != "files"}))
