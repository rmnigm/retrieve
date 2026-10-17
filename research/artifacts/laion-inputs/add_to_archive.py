"""Add the rebuilt LAION inputs to the anonymous archive: ARCHIVE/eval-laion30m/ in the layout of the harness's
`<owner>/eval-laion30m` repo (what `eval-data publish laion30m` would upload: the staged dir minus its ignore
patterns), hardlinked from DATA/laion30m. The two embedding sidecars are rewritten: their `sources` named this
machine's paths, now the ETL's relative ones, plus the encoder and its pinned revision. The 1 / 3 / 10 M subsets
are rebuilt from it (`eval-data subset`), so only their MANIFEST.sha256 ship, under subsets/.

usage: add_to_archive.py DATA ARCHIVE"""

import json
import os
import shutil
import sys
from pathlib import Path

from eval_datasets.hub import EVAL_IGNORE_PATTERNS, _list_files_filtered

data, archive = Path(sys.argv[1]), Path(sys.argv[2])
src, dst = data / "laion30m", archive / "eval-laion30m"
ENCODE = json.loads((data / "_raw/relaion/laion30m/encode_params.json").read_text())
REVISION = {"nomic-ai/nomic-embed-text-v1.5": "e9b6763023c676ca8431644204f50c2b100d9aab",
            "nomic-ai/nomic-bert-2048 (remote code)": "7710840340a098cfb869c4f65e87cf2b1b70caca"}  # fmt: skip

if dst.exists():
    shutil.rmtree(dst)
for p in _list_files_filtered(src, EVAL_IGNORE_PATTERNS):
    out = dst / p.relative_to(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    if p.name.endswith(".meta.json"):
        meta = json.loads(p.read_text())
        meta["sources"] = [
            f"_raw/relaion/laion30m/{Path(s).name}" for s in meta["sources"]
        ]
        meta.update(encode=ENCODE, model_revisions=REVISION)
        out.write_text(json.dumps(meta, indent=2) + "\n")
    else:
        os.link(p, out)
for name in ("laion1m", "laion3m", "laion10m"):
    (dst / "subsets").mkdir(exist_ok=True)
    shutil.copy(
        data / name / "MANIFEST.sha256", dst / "subsets" / f"{name}.MANIFEST.sha256"
    )
print(sum(1 for _ in dst.rglob("*") if _.is_file()), "files,",
      round(sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1e9, 2), "GB")  # fmt: skip
