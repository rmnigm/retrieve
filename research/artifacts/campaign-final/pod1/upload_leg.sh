#!/usr/bin/env bash
# The corrected final-pass upload (brief, CORRECTION 15:20Z): hardlink this leg's records + sidecars into $T/<suite>/, upload
# --results $T to campaign-final/<leg> --verify, then check the Hub MANIFEST: n_records = the leg's record count, code_versions
# = [LIBRARY], results.parquet listed. Usage (from evaluation/): upload_leg.sh LEG UPLOAD_PATH SUITE DATASET DIM VENV OUT LIBRARY
set -eu
LEG=$1 UP=$2 SUITE=$3 DS=$4 DIM=$5 VENV=$6 OUT=$7 LIB=$8
T=$(mktemp -d /scratch/final/upload.XXXXXX); mkdir -p "$T/$SUITE"
cp -al "$OUT/$SUITE/$DS-d$DIM".* "$T/$SUITE/"
N=$(wc -l < "$T/$SUITE/$DS-d$DIM.jsonl")
"$VENV/bin/python" -m bench.cli upload --results "$T" --path-in-repo "$UP" --verify
"$VENV/bin/python" - "$UP" "$N" "$LIB" <<'PY'
import json, sys
from huggingface_hub import hf_hub_download
up, n, lib = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = json.load(open(hf_hub_download("pinkmeme/eval-results", f"{up}/MANIFEST.json", repo_type="dataset", force_download=True)))
files = {f["path"] for f in m["files"]}
ok = m["n_records"] == n and m["code_versions"] == [lib] and "results.parquet" in files
print(f"manifest check {'OK' if ok else 'FAILED'}: n_records {m['n_records']} (expect {n}), code_versions {m['code_versions']}, "
      f"commits {m['commits']}, status {m['status']}, unstable {m['unstable']}, results.parquet {'results.parquet' in files}")
sys.exit(0 if ok else 1)
PY
rm -rf "$T"
