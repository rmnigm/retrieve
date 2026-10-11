#!/usr/bin/env bash
# Build the fork at one commit for its gate, on CPU (never while pod b times a final-pass leg).
# The build tree /scratch/wt/meta-fork-build is checked out (detached) at <commit>; the venv
# /venvs/meta-fork is a copy of the final pass's /venvs/final-ext-o3 (Meta's silvertorch built -O3,
# the gates' upstream arm; never re-synced), and silvertorch-fork is installed editable from the
# build tree, so a rebuild is ninja's incremental one. Records the .so's sha256 beside the log.
#   build.sh <commit> <log_dir>
set -eu
COMMIT=$1 LOGS=$2
REPO=/workspace/retrieve TREE=/scratch/wt/meta-fork-build VENV=/venvs/meta-fork
mkdir -p "$LOGS"
[ -d "$VENV" ] || cp -a /venvs/final-ext-o3 "$VENV"
[ -d "$TREE" ] || git -C "$REPO" worktree add --detach "$TREE" "$COMMIT"
git -C "$TREE" checkout -q --detach "$COMMIT"
export TORCH_CUDA_ARCH_LIST=8.0 CUDA_HOME=/usr/local/cuda MAX_JOBS="${MAX_JOBS:-16}"
nice -n 10 uv pip install --python "$VENV/bin/python" --no-build-isolation --no-cache --no-deps \
  -e "$TREE/official-fork" > "$LOGS/build-$COMMIT.log" 2>&1
"$VENV/bin/python" - "$LOGS/build-$COMMIT.json" "$COMMIT" <<'PY'
import hashlib, importlib.util, json, pathlib, sys
pkg = pathlib.Path(next(iter(importlib.util.find_spec("silvertorch_fork").submodule_search_locations)))
so = sorted(pkg.glob("_C*.so"))[0]
up = pathlib.Path(next(iter(importlib.util.find_spec("silvertorch").submodule_search_locations)))
up_so = sorted(up.glob("_C*.so"))[0]
rec = {"commit": sys.argv[2], "fork_so": str(so), "fork_sha256": hashlib.sha256(so.read_bytes()).hexdigest(),
       "upstream_so": str(up_so), "upstream_sha256": hashlib.sha256(up_so.read_bytes()).hexdigest()}
pathlib.Path(sys.argv[1]).write_text(json.dumps(rec, indent=1))
print(rec)
PY
