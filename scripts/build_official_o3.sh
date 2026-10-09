#!/usr/bin/env bash
# Build Meta's official extension (silvertorch at the pyproject pin) with host + nvcc -O3 into the venv
# UV_PROJECT_ENVIRONMENT names, kernels unmodified (OF-11: Meta's setup.py passes no -O, so its host code builds at
# gcc -O0), and write the flags beside the .so for the records' env.official_build. A plain `uv sync` would reuse
# uv's cached wheel (built without the flags), so silvertorch is rebuilt with --no-cache after the sync, and the
# flags file is written only if the log shows it was built. Run it on a leg's own venv, never on a shared venv under a
# running leg. Usage: UV_PROJECT_ENVIRONMENT=/venvs/<leg> scripts/build_official_o3.sh
set -eu
: "${UV_PROJECT_ENVIRONMENT:?set UV_PROJECT_ENVIRONMENT to the venv to build}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
FLAGS="-O3 -Xcompiler -O3"
cd "$(dirname "$0")/.."
uv sync --frozen --all-packages --all-groups --extra official
REV=$(sed -n 's/^silvertorch = { git = "\(.*\)", rev = "\(.*\)" }$/\1@\2/p' pyproject.toml)
LOG=$(mktemp)
NVCC_APPEND_FLAGS="$FLAGS" uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-build-isolation \
  --no-cache --no-deps --reinstall "silvertorch @ git+$REV" 2>&1 | tee "$LOG"
grep -q "Built silvertorch" "$LOG" || { echo "silvertorch was not rebuilt; no flags file written" >&2; exit 1; }
"$UV_PROJECT_ENVIRONMENT/bin/python" - "$FLAGS" <<'PY'
import importlib.util, json, pathlib, sys
pkg = pathlib.Path(next(iter(importlib.util.find_spec("silvertorch").submodule_search_locations)))
(pkg / "_build_flags.json").write_text(json.dumps({"nvcc_append_flags": sys.argv[1]}))
print("wrote", pkg / "_build_flags.json")
PY
