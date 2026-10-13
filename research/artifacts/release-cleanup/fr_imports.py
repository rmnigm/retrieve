"""The repository files a FULL-RERUN leg imports at this tree: the harness entry points FR runs (bench.cli, eval_datasets.cli
with its synth-filter / subset commands), every op namespace and both official backends, and the module-level imports of the FR
scripts (full-rerun/, results-draft/, meta-fork/gate_fork.py), one repo-relative path per line. The release cleanup's gate:
every listed file is byte-identical to final-freeze.

usage: fr_imports.py REPO > fr-imports.txt   (run from REPO/evaluation in the tree's venv)
"""

import ast
import importlib
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
scripts = [
    *sorted((repo / "research/artifacts/full-rerun").glob("*.py")),
    *sorted((repo / "research/artifacts/results-draft").glob("*.py")),
    repo / "research/artifacts/meta-fork/gate_fork.py",
]
mods = [
    "bench.cli",
    "bench.run",
    "bench.oracle",
    "bench.report",
    "bench.upload",
    "bench.measure",
    "eval_datasets.cli",
]
for s in scripts:
    sys.path.insert(0, str(s.parent))
    for node in ast.walk(ast.parse(s.read_text())):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.append(node.module)
for m in dict.fromkeys(mods):
    try:
        importlib.import_module(m)
    except ImportError as exc:
        print(f"# skip {m}: {exc}", file=sys.stderr)
from retrieve.interfaces import ops_for  # noqa: E402
from retrieve.ops import official  # noqa: E402

for b in ("triton", "torch", "official", "official-fork"):
    ops_for(b)
for b in ("official", "official-fork"):
    official.ensure_loaded(b)
files = set(map(str, scripts))
for m in list(sys.modules.values()):
    f = getattr(m, "__file__", None)
    if f:
        files.add(str(Path(f).resolve()))
print(
    "\n".join(
        sorted(
            str(Path(f).relative_to(repo))
            for f in files
            if Path(f).is_relative_to(repo)
        )
    )
)
