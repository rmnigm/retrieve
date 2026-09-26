"""One reader per ``RETRIEVE_*`` environment variable: every ``os.environ.get``,
``os.getenv`` and ``os.environ[...]`` of such a name under ``evaluation/`` sits in the
module that owns it; everyone else calls the owner."""

from __future__ import annotations

import ast
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1]
OWNERS = {"RETRIEVE_DATA_ROOT": "eval_datasets/hub.py"}  # read through hub.data_root()
MIN_FILES = 40  # 57 today; fewer means the walk is broken


def _reads(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            f, arg = node.func, node.args[0] if node.args else None
            env_get = isinstance(f, ast.Attribute) and ast.unparse(f) in {
                "os.environ.get",
                "os.getenv",
            }
            key = arg if env_get else None
        elif isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ":
            key = node.slice
        else:
            continue
        if isinstance(key, ast.Constant) and str(key.value).startswith("RETRIEVE_"):
            yield key.value, node.lineno


def test_each_retrieve_env_var_is_read_only_by_its_owner():
    files = sorted(p for p in EVAL.rglob("*.py") if ".venv" not in p.parts)
    assert len(files) > MIN_FILES, f"scanned {len(files)} files: the walk is broken"
    reads = [(p.relative_to(EVAL).as_posix(), *r) for p in files for r in _reads(p)]
    bad = [
        f"{rel}:{line} reads {name}; call {OWNERS.get(name, '<no owner: add one to OWNERS>')}"
        for rel, name, line in reads
        if OWNERS.get(name) != rel
    ]
    assert not bad, "\n".join(bad)
    stale = sorted(set(OWNERS) - {name for rel, name, _ in reads if OWNERS[name] == rel})
    assert not stale, f"no owner reads {stale}; drop the stale entry"
