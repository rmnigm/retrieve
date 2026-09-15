"""The one dependency direction (plan V §4, X §2): ``bench`` → ``training`` → ``eval_datasets``,
nothing backwards, and the library reached only through the allow-list below. Every
``import`` / ``from`` statement of the three packages is resolved with ``ast``; a new edge is
a deliberate edit of this file."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[1]
PACKAGES = ("bench", "training", "eval_datasets")
ALLOWED = {
    "bench": {"bench", "training.encode", "eval_datasets.layout", "eval_datasets.hub"},
    "training": {
        "training",
        "eval_datasets.hub",
        "eval_datasets.layout",
        "eval_datasets.timesplit",
    },
    "eval_datasets": {"eval_datasets"},
}
LIBRARY = {
    "bench": {"retrieve", "retrieve.functional", "retrieve.interfaces"},
    "training": set(),
    "eval_datasets": {"retrieve.functional"},
}
LIBRARY_NAMES = {
    "retrieve": {
        "LinrBackend", "SilverTorchBackend", "FilterModule", "RetrievalModule", "SilverTorch",
        "SilverTorchBuilder", "OfficialConfig", "LiNRV1", "LiNRV2", "LiNRV3", "LiNRV4",
        "LiNRBuilder", "BloomFilter", "ExactAttributeFilter",
    },
    "retrieve.interfaces": {"DISPATCH", "FilterModule", "LinrBackend", "SilverTorchBackend"},
}  # fmt: skip


def _imports(path: Path):
    """``(module, names)`` per import statement; relative imports resolved to absolute."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, set()
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = ".".join(path.relative_to(EVAL).with_suffix("").parts[: -node.level])
                base = f"{parent}.{base}" if base else parent
            if base in PACKAGES:  # ``from eval_datasets import layout`` names submodules
                for a in node.names:
                    yield f"{base}.{a.name}", set()
            else:
                yield base, {a.name for a in node.names}


def _owner(module: str) -> str | None:
    top = module.split(".")[0]
    return top if top in PACKAGES or top == "retrieve" else None


@pytest.mark.parametrize("package", PACKAGES)
def test_only_allowed_edges(package):
    bad = []
    for path in sorted((EVAL / package).rglob("*.py")):
        for module, names in _imports(path):
            owner = _owner(module)
            if owner is None:
                continue
            if owner == "retrieve":
                ok = module in LIBRARY[package] and (
                    module not in LIBRARY_NAMES or names <= LIBRARY_NAMES[module]
                )
            else:
                ok = owner == package or any(
                    module == a or module.startswith(a + ".") for a in ALLOWED[package] - {package}
                )
            if not ok:
                bad.append(f"{path.relative_to(EVAL)}: {module} {sorted(names)}")
    assert not bad, "\n".join(bad)
