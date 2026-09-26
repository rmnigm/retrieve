"""The one dependency direction (plan V §4, X §2): ``bench`` → ``training`` → ``eval_datasets``,
nothing backwards, and the library reached only through the allow-list below. Every
``import`` / ``from`` statement of the three packages is resolved with ``ast``; a new edge is
a deliberate edit of this file, and an entry no import uses any more fails as stale."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[1]
PACKAGES = ("bench", "training", "eval_datasets")
MIN_FILES = 25  # 37 today; fewer means the walk is broken, not that the tree shrank
ALLOWED = {
    "bench": {"bench", "training.encode", "eval_datasets.layout"},
    "training": {"training", "eval_datasets.hub", "eval_datasets.layout"},
    "eval_datasets": {"eval_datasets"},
}
LIBRARY = {
    "bench": {"retrieve", "retrieve.interfaces"},
    "training": set(),
    "eval_datasets": set(),
}
LIBRARY_NAMES = {
    "retrieve": {
        "SilverTorch", "LiNRV1", "LiNRV2", "LiNRV3", "LiNRV4", "BloomFilter",
        "ExactAttributeFilter",
    },
    "retrieve.interfaces": {"DISPATCH", "FilterModule"},
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


def _edges() -> list[tuple[str, Path, str, set[str]]]:
    """``(package, file, module, names)`` for every import of a package or the library."""
    out = []
    for package in PACKAGES:
        for path in sorted((EVAL / package).rglob("*.py")):
            for module, names in _imports(path):
                if module.split(".")[0] in (*PACKAGES, "retrieve"):
                    out.append((package, path, module, names))
    return out


def _allowed(package: str, module: str, names: set[str]) -> bool:
    if module.split(".")[0] == "retrieve":
        return module in LIBRARY[package] and names <= LIBRARY_NAMES.get(module, names)
    return any(module == a or module.startswith(a + ".") for a in ALLOWED[package])


@pytest.mark.parametrize("package", PACKAGES)
def test_only_allowed_edges(package):
    bad = [
        f"{path.relative_to(EVAL)}: {module} {sorted(names)}"
        for pkg, path, module, names in _edges()
        if pkg == package and not _allowed(pkg, module, names)
    ]
    assert not bad, "\n".join(bad)


def test_the_walk_saw_the_tree_and_every_allow_list_entry_is_used():
    n_files = sum(len(list((EVAL / p).rglob("*.py"))) for p in PACKAGES)
    assert n_files > MIN_FILES, f"scanned {n_files} files: the walk is broken"
    edges = _edges()
    stale = [
        f"ALLOWED[{pkg!r}] {a!r}"
        for pkg, entries in ALLOWED.items()
        for a in entries - {pkg}
        if not any(p == pkg and (m == a or m.startswith(a + ".")) for p, _, m, _ in edges)
    ]
    stale += [
        f"LIBRARY[{pkg!r}] {mod!r}"
        for pkg, mods in LIBRARY.items()
        for mod in mods
        if not any(p == pkg and m == mod for p, _, m, _ in edges)
    ]
    used = {(m, n) for _, _, m, names in edges for n in names}
    stale += [
        f"LIBRARY_NAMES[{mod!r}] {name!r}"
        for mod, names in LIBRARY_NAMES.items()
        for name in names
        if (mod, name) not in used
    ]
    assert not stale, "no import uses these; drop the stale entry:\n" + "\n".join(stale)
