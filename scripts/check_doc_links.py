"""Report relative markdown links whose target does not exist; in the wiki (AGENTS.md,
README.md, docs/ minus paper/ and artifacts/) also backticked repo paths that do not exist
and `bench` / `eval-data` / `train` subcommands the click groups do not define. The command
table is read from the CLI sources with ast: nothing is imported or run."""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FENCE = re.compile(r"^\s*```")
CODE = re.compile(r"`([^`]+)`")
REPO_PATH = re.compile(r"(?:retrieve|evaluation|scripts|docs)/[\w./-]*")
LINE_SUFFIX = re.compile(r"(#.*|:\d+(-\d+)?)$")
CLI_CALL = re.compile(r"(?<![\w./-])(bench|eval-data|train) ([a-z][\w-]*)")
CLIS = {
    "bench": "evaluation/bench",
    "eval-data": "evaluation/eval_datasets",
    "train": "evaluation/training",
}
RUNTIME = {"evaluation/data"}  # gitignored: the data-root symlink each checkout makes
MIN_PATHS, MIN_CLI = 60, 50  # 100 and 85 today; fewer means a pattern broke


def _click_name(fn: ast.FunctionDef) -> tuple[str | None, str | None] | None:
    """``(group, name)`` of a click-decorated function: ``group`` is None for
    ``click.command``, ``name`` None when it is a variable (the ETL loop below)."""
    for d in fn.decorator_list:
        if (
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr == "command"
        ):
            owner = d.func.value.id if isinstance(d.func.value, ast.Name) else None
            arg = (
                d.args[0]
                if d.args
                else next((k.value for k in d.keywords if k.arg == "name"), None)
            )
            name = fn.name.replace("_", "-") if arg is None else getattr(arg, "value", None)
            return (None if owner == "click" else owner), name
    return None


def commands(package: Path) -> set[str]:
    """The subcommands of ``package/cli.py``'s ``main`` group."""
    defined = {}
    for py in package.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.FunctionDef) and (found := _click_name(node)):
                defined[node.name] = found[1]
    tree = ast.parse((package / "cli.py").read_text())
    dicts = {
        t.id: [k.value for k in n.value.keys]
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
        for t in n.targets
    }
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (found := _click_name(node)):
            if found[0] == "main" and found[1]:
                out.add(found[1])
        elif isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            base = node.iter.func.value  # ``for command, module in ETL.items()``
            out.update(dicts.get(base.id, []) if isinstance(base, ast.Name) else [])
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_command"
        ):
            named = node.args[1].value if len(node.args) > 1 else None
            out.add(named or defined[node.args[0].id])
    return out


def in_wiki(rel: str) -> bool:
    if rel == "docs/log.md":  # append-only history: it names what was removed
        return False
    if rel in ("AGENTS.md", "README.md"):
        return True
    return rel.startswith("docs/") and not rel.startswith(("docs/paper/", "docs/artifacts/"))


only = sys.argv[1:]
table = {prog: commands(ROOT / pkg) for prog, pkg in CLIS.items()}
broken = n_paths = n_cli = 0
runtime_seen = set()
for md in sorted(ROOT.rglob("*.md")):
    if any(p in md.parts for p in (".git", ".venv", "node_modules")):
        continue
    rel = md.relative_to(ROOT).as_posix()
    if only and not any(rel.startswith(o) for o in only):
        continue
    wiki = in_wiki(rel)
    in_fence = False
    for lineno, line in enumerate(md.read_text().splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        spans = [line] if in_fence else CODE.findall(line)
        if wiki:
            for span in spans:
                for prog, sub in CLI_CALL.findall(span):
                    n_cli += 1
                    if sub not in table[prog]:
                        print(
                            f"UNKNOWN {rel}:{lineno} -> {prog} {sub} (have {sorted(table[prog])})"
                        )
                        broken += 1
            for span in [] if in_fence else spans:
                token = LINE_SUFFIX.sub("", span)
                if "<" in token or "*" in token or not REPO_PATH.fullmatch(token):
                    continue
                n_paths += 1
                runtime = {r for r in RUNTIME if token == r or token.startswith(r + "/")}
                runtime_seen |= runtime
                if not runtime and not (ROOT / token).exists():
                    print(f"MISSING {rel}:{lineno} -> {span}")
                    broken += 1
        if in_fence:
            continue
        for target in LINK.findall(line):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            # `**kwargs` / `...` inside inline code read as links to the naive
            # regex; a real relative target never contains these.
            if "*" in target or set(target) == {"."}:
                continue
            path = target.split("#")[0]
            if not path:
                continue
            if not (md.parent / path).exists():
                print(f"BROKEN {rel}:{lineno} -> {target}")
                broken += 1
if not only and RUNTIME - runtime_seen:
    print(f"no doc names {sorted(RUNTIME - runtime_seen)}; drop the stale RUNTIME entry")
    broken += 1
if not only and (n_paths < MIN_PATHS or n_cli < MIN_CLI):
    print(
        f"found {n_paths} repo paths / {n_cli} CLI calls (< {MIN_PATHS} / {MIN_CLI}): "
        "the pattern is broken"
    )
    broken += 1
print(f"\n{broken} problem(s); {n_paths} repo paths and {n_cli} CLI calls checked")
sys.exit(1 if broken else 0)
