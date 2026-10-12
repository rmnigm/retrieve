"""wiki-split: rewrite every relative markdown link and every repo-path mention after the
`docs/` research pages moved to `research/` (git mv already done). Links are resolved
against the file's old location and re-relativised from its new one. Run from the repo root.

A link that already resolves from the file's new location is left alone, so a rerun changes
nothing and a branch merged after the split is fixed by `git mv docs/artifacts/<plan>
research/artifacts/<plan>` and a rerun. Paths as arguments restrict it to those files."""

import os
import re
import subprocess
import sys
from pathlib import Path

MOVED = {  # old -> new, files or directories
    **{f"docs/{p}": f"research/{p}" for p in (
        "SCHEMA.md", "index.md", "log.md", "roadmap.md", "backlog.md", "decisions.md",
        "validation.md", "contracts", "paper", "artifacts")},
    "docs/system/storage.md": "research/environment/storage.md",
}
# Incoming links to the old storage page keep pointing at docs/system/storage.md (a short
# project page is written there); the pod-specific ones are fixed by hand.
REDIRECT = {k: v for k, v in MOVED.items() if k != "docs/system/storage.md"}
LINK = re.compile(r"(\[[^\]]*\]\()([^)\s]+)(\))")
FENCE = re.compile(r"^\s*```")
# docs/index.md is left out: after the split it is the project's own index.
TEXT = re.compile(r"(?<![\w./-])docs/(SCHEMA\.md|log\.md|roadmap\.md|backlog\.md"
                  r"|decisions\.md|validation\.md|contracts|paper|artifacts)(?![\w-])")


def through(path: str, table: dict[str, str]) -> str:
    for old, new in table.items():
        if path == old or path.startswith(old + "/"):
            return new + path[len(old):]
    return path


def old_of(new: str) -> str:
    for old, n in MOVED.items():
        if new == n or new.startswith(n + "/"):
            return old + new[len(n):]
    return new


def fix_link(m: re.Match, old_dir: str, new_dir: str) -> str:
    target = m.group(2)
    if target.startswith(("http://", "https://", "mailto:", "#")) or "*" in target:
        return m.group(0)
    path, hash_, frag = target.partition("#")
    if not path or set(path) == {"."}:
        return m.group(0)
    if os.path.exists(os.path.join(new_dir, path)):  # already right for the new layout
        return m.group(0)
    resolved_old = os.path.normpath(os.path.join(old_dir, path))
    if resolved_old.startswith(".."):
        return m.group(0)
    want = through(resolved_old, REDIRECT)
    if os.path.normpath(os.path.join(new_dir, path)) == want:
        return m.group(0)
    rel = os.path.relpath(want, new_dir or ".")
    if path.endswith("/"):
        rel += "/"
    return m.group(1) + rel + hash_ + frag + m.group(3)


RECORDS = {  # historical inventories and generated output: the paths they name are as recorded
    "research/log.md",
    "research/artifacts/h1-results-storage/cleanup.tsv",
    "research/artifacts/results-draft/page/body.html",
}
files = subprocess.run(
    ["git", "ls-files", *sys.argv[1:]], capture_output=True, text=True, check=True
).stdout.split()
changed = 0
for new in files:
    p = Path(new)
    try:
        src = p.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        continue
    out = src
    if new.endswith(".md"):
        old = old_of(new)
        lines, fenced = [], False
        for line in out.split("\n"):
            if FENCE.match(line):
                fenced = not fenced
            elif not fenced:
                line = LINK.sub(lambda m: fix_link(m, os.path.dirname(old), os.path.dirname(new)), line)
            lines.append(line)
        out = "\n".join(lines)
    if new not in RECORDS and not new.startswith("research/artifacts/wiki-split/"):
        out = TEXT.sub(lambda m: "research/" + m.group(1), out)
    if out != src:
        p.write_text(out)
        changed += 1
print(f"{changed} files rewritten")
