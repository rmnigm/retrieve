"""Report relative markdown links whose target does not exist."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FENCE = re.compile(r"^\s*```")

only = sys.argv[1:]
broken = 0
for md in sorted(ROOT.rglob("*.md")):
    if any(p in md.parts for p in (".git", ".venv", "node_modules")):
        continue
    rel = md.relative_to(ROOT).as_posix()
    if only and not any(rel.startswith(o) for o in only):
        continue
    in_fence = False
    for lineno, line in enumerate(md.read_text().splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
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
print(f"\n{broken} broken link(s)")
sys.exit(1 if broken else 0)
