"""The identifier scan over the whole archive. Text files: every byte. Binary files: everything that
can carry text: zip member names and pickles (.pt, .npz), parquet schema, key-value metadata and
every string value. Prints each hit; exit 1 if any."""

import re
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(sys.argv[1])
CI = re.compile(
    r"rmnigm|pinkmeme|nigmatullin|anthropic|claude|github\.com/|hf\.co/|huggingface\.co/"
    r"|[a-z0-9._%+-]+@[a-z][a-z0-9-]*(?:\.[a-z0-9-]+)*\.[a-z]{2,}"
    r"|\bpod[ -]?(?:[1-9]|[a-h])\b|\bpod[1-9a-z]\b|runpod|a100-x\d"
    r"|/scratch/|/venvs/|/workspace/|/root/|/home/|/users/"
    r"|full-rerun|synth-mid|v-ax-corr|\be1c\b|campaign-v2|meta-fork|roadmap|\.chains|\bthesis\b|wandb\.ai"
    r"|\bfr-[a-z]",
    re.I,
)
CS = re.compile(r"\bRoman\b")
TEXT = {".json", ".jsonl", ".md", ".csv", ".yaml", ".yml", ".txt", ".py"}
hits = 0
# data, not identifiers: a Goodreads book-format string (German / French for "novel") in the format code table
ALLOW = {("eval-goodreads-work-id/format_vocab.json", "Roman")}


def scan(label: str, s: str) -> None:
    global hits
    for rx in (CI, CS):
        for m in rx.finditer(s):
            if any(label.endswith(f) and m.group(0) == w for f, w in ALLOW):
                print(f"ALLOWED {label}: {m.group(0)!r}")
                continue
            hits += 1
            a, b = max(0, m.start() - 40), m.end() + 40
            print(f"HIT {label}: …{s[a:b]!r}…")


def scan_zip(p: Path) -> None:
    with zipfile.ZipFile(p) as z:
        for info in z.infolist():
            scan(f"{p}:{info.filename} (name)", info.filename)
            if info.filename.endswith(".pkl") or info.filename.endswith("version") or "/" not in info.filename:
                data = z.read(info.filename)
                if info.filename.endswith(".npy"):
                    data = data[:256]  # the npy header; the body is numeric
                scan(f"{p}:{info.filename}", data.decode("latin-1"))


def scan_parquet(p: Path) -> None:
    f = pq.ParquetFile(p)
    scan(f"{p} (schema)", str(f.schema_arrow))
    for k, v in (f.schema_arrow.metadata or {}).items():
        scan(f"{p} (metadata)", k.decode("latin-1") + "=" + v.decode("latin-1"))
    t = f.read()
    for name in t.column_names:
        col = t[name]
        def strings(arr):
            if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
                return [x for x in arr.unique().to_pylist() if x is not None]
            if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
                return strings(pa.chunked_array([c.flatten() for c in arr.chunks]) if isinstance(arr, pa.ChunkedArray) else arr.flatten())
            if pa.types.is_struct(arr.type):
                out = []
                for i in range(arr.type.num_fields):
                    out += strings(pa.chunked_array([c.field(i) for c in arr.chunks]) if isinstance(arr, pa.ChunkedArray) else arr.field(i))
                return out
            return []
        for s in strings(col):
            scan(f"{p}:{name}", s)


n = 0
for p in sorted(ROOT.rglob("*")):
    if not p.is_file():
        continue
    n += 1
    scan(f"{p} (path)", str(p.relative_to(ROOT)))
    if p.suffix in TEXT:
        scan(str(p), p.read_text(errors="replace"))
    elif p.suffix in (".pt", ".npz"):
        scan_zip(p)
    elif p.suffix == ".parquet":
        scan_parquet(p)
    else:
        print(f"UNSCANNED {p}")
print(f"{n} files scanned, {hits} hits")
sys.exit(1 if hits else 0)
