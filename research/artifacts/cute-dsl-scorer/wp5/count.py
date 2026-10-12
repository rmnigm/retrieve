"""Line accounting for the SilverTorch backends (WP-5 SIZE.md).

Python: tokenize for comments, ast for docstrings. A line is
  blank      — nothing but whitespace
  comment    — every token on it is COMMENT/NL/NEWLINE/INDENT/DEDENT, or every
               non-trivial token belongs to a docstring
  code       — everything else (a code line with a trailing comment is code)
CUDA: blank; comment = starts with // or lies inside a /* */ block; else code.
"""
import ast, io, re, sys, tokenize
from pathlib import Path

ROOT = Path("/workspace/retrieve/retrieve/src/retrieve/kernels")
K = ROOT / "silvertorch"

def py_classes(src: str):
    """Return per-line class: 'blank' | 'comment' | 'code' (1-based list)."""
    lines = src.splitlines()
    n = len(lines)
    cls = ["code"] * (n + 1)
    doc_lines = set()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                d = node.body[0]
                doc_lines.update(range(d.lineno, d.end_lineno + 1))
    trivial = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}
    has_code = [False] * (n + 2)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in trivial:
            continue
        if tok.type == tokenize.STRING and tok.start[0] in doc_lines and tok.end[0] in doc_lines:
            continue
        for ln in range(tok.start[0], tok.end[0] + 1):
            has_code[ln] = True
    for i, line in enumerate(lines, 1):
        if not line.strip():
            cls[i] = "blank"
        elif not has_code[i]:
            cls[i] = "comment"
    return cls[1:]

def cu_classes(src: str):
    out = []
    in_block = False
    for line in src.splitlines():
        s = line.strip()
        if not s:
            out.append("blank"); continue
        if in_block:
            out.append("comment")
            if "*/" in s:
                in_block = False
            continue
        if s.startswith("//"):
            out.append("comment"); continue
        if s.startswith("/*"):
            if "*/" in s and s.rstrip().endswith("*/"):
                out.append("comment")
            else:
                out.append("comment"); in_block = True
            continue
        out.append("code")
    return out

def tally(cls, lo=None, hi=None):
    seg = cls[(lo or 1) - 1 : (hi or len(cls))]
    return dict(total=len(seg), blank=seg.count("blank"), comment=seg.count("comment"), code=seg.count("code"))

def fmt(name, t):
    return f"| {name} | {t['total']} | {t['blank']} | {t['comment']} | {t['code']} |"

files = {
    "cuda/codesigned_probe_score.cu (C++ device + launchers + pybind)": (K / "cuda/codesigned_probe_score.cu", cu_classes),
    "codesigned_probe_score_cuda.py (C++ host)": (K / "codesigned_probe_score_cuda.py", py_classes),
    "cute/codesigned_probe_score.py (DSL device + launchers + compile cache)": (K / "cute/codesigned_probe_score.py", py_classes),
    "codesigned_probe_score_cute.py (DSL host)": (K / "codesigned_probe_score_cute.py", py_classes),
    "codesigned_probe_score.py (Triton, none/bloom)": (K / "codesigned_probe_score.py", py_classes),
    "codesigned_probe_score_exact.py (Triton, exact)": (K / "codesigned_probe_score_exact.py", py_classes),
}
CLS = {}
rows = []
for name, (path, fn) in files.items():
    src = path.read_text()
    CLS[name] = fn(src)
    rows.append((name, tally(CLS[name])))

# Triton shared helpers actually used by the two kernels.
common_src = (ROOT / "common.py").read_text()
common_cls = py_classes(common_src)
tree = ast.parse(common_src)
helpers = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        lo = min([d.lineno for d in node.decorator_list] + [node.lineno])
        helpers[node.name] = (lo, node.end_lineno)
used = ["bloom_subset_pass", "or_combine", "clause_pass"]
helper_tot = dict(total=0, blank=0, comment=0, code=0)
helper_rows = []
for h in used:
    lo, hi = helpers[h]
    t = tally(common_cls, lo, hi)
    helper_rows.append((f"common.py::{h} (lines {lo}-{hi})", t))
    for k in helper_tot: helper_tot[k] += t[k]

print("## Table 1 — per file\n")
print("| file | total | blank | comment/docstring | code |\n|---|---:|---:|---:|---:|")
sums = {"C++ backend (.cu + host .py)": [0,1], "CuTe DSL backend (device .py + host .py)": [2,3], "Triton (2 kernels + used common.py helpers)": [4,5]}
for name, t in rows: print(fmt(name, t))
for name, t in helper_rows: print(fmt(name, t))
print(fmt("common.py helpers used, subtotal", helper_tot))
print()
for label, idx in sums.items():
    agg = dict(total=0, blank=0, comment=0, code=0)
    for i in idx:
        for k in agg: agg[k] += rows[i][1][k]
    if label.startswith("Triton"):
        for k in agg: agg[k] += helper_tot[k]
    print(fmt(f"**{label}**", agg))

# Role split, device-side files.
cu = CLS["cuda/codesigned_probe_score.cu (C++ device + launchers + pybind)"]
dsl = CLS["cute/codesigned_probe_score.py (DSL device + launchers + compile cache)"]
cu_roles = [
    ("header: file comment, includes, `#error` sm_61 gate", [(1, 37)]),
    ("device helpers: `seg_reduce_add`, constants, namespace", [(38, 56)]),
    ("kernel bodies: bloom mask, clause mask, score, generic (incl. their comment blocks)", [(57, 414)]),
    ("launchers + pybind: `cdiv`/`i64_ptr`, 3 launchers incl. `TORCH_CHECK`s, dispatch macros/switches, `PYBIND11_MODULE`", [(415, 672)]),
]
dsl_roles = [
    ("header: module docstring, imports, constants", [(1, 43)]),
    ("device helpers: `dp4a`, `load_int4`, `cttz64`, `seg_reduce_add`, `ld`, `st`, `@cute.jit` `_ld_id`/`_mask_bit`/`_mask_keep`/`_ld_row`", [(44, 83), (212, 264)]),
    ("kernel bodies: bloom mask, clause mask, score, generic (incl. docstrings)", [(84, 211), (265, 429)]),
    ("`@cute.jit` launchers + compile cache: 4 launchers, `gmem_ptr`, `cu_stream`, `_current_device`, `_compile`, 4 `compile_*`, per-device executor cache", [(430, 678)]),
]
def roles(cls, spec):
    print("| role | lines | total | blank | comment/docstring | code |\n|---|---|---:|---:|---:|---:|")
    agg = dict(total=0, blank=0, comment=0, code=0)
    for name, ranges in spec:
        t = dict(total=0, blank=0, comment=0, code=0)
        for lo, hi in ranges:
            for k, v in tally(cls, lo, hi).items(): t[k] += v
        for k in agg: agg[k] += t[k]
        rng = ", ".join(f"{lo}–{hi}" for lo, hi in ranges)
        print(f"| {name} | {rng} | {t['total']} | {t['blank']} | {t['comment']} | {t['code']} |")
    print(f"| **all** | | {agg['total']} | {agg['blank']} | {agg['comment']} | {agg['code']} |")
    assert agg["total"] == len(cls), (agg, len(cls))
print("\n## Table 2 — device-side files by role\n")
print("### C++ `cuda/codesigned_probe_score.cu`\n"); roles(cu, cu_roles)
print("\n### CuTe DSL `cute/codesigned_probe_score.py`\n"); roles(dsl, dsl_roles)
cu_src = (K / "cuda/codesigned_probe_score.cu").read_text()
print(f"\nTORCH_CHECK lines in the .cu launchers: {sum('TORCH_CHECK' in l for l in cu_src.splitlines())}")
host = (K / "codesigned_probe_score_cute.py").read_text().splitlines()
print(f"raise lines in the cute host module: {sum(l.strip().startswith('raise ') for l in host)}")
