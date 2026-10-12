"""V-YFCC synth chunk plan (yfcc10m-synth `synth`, shared between pods 1 and c), checked against the real suite on
CPU like V-PUBMED's chunks.py: every chunk is a CLI narrowing (--algo / --backend / --filter-kind / --sweep), its
cells' resume keys and interleave units equal the full expansion's, the chunks partition the suite, and no chunk
splits an interleave unit. Prints `name  cells  args` per chunk, the costliest groups (V3, V1+V2) first.

    python yfcc_chunks.py            (cwd evaluation/)
"""

from pathlib import Path

from bench.config import interleave_units, load_matrix
from bench.records import resume_key

DS, SUITE = Path("config/yfcc10m-synth.yaml"), Path("config/suites.yaml")
SWEEPS = ("p0001", "p001", "p003", "p01", "p1")
GROUPS = (  # name, algos, backends, filter kinds
    ("v3", ["linr_v3"], ["triton"], ["clause", "bloom"]),
    ("v1v2", ["linr_v1_filter_mask", "linr_v2"], ["triton"], ["clause", "bloom"]),
    ("post", ["postfilter"], ["torch"], ["clause", "bloom"]),
    ("st", ["silvertorch"], ["triton"], ["clause"]),
    ("stbloom", ["silvertorch"], ["triton", "official"], ["bloom"]),
)
CHUNKS = [
    (
        f"{g}-{k}-{s}" if g != "stbloom" else f"{g}-{s}",
        dict(algos=a, backends=b, filter_kinds=[k], sweeps=[s]),
    )
    for g, a, b, ks in GROUPS
    for k in ks
    for s in SWEEPS
]


def cells(jobs):
    return {resume_key(j.key({**j.build, **q}), "CV") for j in jobs for q in j.query}


def units(jobs):
    return {
        tuple(sorted(resume_key(j.key(j.build), "CV") for j in m))
        for by, m in interleave_units(jobs)
        if by
    }


full = load_matrix(DS, SUITE, "synth")
seen = set()
for name, c in CHUNKS:
    js = load_matrix(DS, SUITE, "synth", **c)
    assert not any(j.narrowed for j in js), name
    sub = [j for j in full if j.algo in c["algos"] and j.backend in c["backends"]
           and j.filter_kind in c["filter_kinds"] and j.sweep in c["sweeps"]]  # fmt: skip
    assert sub, name
    assert cells(js) == cells(sub) and units(js) == units(sub) <= units(full), name
    assert not (cells(js) & seen), name
    seen |= cells(js)
    args = " ".join(
        f"--{f.rstrip('s').replace('_', '-')} {v}" for f, vs in c.items() for v in vs
    )
    print(f"{name}  {len(cells(js))}  {args}")
assert seen == cells(full), (len(seen), len(cells(full)))
print(f"{len(seen)} cells = the full suite; every interleave unit inside one chunk")
