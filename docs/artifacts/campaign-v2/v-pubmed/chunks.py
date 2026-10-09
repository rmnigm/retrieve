"""V-PUBMED filter chunk plan, checked against the real `filter` suite on CPU: every chunk is a CLI narrowing (--algo /
--backend / --filter-kind / --sweep / --seed, never --k / --bs / --mode), its cells' resume keys and interleave units equal
the full expansion's, the chunks partition the whole suite (no manifest reuse on PubMed), and no chunk splits an interleave
unit. Prints one `bench run` argument line per chunk, shortest first (est_min).

    python chunks.py            (cwd evaluation/)
"""

from pathlib import Path

from bench.config import interleave_units, load_matrix
from bench.records import resume_key

DS, SUITE = Path("config/pubmed.yaml"), Path("config/suites.yaml")
SEEDS = (0, 1, 2)
KS = [("clause", w) for w in ("c0_mesh", "c3_journal_reverse", "all5")] + [("bloom", "c0_mesh")]
GROUPS = (
    (["linr_v1_filter_mask", "linr_v2"], ["triton"]),
    (["linr_v3"], ["triton"]),
    (["silvertorch"], ["triton", "official"]),
    (["postfilter"], ["torch"]),
)
CHUNKS = [dict(algos=a, backends=b, filter_kinds=[k], sweeps=[w], seeds=[s])
          for a, b in GROUPS for s in SEEDS for k, w in KS]  # fmt: skip


def est_min(c):
    """Rough minutes per chunk (st-dloop meets short gaps first): V1+V2 measured 15 at c0_mesh and >= 60 at
    c3_journal_reverse (V2's per-row gather at p ~ 1, D1's slowest cell); the rest unmeasured at v2.1."""
    algo, kind, sweep = c["algos"][0], c["filter_kinds"][0], c["sweeps"][0]
    if algo == "linr_v1_filter_mask":
        return 75 if sweep == "c3_journal_reverse" else 15
    return {"silvertorch": 25 if kind == "bloom" else 12, "linr_v3": 18, "postfilter": 20}[algo]


CHUNKS.sort(key=est_min)  # stable: seed order kept within an estimate


def cells(jobs):
    return {resume_key(j.key({**j.build, **q}), "CV") for j in jobs for q in j.query}


def units(jobs):
    return {tuple(sorted(resume_key(j.key(j.build), "CV") for j in m)) for by, m in interleave_units(jobs) if by}


full = load_matrix(DS, SUITE, "filter")
seen = set()
for c in CHUNKS:
    js = load_matrix(DS, SUITE, "filter", **c)
    assert not any(j.narrowed for j in js), c
    sub = [j for j in full if j.algo in c["algos"] and j.backend in c["backends"] and j.filter_kind in c["filter_kinds"]
           and j.sweep in c["sweeps"] and j.seed in c["seeds"]]  # fmt: skip
    if not sub:
        continue
    assert cells(js) == cells(sub) and units(js) == units(sub) <= units(full), c
    assert not (cells(js) & seen), c
    seen |= cells(js)
    args = " ".join(f"--{f.rstrip('s').replace('_', '-')} {v}" for f, vs in c.items() for v in vs)
    print(f"{len(cells(js)):3d} cells  {args}")
assert seen == cells(full), (len(seen), len(cells(full)))
print(f"{len(seen)} cells = the full suite; every interleave unit inside one chunk")
