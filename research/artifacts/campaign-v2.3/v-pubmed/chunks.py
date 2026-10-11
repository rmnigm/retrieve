"""V-PUBMED filter chunk plan, checked against the real `filter` suite on CPU: every chunk is a CLI narrowing (--algo /
--backend / --filter-kind / --sweep / --seed, never --k / --bs / --mode), its cells' resume keys and interleave units equal
the full expansion's, the chunks partition the whole suite (no manifest reuse on PubMed), and no chunk splits an interleave
unit. Prints one `bench run` argument line per chunk, shortest first (est_min).

    VPUBMED_CHUNKS=all|nost|st [VPUBMED_DONE=a.jsonl:b.jsonl] python chunks.py   (cwd evaluation/)

The assertion always covers all chunks; VPUBMED_CHUNKS only filters the printed list: `nost` every chunk but SilverTorch,
`st` the SilverTorch ones (run last, after V-SEEDS arXiv: controller, 2026-10-09). VPUBMED_DONE lists record files of
earlier code versions: a chunk whose cells are all `ok` there (key block, code_version aside) is not printed, so a new tag
runs only what remains (campaign-v2.3: earlier records stay at their versions, on the redo ledger). VPUBMED_SEEDS
(e.g. `0`) prints only those seeds' chunks: claims first (user, 2026-10-10), seeds 1-2 go to F-REPRO.
"""

import os
from pathlib import Path

from bench.config import interleave_units, load_matrix
from bench.records import key_block, read_records, resume_key

DS, SUITE = Path("config/pubmed.yaml"), Path("config/suites.yaml")
SEEDS = (0, 1, 2)
KS = [("clause", w) for w in ("c0_mesh", "c3_journal_reverse", "all5")] + [
    ("bloom", "c0_mesh")
]
GROUPS = (
    (["linr_v1_filter_mask", "linr_v2"], ["triton"]),
    (["linr_v3"], ["triton"]),
    (["silvertorch"], ["triton", "official"]),
    (["postfilter"], ["torch"]),
)
CHUNKS = [dict(algos=a, backends=b, filter_kinds=[k], sweeps=[w], seeds=[s])
          for a, b in GROUPS for s in SEEDS for k, w in KS]  # fmt: skip


def est_min(c):
    """Minutes per chunk, measured at v2.1 where known (st-dloop meets short gaps first): V1+V2 15 at low p, 62 at
    c3_journal_reverse (V2's per-row gather at p ~ 1); SilverTorch clause 37 (the n_probe 1024 cell: 1,998 s), bloom
    with official at 1024 ~70 (est.); V3 and postfilter unmeasured (est. 18 / 20)."""
    algo, kind, sweep = c["algos"][0], c["filter_kinds"][0], c["sweeps"][0]
    if algo == "linr_v1_filter_mask":
        return 75 if sweep == "c3_journal_reverse" else 15
    if algo == "silvertorch":
        return 70 if kind == "bloom" else 37
    return {"linr_v3": 18, "postfilter": 20}[algo]


CHUNKS.sort(key=est_min)  # stable: seed order kept within an estimate


def cells(jobs):
    return {resume_key(j.key({**j.build, **q}), "CV") for j in jobs for q in j.query}


def units(jobs):
    return {
        tuple(sorted(resume_key(j.key(j.build), "CV") for j in m))
        for by, m in interleave_units(jobs)
        if by
    }


full = load_matrix(DS, SUITE, "filter")
SELECT = os.environ.get("VPUBMED_CHUNKS", "all")
SEEDS = {int(x) for x in os.environ.get("VPUBMED_SEEDS", "").split(",") if x}
DONE = {resume_key(key_block(r), "CV") for f in filter(None, os.environ.get("VPUBMED_DONE", "").split(":"))
        for r in read_records(Path(f)) if r["status"] == "ok"}  # fmt: skip
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
    args = " ".join(
        f"--{f.rstrip('s').replace('_', '-')} {v}" for f, vs in c.items() for v in vs
    )
    st = c["algos"] == ["silvertorch"]
    seeds_ok = not SEEDS or c["seeds"][0] in SEEDS
    if (
        (SELECT == "all" or (SELECT == "st") == st)
        and seeds_ok
        and not cells(js) <= DONE
    ):
        print(f"{len(cells(js)):3d} cells  {args}")
assert seen == cells(full), (len(seen), len(cells(full)))
print(f"{len(seen)} cells = the full suite; every interleave unit inside one chunk")
