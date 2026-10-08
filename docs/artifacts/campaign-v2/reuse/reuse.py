"""Which campaign-v2 cells existing records cover under the record reuse rule (roadmap Phase P,
"Manifest reuse entries"; docs/decisions.md § Campaign v2), CPU only.

    cd evaluation && uv run python ../docs/artifacts/campaign-v2/reuse/reuse.py <fetched root> <out>

<fetched root> holds one directory per hub-index subtree (`bench fetch --results <root>/<subtree>`,
as docs/artifacts/campaign-v2/inventory.py reads it). The v2 grid is expanded from the checked-in
evaluation/config by the harness's own `load_matrix`, so a cell here is exactly a key block the
campaign writes. Writes into <out>:

  cells.csv     one row per v2 cell: quality and perf each `reuse <code_version> <subtree>` or `run`,
                and why
  groups.csv    one row per manifest match group (dataset, suite, algo, backend, filter_kind, sweep)
  entries.yaml  the manifest entries those groups imply, coarsest uniform match first
  claims.csv    every claims.yaml selector against the v2 grid: the arms it hits (must be 1)
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from loguru import logger

from bench.config import load_dataset, load_matrix
from bench.records import key_block

CONFIG = Path(__file__).resolve().parents[4] / "evaluation"
FROZEN = "408b1188d542634b3d18a2f5077bd23537a845fc"
# Subtrees whose arXiv records postdate the license-clause attrs fix (validation.md, Campaign).
ARXIV_FINAL = ("d1/arxiv", "d1/arxiv-deep", "d1/arxiv-codesign", "artifacts/h2")
# The device max SM clock for records that predate env.sm_max_mhz (nvidia-smi max clocks).
DEVICE_MAX_MHZ = {"NVIDIA A100-SXM4-80GB": 1410}
CLOCK_FRACTION_MAX = 0.10
# Suites whose claims are ratios (C1/C2/C3 synth, C5 codesign, C7 h2h): timed interleaved only.
INTERLEAVED_ONLY = {"synth", "codesign", "h2h"}
# Code versions of the library tree, oldest first: 72e5a90 (D1) -> c0e42d1 (32e6a66: per-width
# probe tiles, chunked 1-bit build) -> cf2c941 (#7: V4 / Int8 removed) -> 408b1188 (#14: item-range
# evaluate_mask). The per-arm evidence that each step left the arm unchanged is ARM below.
KNOWN = {
    "72e5a90c148435d070496ae59e8b4c26bb825d68",
    "c0e42d1ae2de37774ba01d7723a2b7a190b9d73d",
    "cf2c94111cf3e6ca0264102afd17527b4258e3a6",
    FROZEN,
}

# (algo, backend) -> (quality evidence, timing evidence or None = timing not reusable at any dim,
# max D_PAD whose timing is reusable)
ARM = {
    ("linr_v1_filter_mask", "triton"): (
        "golden V1 bit-exact 408b1188=c0e42d1=72e5a90 (freeze gate 3, H2 27 cells); diff: "
        "evaluate_mask slice [0:None] only (G-range bit-exact)",
        "query kernels unchanged (cuBLAS mm + clause_mask/bloom_match + topk; #14 adds a host-side "
        "row-slice view)",
        None,
    ),
    ("linr_v2", "triton"): (
        "golden V2 bit-exact 408b1188=c0e42d1=72e5a90 (freeze gate 3, H2 27 cells); diff: none on "
        "its path (evaluate_indices, fused_masked_knn_topk untouched)",
        "query kernels unchanged",
        None,
    ),
    ("linr_v3", "triton"): (
        "golden V3 bit-exact 408b1188=c0e42d1=72e5a90 (freeze gate 3, H2 27 cells); diff: chunked "
        "1-bit build, bits torch.equal on arXiv and yfcc10m (row Build-time 1-bit quantization)",
        "query kernels unchanged (32e6a66 touched the build only)",
        None,
    ),
    ("silvertorch", "triton"): (
        "golden SilverTorch triton bit-exact 408b1188=c0e42d1=72e5a90 (freeze gate 3, H2 27 cells); "
        "diff: per-width tiles, every config torch.equal to 256x4 at every width",
        "probe tiles unchanged at D_PAD <= 256, SASS identical at D 128 / 192 (7 of 7 cubins); "
        "#16 reverted",
        256,
    ),
    ("silvertorch", "official"): (
        "official_commit 21aa35e unchanged; no diff in the adapter since 72e5a90; golden official "
        "cells bit-exact (freeze gate 3)",
        "official_commit and adapter unchanged",
        None,
    ),
}


def subtree_records(root: Path):
    for path in sorted(root.rglob("*.jsonl")):
        if path.name.endswith(".samples.jsonl"):
            continue
        sub = "/".join(path.relative_to(root).parts[:-2])
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(rec, dict)
                and {"algo", "suite", "dataset", "env"} <= rec.keys()
            ):
                yield sub, rec


def v2_cells() -> dict[str, dict]:
    """Every key block the checked-in v2 suites expand to, with its (ks, batch sizes)."""
    logger.remove()
    suites = yaml.safe_load((CONFIG / "config/suites.yaml").read_text())
    out = {}
    for name, spec in suites.items():
        if not isinstance(spec, dict) or "arms" not in spec:
            continue
        for ds in spec["datasets"]:
            jobs = load_matrix(
                CONFIG / f"config/{ds}.yaml", CONFIG / "config/suites.yaml", name
            )
            for job in jobs:
                for params in job.cells():
                    kb = job.key(params)
                    out[json.dumps(kb, sort_keys=True)] = {
                        "key": kb,
                        "ks": job.ks,
                        "bs": job.batch_sizes,
                    }
    return out


def clock_fraction(rec: dict, ks, bss) -> float | None:
    env = rec["env"]
    mx = env.get("sm_max_mhz") or DEVICE_MAX_MHZ.get(env.get("gpu"))
    w = [
        s
        for e in rec.get("perf") or []
        if e.get("k") in ks and e.get("bs") in bss
        for s in e.get("window_sm_mhz") or []
        if s
    ]
    return sum(s < mx for s in w) / len(w) if w and mx else None


def judge(sub: str, rec: dict, cell: dict, e1c: str) -> tuple[str | None, str | None]:
    """(why quality is not reusable, why perf is not) for one record of a v2 cell; None = reusable."""
    d, cv = rec["dataset"], rec["env"]["code_version"]
    if rec.get("status") not in ("ok", "partial"):
        return "status " + str(rec.get("status")), "status " + str(rec.get("status"))
    if d.startswith("goodreads") and key_block(rec)["inputs"] != e1c:
        inp = "inputs " + key_block(rec)["inputs"] + " (E1c is final)"
        return inp, inp
    if d.startswith("arxiv") and sub not in ARXIV_FINAL:
        return (
            "arXiv attrs before the license fix",
            "arXiv attrs before the license fix",
        )
    if d == "pubmed":
        return (
            "pending V-PUBMED embedding identity check",
            "pending V-PUBMED embedding identity check",
        )
    if cv not in KNOWN:
        off = f"code_version {cv[:8]} not on the 72e5a90 -> 408b1188 chain"
        return off, off
    arm = ARM.get((rec["algo"], rec["backend"]))
    if arm is None:
        return "no per-arm evidence", "no per-arm evidence"
    wq = None if rec.get("quality") is not None else "no quality"
    if not rec.get("perf"):
        return wq, "no perf"
    if arm[2] is not None and 1 << (rec["dim"] - 1).bit_length() > arm[2]:
        return (
            wq,
            f"timed on the D_PAD {1 << (rec['dim'] - 1).bit_length()} tile the frozen code replaced",
        )
    if rec["suite"] in INTERLEAVED_ONLY and not rec.get("interleave"):
        return wq, "ratio suite, not interleaved"
    frac = clock_fraction(rec, cell["ks"], cell["bs"])
    if frac is None:
        return wq, "no window clock samples"
    if frac >= CLOCK_FRACTION_MAX:
        return wq, f"clock: {frac:.2f} of windows below the device max"
    return wq, None


def main(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cells = v2_cells()
    e1c = load_dataset(CONFIG / "config/goodreads.yaml", 128).inputs
    # per v2 cell: code_version -> (subtree, quality why, perf why, elapsed_s)
    found: dict[str, dict[str, tuple]] = defaultdict(dict)
    outside: dict[tuple, set] = defaultdict(
        set
    )  # level-5 match -> sweeps of records off the v2 grid
    v2_datasets = {c["key"]["dataset"] for c in cells.values()}
    for sub, rec in subtree_records(root):
        if rec["dataset"] not in v2_datasets:
            continue  # kuairand / openalex / yambda: no v2 cell, and no input identity left
        kb = key_block(rec)
        s = json.dumps(kb, sort_keys=True)
        if s not in cells:
            if rec.get("status") in ("ok", "partial"):
                outside[
                    (
                        kb["dataset"],
                        kb["suite"],
                        kb["algo"],
                        kb["backend"],
                        kb["filter_kind"],
                    )
                ].add((kb["sweep"], json.dumps(kb["params"], sort_keys=True)))
            continue
        wq, wp = judge(sub, rec, cells[s], e1c)
        prev = found[s].get(rec["env"]["code_version"])
        if prev is None or (prev[1] or prev[2]) and not (wq or wp):
            found[s][rec["env"]["code_version"]] = (sub, wq, wp, rec.get("elapsed_s"))

    rows, groups = [], defaultdict(list)
    for s, cell in cells.items():
        kb = cell["key"]
        g = (
            kb["dataset"],
            kb["suite"],
            kb["algo"],
            kb["backend"],
            kb["filter_kind"],
            kb["sweep"],
        )
        q = {cv: v[0] for cv, v in found[s].items() if v[1] is None}
        p = {cv: v[0] for cv, v in found[s].items() if v[2] is None}
        whyq = "; ".join(
            f"{cv[:8]}@{v[0]}: {v[1]}" for cv, v in found[s].items() if v[1]
        ) or ("" if q else "no record")
        whyp = "; ".join(
            f"{cv[:8]}@{v[0]}: {v[2]}" for cv, v in found[s].items() if v[2]
        ) or ("" if p else "no record")
        el = max((v[3] or 0 for v in found[s].values()), default=0)
        groups[g].append((q, p, el))
        rows.append(
            [*g, json.dumps(kb["params"], sort_keys=True), kb["seed"], q, p, whyq, whyp]
        )

    decision = {}
    for g, cs in groups.items():

        def common(side):
            sets = [set(c[side]) for c in cs]
            cvs = set.intersection(*sets)
            return (
                (sorted(cvs)[0], sorted({c[side][sorted(cvs)[0]] for c in cs}))
                if cvs
                else None
            )

        dq, dp = common(0), common(1)
        if dq and dp and dp[0] != dq[0]:
            dp = None  # one record per cell supplies perf; a mixed pair is not worth an entry
        if dq or dp:
            decision[g] = (dq, dp)

    # Entries, coarsest uniform level first: an entry covers every cell under its match, so it is
    # taken only when every v2 group below shares the decision and no record off the v2 grid (a
    # dropped sweep or grid point, e.g. n_probe 32) sits under it: those stay unreferenced.
    def side(d, s):
        return (
            {"code_version": d[s][0], "hub": ",".join(d[s][1])}
            if d[s]
            else {"code_version": FROZEN, "hub": "campaign-v2/{dataset}-{suite}"}
        )

    entries, done = [], set()
    for level in (4, 5, 6):
        by = defaultdict(list)
        for g in groups:
            by[g[:level]].append(g)
        for pre, gs in sorted(by.items()):
            if any(g in done for g in gs) or not all(g in decision for g in gs):
                continue
            if len({json.dumps(decision[g], sort_keys=True) for g in gs}) != 1:
                continue
            if any((*o, sw)[:level] == pre for o, v in outside.items() for sw, _ in v):
                continue
            d = decision[gs[0]]
            match = dict(
                zip(
                    ("dataset", "suite", "algo", "backend", "filter_kind", "sweep"), pre
                )
            )
            arm = ARM[(pre[2], pre[3])]
            entries.append(
                {
                    "match": match,
                    "quality": side(d, 0),
                    "perf": side(d, 1),
                    "_evidence": arm[0] if d[0] else "",
                    "_timing": arm[1] if d[1] else "",
                    "_cells": sum(len(groups[g]) for g in gs),
                }
            )
            done.update(gs)
    (out / "entries.yaml").write_text(
        yaml.safe_dump(entries, sort_keys=False, width=200)
    )

    def via(g, i):
        return "reuse" if g in done and decision[g][i] else "run"

    with open(out / "groups.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "dataset",
                "suite",
                "algo",
                "backend",
                "filter_kind",
                "sweep",
                "v2_cells",
                "quality_covered",
                "perf_covered",
                "entry_quality",
                "entry_perf",
                "old_elapsed_s_perf_covered",
            ]
        )
        for g, cs in sorted(groups.items()):
            wr.writerow(
                [
                    *g,
                    len(cs),
                    sum(bool(c[0]) for c in cs),
                    sum(bool(c[1]) for c in cs),
                    via(g, 0),
                    via(g, 1),
                    round(sum(c[2] for c in cs if c[1]), 1),
                ]
            )
    with open(out / "cells.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "dataset",
                "suite",
                "algo",
                "backend",
                "filter_kind",
                "sweep",
                "params",
                "seed",
                "quality",
                "perf",
                "quality_why",
                "perf_why",
                "entry_quality",
                "entry_perf",
            ]
        )
        for r in sorted(rows, key=lambda r: [str(x) for x in r[:8]]):
            q = " ".join(f"{cv[:8]}@{sub}" for cv, sub in r[8].items()) or "run"
            p = " ".join(f"{cv[:8]}@{sub}" for cv, sub in r[9].items()) or "run"
            wr.writerow(
                [*r[:8], q, p, *r[10:], via(tuple(r[:6]), 0), via(tuple(r[:6]), 1)]
            )

    claims = yaml.safe_load((CONFIG / "claims.yaml").read_text())["claims"]
    with open(out / "claims.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["claim", "label", "side", "arms"])
        for c in claims:
            for item in c["ours"]:
                for sd in ("where", "vs"):
                    if sd not in item:
                        continue
                    where = dict(item[sd])
                    params = where.pop("params", {})
                    arms = {
                        (
                            cl["key"]["algo"],
                            cl["key"]["backend"],
                            json.dumps(cl["key"]["params"], sort_keys=True),
                        )
                        for cl in cells.values()
                        if all(
                            cl["key"].get(f) == v
                            for f, v in where.items()
                            if v is not None
                        )
                        and all(
                            cl["key"]["params"].get(n) == v for n, v in params.items()
                        )
                    }
                    wr.writerow([c["id"], item["label"], sd, len(arms)])

    n = len(cells)
    nq = sum(len(groups[g]) for g in done if decision[g][0])
    npf = sum(len(groups[g]) for g in done if decision[g][1])
    cq = sum(bool(r[8]) for r in rows)
    cp = sum(bool(r[9]) for r in rows)
    print(
        f"{n} v2 cells; reused through the manifest: quality {nq}, perf {npf} "
        f"({len(entries)} entries); a reusable record exists for quality {cq}, perf {cp} -> {out}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
