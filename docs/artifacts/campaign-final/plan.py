"""Writes docs/artifacts/campaign-final/recipe.yaml (README.md): the final pass's legs per GPU, ordered, with their
dependencies and commands, priced from the F-REPRO sizing inventory (docs/artifacts/f-repro/sizing.py, `inventory.csv`).

Placement (Plan A', controller 2026-10-13: box = host, seed 0 + filter seeds 0-2): every family that compares against
V1 / V2 shares pod d (C1, C2's V3 / V2, C6's IVF vs exact, C7, T2's columns incl. postfilter, C3's synth V2 torch);
self-contained families go elsewhere (pod b: PubMed / YFCC tune, v3bits; pod 1: c3 / c3-real, C4, C5, the goodreads /
arXiv tune). Within a host, the two GPUs split the families; one interleave group always runs in one process.

usage: plan.py INVENTORY.csv > recipe.yaml
"""

import collections
import csv
import json
import sys

OVERHEAD = 1.15  # median wall / (summed cell time + index builds) over 20 legs (sizing.py, overhead.csv)
DIM = {"laion1m": 256, "laion3m": 256, "laion10m": 256, "laion1m-synth": 256, "laion3m-synth": 256,
       "laion10m-synth": 256, "goodreads": 128, "goodreads-synth": 128, "arxiv": 128, "arxiv-synth": 128, "arxiv-corr-synth": 128,
       "yfcc10m": 192, "yfcc10m-synth": 192, "pubmed": 768, "laion30m": 256, "laion30m-synth": 256}  # fmt: skip
# quality-only tune legs are build-bound, which elapsed_s misses: the measured legs (validation, night-queue item 8)
TUNE_Q_MEASURED = {"pubmed": 2.19, "yfcc10m": 0.4, "yfcc10m-synth": 0.32, "goodreads": 0.2, "goodreads-synth": 0.22,
                   "arxiv": 0.25, "arxiv-synth": 0.22, "arxiv-corr-synth": 0.21}  # fmt: skip
# timed tune legs measured at v2.11 (campaign-v2.11/{goodreads,yfcc10m}-tune-timed: (elapsed + builds) x OVERHEAD over the
# final grid's capped cells; pubmed-tune-timed: its leg's 3.25 GPU-h wall); arXiv's is not measured yet
TUNE_T_MEASURED = {"goodreads": 1.22, "goodreads-synth": 1.81, "yfcc10m": 0.98, "yfcc10m-synth": 0.98, "pubmed": 3.25}  # fmt: skip
# the N-sweep (never run): per-leg cell seconds from the 30 M analogues (sizing inventory): exact arms laion30m-synth
# 252 s/cell and laion30m-x 257 / 151 s scaled by N / 30 M with a 30 s floor (arXiv 3 M d128 measured 32 s), arXiv d256 at
# 1.5x its d128 cells; SilverTorch and co-design cells ~flat (30 M: 34 / 35 / 28 s)
NSWEEP_CELL_S = {
    ("nsweep-synth", "laion1m-synth"): 18 * 30 + 12 * 34, ("nsweep-synth", "laion3m-synth"): 18 * 30 + 12 * 34,
    ("nsweep-synth", "laion10m-synth"): 18 * 84 + 12 * 34, ("nsweep-synth", "arxiv-synth"): 18 * 47 + 12 * 44,
    ("nsweep", "laion1m"): 4 * 30 + 8 * 35, ("nsweep", "laion3m"): 4 * 30 + 8 * 35,
    ("nsweep", "laion10m"): 2 * 86 + 2 * 50 + 8 * 35,
    ("nsweep-codesign", "laion1m"): 16 * 28, ("nsweep-codesign", "laion3m"): 16 * 28,
    ("nsweep-codesign", "laion10m"): 16 * 28,
    ("nsweep-synth", "laion30m-synth"): 18 * 252 + 12 * 34, ("nsweep", "laion30m", "linr_v2", "silvertorch"): 2 * 151 + 8 * 35,
    ("nsweep", "laion30m", "linr_v1_filter_mask"): 2 * 257,
    ("nsweep-codesign", "laion30m"): 16 * 28,
    # seeds legs, seeds 0-2 (bs 16 / 64 only, so upper bounds): arXiv 3 M 32 s, YFCC synth 83 s, LAION 30 M synth 252 s, c7 35 s
    ("seeds-c1", "arxiv-synth"): 12 * 32, ("seeds-c1", "yfcc10m-synth"): 12 * 83,
    ("seeds-c1-laion30m", "laion30m-synth"): 12 * 252, ("seeds-c7", "laion30m"): 36 * 35,
}  # fmt: skip
REAL = ["goodreads", "arxiv", "yfcc10m", "pubmed"]
OFFICIAL_VENV = "/venvs/final-o3"  # scripts/build_official_o3.sh; env.official_build records the .so sha

# (gpu, family, suite, datasets, algos (None = whole suite), backend, flags, note)
LEGS = [
    ("d0", "M1", "filter", ["arxiv"], ["silvertorch", "linr_v2"], None, "", "M1 gate: alone, then with GPU 1 loaded (roadmap M1); gates every d1 timed leg"),
    *[("d0", "C1", "filter", [d], ["linr_v1_filter_mask", "linr_v2"], None, "--interleave", "") for d in REAL],
    *[("d0", "C1", "synth", [d], ["linr_v1_filter_mask", "linr_v2"], "triton", "--interleave", "") for d in ["goodreads-synth", "arxiv-synth", "yfcc10m-synth"]],
    ("d0", "C1", "laion30m", ["laion30m"], ["linr_v1_filter_mask", "linr_v2"], None, "--interleave", "one process per sweep (68 GB)"),
    ("d0", "C6", "laion30m-x", ["laion30m"], ["linr_v2", "silvertorch"], None, "--interleave", "per sweep; then V1 alone (next leg)"),
    ("d0", "C1", "laion30m-x", ["laion30m"], ["linr_v1_filter_mask"], None, "", ""),
    *[("d0", "T2", "filter", [d], ["postfilter"], None, "", "T2's generic-torch column, same box as the exact arms") for d in REAL],
    ("d0", "C3", "synth", ["goodreads-synth"], ["linr_v2"], "torch", "", "C3's V2 side, same box as V2 triton"),
    *[("d0", "T2", "synth", [d], ["postfilter"], None, "", "") for d in ["goodreads-synth", "arxiv-corr-synth"]],
    *[("d1", "C6+C7", "filter", [d], ["silvertorch"], None, "--interleave", "triton then official in one stream (parity spill); official from the -O3 venv") for d in REAL],
    *[("d1", "C6+C7", "deep", [d], None, None, "--interleave", "official bloom curve paired with triton (goodreads, arXiv)") for d in REAL],
    *[("d1", "C6+C7", "synth", [d], ["silvertorch"], None, "--interleave", "") for d in ["goodreads-synth", "arxiv-synth", "arxiv-corr-synth", "yfcc10m-synth"]],
    ("d1", "C6", "laion30m", ["laion30m"], ["silvertorch"], None, "", ""),
    ("d1", "C6", "laion30m-bs1", ["laion30m"], None, None, "", ""),
    ("d1", "C1+C6", "laion30m-synth", ["laion30m-synth"], None, None, "--interleave", "V1 / V2 (C1) and SilverTorch (C6) at 30 M synth"),
    *[("d1", "C7", "h2h", [d], None, None, "--interleave --profile", "") for d in ["goodreads", "arxiv"]],
    ("d1", "C7", "c7-scorepath", ["laion30m"], None, None, "--interleave --profile --mode eager", ""),
    *[("d1", "C2", "filter", [d], ["linr_v3"], None, "", "V3 / V2 against d0's V2, same host") for d in ["goodreads", "yfcc10m"]],
    ("d1", "C2", "synth", ["arxiv-corr-synth"], ["linr_v3"], None, "", ""),
    # pod d rebalance (controller 2026-10-13): three C2 legs moved d1 -> d0 (same host, so V3 / V2 stays one box), end of
    # laion's queue; d1 keeps its other ids (holes at 20, 22, 23, see SKIP)
    *[("d0", "C2", "filter", [d], ["linr_v3"], None, "", "V3 / V2 against V2, same host; moved from d1 (rebalance)") for d in ["arxiv", "pubmed"]],
    ("d0", "C2", "synth", ["goodreads-synth"], ["linr_v3"], None, "", "moved from d1 (rebalance)"),
    *[("b", "tune", s, [d], None, None, "--skip-perf" if s == "tune-q" else "", "") for s in ("tune-q", "tune-timed") for d in ["pubmed", "yfcc10m", "yfcc10m-synth"]],
    # the goodreads bench moved off pod 1 (controller 2026-10-13: v2.11 timed tune slower than sized, pod 1 needs arXiv headroom)
    *[("b", "tune", s, [d], None, None, "--skip-perf" if s == "tune-q" else "", "") for s in ("tune-q", "tune-timed") for d in ["goodreads", "goodreads-synth"]],
    *[("b", "C2", "v3bits", [d], None, None, "", "k_bits variants against each other") for d in ["goodreads-synth", "goodreads", "pubmed"]],
    # seeds 0-2 on one box where a CI rests on them (controller 2026-10-13): C1's crossover cells, C7's 30 M bs 64 cell
    *[("b", "C1 seeds", "seeds-c1", [d], None, None, "--interleave", "") for d in ["arxiv-synth", "yfcc10m-synth"]],
    ("b", "C1 seeds", "seeds-c1-laion30m", ["laion30m-synth"], None, None, "--interleave", "needs LAION 30 M on pod b"),
    ("b", "C7 seeds", "seeds-c7", ["laion30m"], None, None, "--interleave --profile --mode eager", "official from the -O3 venv; needs LAION 30 M on pod b"),
    *[("p1", "C3", "c3", [d], None, None, "", "") for d in ["goodreads-synth", "arxiv-synth", "yfcc10m-synth", "laion30m-synth"]],
    *[("p1", "C3", "c3-real", [d], None, None, "", "") for d in REAL],
    *[("p1", "C5", "codesign", [d], None, None, "--interleave", "") for d in ["arxiv", "goodreads"]],
    ("p1", "C5", "codesign-pubmed", ["pubmed"], None, None, "--interleave", ""),
    ("p1", "C5", "codesign-laion30m", ["laion30m"], None, None, "--interleave --mode eager", "needs LAION on pod 1, else runs on d1"),
    *[("p1", "C4", s, [d], None, None, "", "") for s in ("bloomwidth", "bloomwidth-timed") for d in ["goodreads", "arxiv", "pubmed"]],
    *[("p1", "tune", s, [d], None, None, "--skip-perf" if s == "tune-q" else "", "") for s in ("tune-q", "tune-timed")
      for d in ["arxiv", "arxiv-synth", "arxiv-corr-synth"]],
    # the fixed-d N-sweep extension (controller 2026-10-13): all of it on pod 1, one box across 1 / 3 / 10 M
    *[("p1", "N-sweep C1+C6", "nsweep-synth", [d], None, None, "--interleave", "") for d in ["laion1m-synth", "laion3m-synth", "laion10m-synth", "laion30m-synth", "arxiv-synth"]],
    *[("p1", "N-sweep C6", "nsweep", [d], None, None, "--interleave", "V1, V2 and SilverTorch in one group per sweep") for d in ["laion1m", "laion3m", "laion10m"]],
    ("p1", "N-sweep C6", "nsweep", ["laion30m"], ["linr_v2", "silvertorch"], None, "--interleave", "30 M: V1 + V2 + SilverTorch exceed one A100; V1 alone next"),
    ("p1", "N-sweep C6", "nsweep", ["laion30m"], ["linr_v1_filter_mask"], None, "", ""),
    *[("p1", "N-sweep C5", "nsweep-codesign", [d], None, None, "--interleave --mode eager", "official from the -O3 venv") for d in ["laion1m", "laion3m", "laion10m", "laion30m"]],
]  # fmt: skip
SHORT = {"linr_v1_filter_mask": "v1", "linr_v2": "v2", "linr_v3": "v3", "silvertorch": "st", "postfilter": "pf"}
# C6 ceiling checks (script legs, quality only): one per bench on C6's GPU, after the bench's tune; GPU-h from the measured runs
# (validation: goodreads + arXiv 9 min, YFCC 3 min, PubMed 21 min, LAION 44 min); LAION has no tune leg in the final grid
CEILING = {"goodreads": (["goodreads", "goodreads-synth"], 0.08), "arxiv": (["arxiv", "arxiv-synth"], 0.08),
           "yfcc10m": (["yfcc10m", "yfcc10m-synth"], 0.05), "pubmed": (["pubmed"], 0.35), "laion30m": ([], 0.75)}  # fmt: skip
# leg numbers left unused so the running workers' ids stay stable after the pod d rebalance (d1-20 / -22 / -23 moved to d0)
SKIP = {"d1": {20, 22, 23}}
GPUS = {"d0": ("a100-x2-d", 0), "d1": ("a100-x2-d", 1), "b": ("a100-x1-b", 0), "p1": ("a100-x1-eval", 0)}
WORKER = {"d0": "laion", "d1": "d-run", "b": "v-pubmed", "p1": "v-pod1-run"}  # the final-pass brief's table
BENCH = '"$VENV/bin/python" -m bench.cli'  # the leg venv's own interpreter: `uv run` masks bench's exit code


def price(inv, suite, ds, algos, backend):
    cells, h, build = 0, 0.0, 0.0
    for r in inv:
        if r["source"] != "suites.yaml" or r["suite"] != suite or r["dataset"] != ds:
            continue
        algo, be = r["arm"].split()[0].split("/")
        if algos is not None and algo not in algos or backend and be != backend:
            continue
        cells += int(r["cells"])
        build += float(r["gpu_h_build"])  # index builds: `elapsed_s` excludes them
        h += (float(r["gpu_h_cells"]) + float(r["gpu_h_build"])) * OVERHEAD
    if suite == "tune-q":
        h = TUNE_Q_MEASURED[ds]
    if suite.startswith(("nsweep", "seeds-")):
        h = (NSWEEP_CELL_S[(suite, ds) if algos is None else (suite, ds, *algos)] / 3600 + build) * OVERHEAD
    if suite == "tune-timed" and ds in TUNE_T_MEASURED:
        h = TUNE_T_MEASURED[ds]
    return cells, h


def next_id(out, gpu):
    used = {int(x["id"].split("-")[1]) for x in out if x["gpu"] == gpu}
    n = 1
    while n in used or n in SKIP.get(gpu, ()):
        n += 1
    return n


def main():
    inv = list(csv.DictReader(open(sys.argv[1])))
    out, totals, prev = [], collections.Counter(), {}
    for n, (gpu, fam, suite, dss, algos, backend, flags, note) in enumerate(LEGS):
        (ds,) = dss
        dim = 256 if suite.startswith("nsweep") else DIM[ds]
        cells, h = price(inv, suite, ds, algos, backend)
        if fam == "M1":
            cells, h = 0, 0.5
        pod, idx = GPUS[gpu]
        lid = f"{gpu}-{next_id(out, gpu):02d}-{suite}-{ds}" + (f"-{'-'.join(SHORT[a] for a in algos)}" if algos else "") + (f"-{backend}" if backend else "")
        res = f"/scratch/final/gpu{idx}" if pod == "a100-x2-d" else "/scratch/final/gpu0"
        if algos is None:
            cmd = f"{BENCH} campaign --suite {suite} --dataset {ds} --dim {dim} {flags} --out {res} --resume"
        else:
            cmd = (f"{BENCH} run --dataset {ds} --dim {dim} --suite {suite} " + " ".join(f"--algo {a}" for a in algos)
                   + (f" --backend {backend}" if backend else "") + f" {flags} --out {res} --resume")  # fmt: skip
        after = [f"stage-{pod}"] + ([prev[gpu]] if gpu in prev else [])
        if gpu == "d1" and gpu not in prev:
            after.append(out[0]["id"])  # M1 (d0's first leg) passes before d1 times anything
        out.append({"id": lid, "gpu": gpu, "pod": pod, "index": idx, "worker": WORKER[gpu], "res": res, "dim": dim, "family": fam, "suite": suite, "dataset": ds,
                    "after": sorted(set(after)), "cells": cells, "gpu_h": round(h, 2),
                    "official": "official" in str(algos) or (algos and "silvertorch" in algos and suite in ("filter", "synth")) or suite in ("h2h", "c7-scorepath", "codesign", "codesign-laion30m", "bloomwidth", "bloomwidth-timed", "deep", "nsweep-codesign", "seeds-c7"),
                    "command": " ".join(cmd.split()), "upload": f"campaign-final/{ds}-{suite}" + (f"-{gpu}" if algos else ""), "note": note})  # fmt: skip
        prev[gpu] = lid
        totals[gpu] += h
    for bench, (tuned, h) in CEILING.items():
        tune = [x["id"] for x in out if x["suite"] == "tune-timed" and x["dataset"] in tuned]
        lid = f"d1-{next_id(out, 'd1'):02d}-ceiling-{bench}"
        res = f"/scratch/final/gpu1/ceiling-int8-{bench}"
        out.append({"id": lid, "gpu": "d1", "pod": "a100-x2-d", "index": 1, "worker": WORKER["d1"], "res": res, "dim": 0, "family": "C6", "suite": "ceiling-script",
                    "dataset": bench, "after": sorted({"stage-a100-x2-d", prev["d1"], *tune}), "cells": 0, "gpu_h": h,
                    "official": False, "command": f"bash docs/artifacts/campaign-final/ceiling.sh {bench} {res}",
                    "upload": f"campaign-final/ceiling-int8-{bench}",
                    "note": "quality only; not grid cells" + ("" if tuned else "; no LAION tune leg (its point is the calibration's 16384 / 4096)")})  # fmt: skip
        prev["d1"] = lid
        totals["d1"] += h
    seen = collections.Counter()
    for lg in out:
        key = (lg["gpu"], lg["suite"], lg["dataset"])
        seen[key] += lg["cells"]
        lg["records"] = seen[key]  # the suite file holds this leg's and the GPU's earlier legs' records
    print(render(out, totals))


def upload_command(lg):
    """The brief's corrected form (CORRECTION, controller 15:20Z): `bench upload` globs
    <results>/<suite>/*.jsonl, so this leg's suite files are hardlinked into a fresh <T>/<suite>/
    and <T> is uploaded; MANIFEST n_records must equal expect.records. A ceiling leg's directory
    holds JSON artifacts, no records: it is uploaded as is (n_records 0)."""
    if lg["suite"] == "ceiling-script":
        return f"{BENCH} upload --results {lg['res']} --path-in-repo {lg['upload']} --verify"
    pattern = f"{lg['res']}/{lg['suite']}/{lg['dataset']}-d{lg['dim']}*"
    return (
        f"T=$(mktemp -d /scratch/final/upload.XXXXXX) && mkdir -p $T/{lg['suite']} && "
        f"cp -al {pattern} $T/{lg['suite']}/ && "
        f"{BENCH} upload --results $T --path-in-repo {lg['upload']} --verify && rm -rf $T"
    )


def render(legs, totals):
    y = ["# The final pass (F-REPRO): pods, datasets, families, legs per GPU, gates. Generated by plan.py from the sizing",
         "# inventory; code_version = FINAL_TAG until the library freezes (controller, the final-pass directive). NOT CITABLE",
         "# until D1-G. Every leg runs from evaluation/ of the FINAL_TAG worktree with its env (CUDA_VISIBLE_DEVICES, its own",
         "# inductor dir, VENV = the leg venv; commands call $VENV/bin/python directly, since `uv run` masks bench's exit code),",
         "# cores pinned NUMA-local; then its upload_command (records the manifest sha; MANIFEST n_records must equal",
         "# expect.records: the suite file also holds the GPU's earlier legs of that suite and dataset) and a hub-index row.",
         "recipe: 1",
         "tag: FINAL_TAG                          # git tag campaign-final; `git rev-parse FINAL_TAG:retrieve/src/retrieve` = library_tree",
         "library_tree: FINAL_TAG_LIBRARY_TREE",
         "extension_tag: campaign-final          # the N-sweep and seeds-0-2 legs' tree: tagged after dev/final-nsweep merges, same library (e16512f5) as FINAL_TAG",
         "campaign_yaml: evaluation/campaign.yaml   # pinned to the one code_version (quality + perf) before the first leg",
         "hub: {repo: pinkmeme/eval-results, prefix: campaign-final/, aggregate: campaign-final/results.parquet}",
         "image: {torch: 2.10.0+cu128, triton: 3.6.0, cuda: '12.8', python: '3.11'}",
         f"official: {{build: scripts/build_official_o3.sh, venv: {OFFICIAL_VENV}, flags: '-O3 -Xcompiler -O3'}}   # check env.official_build.nvcc_append_flags per record",
         "official_so_sha256:                     # per pod: each build differs (and so do the shipped wheels); filled from the records' env.official_build",
         "  a100-x1-eval: 805ab73e…                # verified -O3 (controller 2026-10-13)",
         "  a100-x2-d: FILL_FROM_RECORDS            # d-run's rebuild",
         "  a100-x1-b: FILL_FROM_RECORDS",
         "worktree: /scratch/wt/final           # git worktree at FINAL_TAG on every pod; venv /venvs/final (uv sync --extra official --all-packages); the extension legs (tree: campaign-final) run from a worktree at that tag (same library, its suites.yaml / configs)",
         "",
         "pods:",
         "  - {name: a100-x2-d, host: 38f5e1, gpus: [0, 1], disk: /data + /scratch on the 400 GB container disk}",
         "  - {name: a100-x1-b, host: 1ffe3d, gpus: [0]}",
         "  - {name: a100-x1-eval, host: e75980, gpus: [0]}",
         "",
         "datasets:                              # oracle fingerprints: `bench check` + `bench oracle` at staging, written back here",
         "  goodreads: {dim: 128, inputs: sasrec-ssm-logq-d128, source: {hub: pinkmeme/eval-goodreads-work-id, revision: 4714e9b5fa75327eb7eb952cb5a259f8879cb6fe}}",
         "  arxiv: {dim: 128, inputs: content_d128, source: {hub: pinkmeme/eval-arxiv-papers, revision: 354e6937eec39103773b2a8c538d7d28a3cc0ecc}}",
         "  yfcc10m: {dim: 192, inputs: content_d192, source: {hub: pinkmeme/eval-yfcc10m, revision: 88e4d89a2f9571ab15fbb8deffbb06bf4ed05441}}",
         "  pubmed: {dim: 768, inputs: content_d768, source: {copy_from: 'a100-x1-b:/data/pubmed', sha256_manifest: FILL_AT_STAGING}}   # not on the Hub",
         "  laion30m: {dim: 256, inputs: content_d256, source: {copy_from: 'a100-x2-d:/data/laion30m', sha256_manifest: FILL_AT_STAGING}}   # not on the Hub",
         "  laion1m: {dim: 256, inputs: content_d256, source: {built: 'eval-data subset laion30m laion1m --n-items 1000000 --seed 20261013', on: a100-x2-d, manifest_sha256: 00b61268f187e1ef206f5c2bdb4ed73c814c2a1bde24d032cddf6239a09ab191, rows_sha256: 48164da6e98c27d0474e1d5e9feb697cbdcb7a0a34bceab1e8a9c944f8087cb7}}",
         "  laion3m: {dim: 256, inputs: content_d256, source: {built: 'eval-data subset laion30m laion3m --n-items 3000000 --seed 20261013', on: a100-x2-d, manifest_sha256: 277a1c15acd7a5a263723c6621b7cddf122db069ff356144b79e771f0aec8235, rows_sha256: 6730af14e11ba92200cbc562268c4c90509ce16dfcc10fbad0667d37ddb6fc76}}",
         "  laion10m: {dim: 256, inputs: content_d256, source: {built: 'eval-data subset laion30m laion10m --n-items 10000000 --seed 20261013', on: a100-x2-d, manifest_sha256: 2fccfc39578a6ecadcbdbcd275be9a028f0f18a84836a2871bc85fc4ec237cde, rows_sha256: 9a0c7374a603dd12e71b9481cb6c24bbe1106b146194e17ac9b8121ff4d256bd}}",
         "  laion1m-synth / laion3m-synth / laion10m-synth: {base: 'laion1m / 3m / 10m', attrs: \"the parent's synth rows (data/laionNm/item_attrs_synth.pt)\"}",
         "  goodreads-synth: {base: goodreads, attrs: 'eval-data synth-filter', sha256: FILL_AT_STAGING}",
         "  arxiv-synth: {base: arxiv, attrs: 'eval-data synth-filter', sha256: FILL_AT_STAGING}",
         "  arxiv-corr-synth: {base: arxiv, attrs: 'eval-data synth-filter --correlated (seed 20261008)', sha256: {item_attrs_corr.pt: 4025c823…, query_attrs_corr.pt: bef4e61e…, synth_corr.json: 036aca8a…}}",
         "  yfcc10m-synth: {base: yfcc10m, attrs: 'eval-data synth-filter', sha256: FILL_AT_STAGING}",
         "  laion30m-synth: {base: laion30m, attrs: 'eval-data synth-filter', sha256: FILL_AT_STAGING}",
         "",
         "stage:                                 # per pod, before its first leg (id stage-<pod>; every leg of the pod is after it)",
         "  a100-x2-d: {owner: d-run, steps: [copy pubmed from a100-x1-b, bench check + bench oracle for every dataset pod d runs]}",
         "  a100-x1-b: {owner: v-pubmed, steps: [fetch yfcc10m, build yfcc10m-synth / goodreads-synth attrs, bench check + oracle (goodreads, goodreads-synth, yfcc10m, yfcc10m-synth, pubmed)]}",
         "  a100-x1-eval: {owner: v-pod1-run, steps: [replace pod 1's PubMed with the canonical copy from a100-x1-b, laion30m copied from a100-x2-d (steward), bench check + oracle]}",
         "  a100-x1-b-seeds: {owner: v-pubmed, before: b seeds legs, steps: ['laion30m copied from a100-x2-d (steward; sha256 manifest checked)', arxiv-synth attrs present, bench check + oracle (arxiv-synth d128, yfcc10m-synth, laion30m-synth, laion30m bloom)]}",
         "  a100-x1-eval-nsweep: {owner: v-pod1-run, before: p1 N-sweep legs, steps: ['laion1m / laion3m / laion10m copied from a100-x2-d:/data (steward; MANIFEST.sha256 checked on arrival)', 'arxiv-papers content (d256) present', bench check + oracle per N-sweep dataset]}",
         "",
         "families:                              # one box (host) each; the T1 rows they feed",
         "  C1: {box: a100-x2-d, feeds: [T1.C1, F1, c1-across-n, c1-real, exact refs of C6 / QPS bands]}",
         "  C2: {box: [a100-x2-d (V3 vs V2), a100-x1-b (v3bits, self-contained)], feeds: [T1.C2]}",
         "  C3: {box: [a100-x1-eval (c3, c3-real), a100-x2-d (synth V2 torch vs triton)], feeds: [T1.C3]}",
         "  C4: {box: a100-x1-eval, feeds: [T1.C4, F4a]}",
         "  C5: {box: a100-x1-eval, feeds: [T1.C5, F4b]}",
         "  C6: {box: a100-x2-d, feeds: [T1.C6, F2, F3, QPS bands, local pass rate]}",
         "  C7: {box: a100-x2-d, feeds: [T1.C7, T3]}",
         "  T2: {box: a100-x2-d, feeds: [T2]}",
         "  tune: {box: [a100-x1-b (pubmed, yfcc10m, goodreads benches), a100-x1-eval (arxiv bench)], feeds: [n_lists / n95 slots, tune fronts]}",
         "",
         "legs:"]  # fmt: skip
    for lg in legs:
        y.append(f"  - id: {lg['id']}")
        for k in ("pod", "index", "worker", "family", "suite", "dataset"):
            y.append(f"    {k}: {lg[k]}")
        ext = lg["suite"].startswith(("nsweep", "seeds-"))
        y.append(f"    tree: {'campaign-final' if ext else 'FINAL_TAG'}")
        y.append(f"    after: [{', '.join(lg['after'])}]")
        venv = OFFICIAL_VENV if lg["official"] else "/venvs/final"
        y.append(f"    env: {{CUDA_VISIBLE_DEVICES: '{lg['index']}', TORCHINDUCTOR_CACHE_DIR: /scratch/inductor/final-gpu{lg['index']}, VENV: {venv}, UV_PROJECT_ENVIRONMENT: {venv}}}")
        y.append(f"    command: {json.dumps(lg['command'])}")
        y.append(f"    upload: {lg['upload']}")
        y.append(f"    upload_command: {json.dumps(upload_command(lg))}")
        y.append(f"    expect: {{cells: {lg['cells']}, records: {lg['records']}, gpu_h: {lg['gpu_h']}}}")
        if lg["note"]:
            y.append(f"    note: {json.dumps(lg['note'])}")
    y += ["", "totals_gpu_h:                           # per GPU, incl. the 1.45 process overhead; wall-clock = the largest"]
    y += [f"  {g}: {round(h, 1)}" for g, h in sorted(totals.items())]
    y += ["", "gates:                                 # D1-G over campaign-final/ after the last leg",
          "  - eager-vs-graph ids from the stored hashes",
          "  - a byte-identical quality subset per leg (--force into a scratch tree)",
          "  - median_ms(bs=16) < 16 x median_ms(bs=1) (real-filter V2 / V3 exceed it by the per-query spread: c1-real)",
          "  - bench report --manifest over campaign-final/; results.parquet + its manifest sha in hub-index"]  # fmt: skip
    return "\n".join(y)


if __name__ == "__main__":
    main()
