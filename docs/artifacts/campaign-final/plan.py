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

OVERHEAD = 1.45  # median wall / summed cell time over 20 legs (sizing.py, overhead.csv)
DIM = {"goodreads": 128, "goodreads-synth": 128, "arxiv": 128, "arxiv-synth": 128, "arxiv-corr-synth": 128,
       "yfcc10m": 192, "yfcc10m-synth": 192, "pubmed": 768, "laion30m": 256, "laion30m-synth": 256}  # fmt: skip
# quality-only tune legs are build-bound, which elapsed_s misses: the measured legs (validation, night-queue item 8)
TUNE_Q_MEASURED = {"pubmed": 2.19, "yfcc10m": 0.4, "yfcc10m-synth": 0.32, "goodreads": 0.2, "goodreads-synth": 0.22,
                   "arxiv": 0.25, "arxiv-synth": 0.22, "arxiv-corr-synth": 0.21}  # fmt: skip
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
    *[("d1", "C6", "deep", [d], None, None, "", "") for d in REAL],
    *[("d1", "C6+C7", "synth", [d], ["silvertorch"], None, "--interleave", "") for d in ["goodreads-synth", "arxiv-synth", "arxiv-corr-synth", "yfcc10m-synth"]],
    ("d1", "C6", "laion30m", ["laion30m"], ["silvertorch"], None, "", ""),
    ("d1", "C6", "laion30m-bs1", ["laion30m"], None, None, "", ""),
    ("d1", "C1+C6", "laion30m-synth", ["laion30m-synth"], None, None, "--interleave", "V1 / V2 (C1) and SilverTorch (C6) at 30 M synth"),
    *[("d1", "C7", "h2h", [d], None, None, "--interleave --profile", "") for d in ["goodreads", "arxiv"]],
    ("d1", "C7", "c7-scorepath", ["laion30m"], None, None, "--interleave --profile --mode eager", ""),
    *[("d1", "C2", "filter", [d], ["linr_v3"], None, "", "V3 / V2 against d0's V2, same host") for d in REAL],
    *[("d1", "C2", "synth", [d], ["linr_v3"], None, "", "") for d in ["goodreads-synth", "arxiv-corr-synth"]],
    *[("b", "tune", s, [d], None, None, "--skip-perf" if s == "tune-q" else "", "") for s in ("tune-q", "tune-timed") for d in ["pubmed", "yfcc10m", "yfcc10m-synth"]],
    *[("b", "C2", "v3bits", [d], None, None, "", "k_bits variants against each other") for d in ["goodreads-synth", "goodreads", "pubmed"]],
    *[("p1", "C3", "c3", [d], None, None, "", "") for d in ["goodreads-synth", "arxiv-synth", "yfcc10m-synth", "laion30m-synth"]],
    *[("p1", "C3", "c3-real", [d], None, None, "", "") for d in REAL],
    *[("p1", "C5", "codesign", [d], None, None, "--interleave", "") for d in ["arxiv", "goodreads"]],
    ("p1", "C5", "codesign-pubmed", ["pubmed"], None, None, "--interleave", ""),
    ("p1", "C5", "codesign-laion30m", ["laion30m"], None, None, "--interleave --mode eager", "needs LAION on pod 1, else runs on d1"),
    *[("p1", "C4", s, [d], None, None, "", "") for s in ("bloomwidth", "bloomwidth-timed") for d in ["goodreads", "arxiv", "pubmed"]],
    *[("p1", "tune", s, [d], None, None, "--skip-perf" if s == "tune-q" else "", "") for s in ("tune-q", "tune-timed")
      for d in ["goodreads", "goodreads-synth", "arxiv", "arxiv-synth", "arxiv-corr-synth"]],
]  # fmt: skip
SHORT = {"linr_v1_filter_mask": "v1", "linr_v2": "v2", "linr_v3": "v3", "silvertorch": "st", "postfilter": "pf"}
GPUS = {"d0": ("a100-x2-d", 0), "d1": ("a100-x2-d", 1), "b": ("a100-x1-b", 0), "p1": ("a100-x1-eval", 0)}


def price(inv, suite, ds, algos, backend):
    cells, h = 0, 0.0
    for r in inv:
        if r["source"] != "suites.yaml" or r["suite"] != suite or r["dataset"] != ds:
            continue
        algo, be = r["arm"].split()[0].split("/")
        if algos is not None and algo not in algos or backend and be != backend:
            continue
        cells += int(r["cells"])
        h += float(r["gpu_h_cells"]) * OVERHEAD
    if suite == "tune-q":
        h = TUNE_Q_MEASURED[ds]
    return cells, h


def main():
    inv = list(csv.DictReader(open(sys.argv[1])))
    out, totals, prev = [], collections.Counter(), {}
    for n, (gpu, fam, suite, dss, algos, backend, flags, note) in enumerate(LEGS):
        (ds,) = dss
        cells, h = price(inv, suite, ds, algos, backend)
        if fam == "M1":
            cells, h = 0, 0.5
        pod, idx = GPUS[gpu]
        lid = f"{gpu}-{len([x for x in out if x['gpu'] == gpu]) + 1:02d}-{suite}-{ds}" + (f"-{'-'.join(SHORT[a] for a in algos)}" if algos else "") + (f"-{backend}" if backend else "")
        res = f"/scratch/final/gpu{idx}" if pod == "a100-x2-d" else "/scratch/final/gpu0"
        if algos is None:
            cmd = f"uv run bench campaign --suite {suite} --dataset {ds} --dim {DIM[ds]} {flags} --out {res} --resume"
        else:
            cmd = (f"uv run bench run --dataset {ds} --dim {DIM[ds]} --suite {suite} " + " ".join(f"--algo {a}" for a in algos)
                   + (f" --backend {backend}" if backend else "") + f" {flags} --out {res} --resume")  # fmt: skip
        after = [f"stage-{pod}"] + ([prev[gpu]] if gpu in prev else [])
        if gpu == "d1" and gpu not in prev:
            after.append(out[0]["id"])  # M1 (d0's first leg) passes before d1 times anything
        out.append({"id": lid, "gpu": gpu, "pod": pod, "index": idx, "family": fam, "suite": suite, "dataset": ds,
                    "after": sorted(set(after)), "cells": cells, "gpu_h": round(h, 2),
                    "official": "official" in str(algos) or (algos and "silvertorch" in algos and suite in ("filter", "synth")) or suite in ("h2h", "c7-scorepath", "codesign", "codesign-laion30m", "bloomwidth", "bloomwidth-timed"),
                    "command": " ".join(cmd.split()), "upload": f"campaign-final/{ds}-{suite}" + (f"-{gpu}" if algos else ""), "note": note})  # fmt: skip
        prev[gpu] = lid
        totals[gpu] += h
    print(render(out, totals))


def render(legs, totals):
    y = ["# The final pass (F-REPRO): pods, datasets, families, legs per GPU, gates. Generated by plan.py from the sizing",
         "# inventory; code_version = FINAL_TAG until the library freezes (controller, the final-pass directive). NOT CITABLE",
         "# until D1-G. Every leg: CUDA_VISIBLE_DEVICES=<index>, its own inductor dir, cores pinned NUMA-local, then",
         "# `bench upload --results <out>/<suite> --path-in-repo <upload> --verify` and a hub-index row.",
         "recipe: 1",
         "tag: FINAL_TAG                          # git tag campaign-final; `git rev-parse FINAL_TAG:retrieve/src/retrieve` = library_tree",
         "library_tree: FINAL_TAG_LIBRARY_TREE",
         "campaign_yaml: evaluation/campaign.yaml   # pinned to the one code_version (quality + perf) before the first leg",
         "hub: {repo: pinkmeme/eval-results, prefix: campaign-final/, aggregate: campaign-final/results.parquet}",
         "image: {torch: 2.10.0+cu128, triton: 3.6.0, cuda: '12.8', python: '3.11'}",
         f"official: {{build: scripts/build_official_o3.sh, venv: {OFFICIAL_VENV}, flags: '-O3 -Xcompiler -O3', so_sha256: FINAL_O3_SO_SHA}}",
         "worktree: /scratch/wt/final           # git worktree at FINAL_TAG on every pod; venv /venvs/final (uv sync --extra official --all-packages)",
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
         "  goodreads-synth: {base: goodreads, attrs: 'eval-data synth-filter', sha256: FILL_AT_STAGING}",
         "  arxiv-synth: {base: arxiv, attrs: 'eval-data synth-filter', sha256: FILL_AT_STAGING}",
         "  arxiv-corr-synth: {base: arxiv, attrs: 'eval-data synth-filter --correlated (seed 20261008)', sha256: {item_attrs_corr.pt: 4025c823…, query_attrs_corr.pt: bef4e61e…, synth_corr.json: 036aca8a…}}",
         "  yfcc10m-synth: {base: yfcc10m, attrs: 'eval-data synth-filter', sha256: FILL_AT_STAGING}",
         "  laion30m-synth: {base: laion30m, attrs: 'eval-data synth-filter', sha256: FILL_AT_STAGING}",
         "",
         "stage:                                 # per pod, before its first leg (id stage-<pod>)",
         "  a100-x2-d: [copy pubmed from a100-x1-b, bench check + bench oracle for every dataset it runs]",
         "  a100-x1-b: [fetch yfcc10m, build yfcc10m-synth / goodreads-synth attrs, bench check + oracle]",
         "  a100-x1-eval: [copy pubmed from a100-x1-b, copy laion30m from a100-x2-d (if disk allows; else codesign-laion30m and c3 laion30m-synth move to d1), bench check + oracle]",
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
         "  tune: {box: [a100-x1-b (pubmed, yfcc10m, yfcc10m-synth), a100-x1-eval (goodreads, arxiv benches)], feeds: [n_lists / n95 slots, tune fronts]}",
         "",
         "legs:"]  # fmt: skip
    for lg in legs:
        y.append(f"  - id: {lg['id']}")
        for k in ("pod", "index", "family", "suite", "dataset"):
            y.append(f"    {k}: {lg[k]}")
        y.append(f"    after: [{', '.join(lg['after'])}]")
        venv = OFFICIAL_VENV if lg["official"] else "/venvs/final"
        y.append(f"    env: {{CUDA_VISIBLE_DEVICES: '{lg['index']}', TORCHINDUCTOR_CACHE_DIR: /scratch/inductor/final-gpu{lg['index']}, UV_PROJECT_ENVIRONMENT: {venv}}}")
        y.append(f"    command: {json.dumps(lg['command'])}")
        y.append(f"    upload: {lg['upload']}")
        y.append(f"    expect: {{cells: {lg['cells']}, gpu_h: {lg['gpu_h']}}}")
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
