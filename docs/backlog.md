---
title: backlog
created: 2026-10-07
updated: 2026-10-08
type: summary
tags: [roadmap]
sources: [docs/roadmap.md, docs/validation.md, evaluation/bench/, evaluation/training/]
---

# Backlog: known, not queued

Work that is known but not in the [roadmap](roadmap.md)'s queue: defects
that do not block a remaining run, measurements nobody has scheduled,
post-paper work and background reading. Nothing here is started without
the user moving it onto the roadmap.

## Baselines outside the study

- **Faiss, HNSW, cuBLAS and cuVS** (formerly roadmap D2): Faiss-GPU and
  Faiss-CPU IVF-Flat, HNSW, a cuBLAS brute-force floor at matched recall;
  cuVS IVF-Flat / IVF-PQ / CAGRA with a bitset prefilter (G13);
  Filtered-DiskANN or ACORN (G14). Struck from the eval plan: the study's
  baseline is generic torch plus the torch-importable arms
  ([decisions](decisions.md#campaign-v2-user-2026-10-08)). Possibly done
  later by a separate agent on a separate VM.

## Datasets not in the study

- **OpenAlex** (dropped 2026-10-08, was roadmap E5): the 10 M × 768 slice
  duplicates PubMed's N, width and domain family, for 28-38 GPU-h plus a
  297 GB restage ([datasets](system/datasets.md#openalex) keeps the
  pipeline). **Option, the first thing added back:** a 30 M scale point at
  low width from the same pipeline — 30 M works, nomic-embed-text-v1.5
  truncated to d128 (Matryoshka) — turning OpenAlex into the scale axis
  (0.8 M, 3 M, 10 M, 30 M). Needs the chunked oracle (campaign-v2 code
  batch) and the item table on the GPU (30 M × 128 fp16 = 7.7 GB). Cost:
  ~9 GPU-h of encoding plus the restage, and a reduced grid (`filter` on 3
  sweeps, `synth` at {0.001, 0.01, 0.1, 1.0}, no `deep`) of ~12-18 GPU-h.
  Only after the campaign's run order is done, and only on the user's
  call.
- **LAION / Re-LAION-5B** above 30 M: the 30 M point is in the study
  (roadmap V-LAION30, [decisions](decisions.md#datasets)); 100 M would
  need fp16 items on the device and the oracle without its fp32 copy
  (Known defects below).

## After the paper

- **Fusing top-k into the probe scorer**: top-k is ~100 µs of ~250 µs
  device time in both implementations
  ([h2h](artifacts/kernel-opt/h2h.md), one run, not validated); beyond
  both papers, so post-paper.
- **TF-3 / TF-4 retune**: TF-3 (retune) and TF-4 (`evict_first`, 0-5 %),
  second-order after TF-9 (probe layout) and TF-1 (transposed bloom
  index). Needs a fresh `bench report` head-to-head against the current
  kernels: the kernel-opt gate checked only the bloom kernel-only ratio,
  and the end-to-end comparison in
  [validation](validation.md#official-against-our-triton-reimplementation-citable-contested)
  reflects the code before TF-1 / TF-9.
- **G-b: extended experiments**: a synthetic scale ladder to 240M and 1B
  items (L4, L5), co-design ablation depth, V3 bit width, an extended batch
  grid (G10-G12, G15, G16).
- **G-c: a resource paper about the library.** After F5.
- **G-e: two re-scoped plans** (deferred until after F5 by the user):
  `torch.export` of the composites is small (the kernel side is
  export-clean; three `Tensor | None` forward params in `modules/linr.py`
  remain). The live upsert/delete API (`LiveIndexMixin` on
  `retrieve.modules`) is medium-large: a new subsystem across five module
  classes and both filters.
- **TF-10: official capturability.** File the upstream issue: Meta's
  scorer syncs because `fused_kmean_ann_cuda.cu` never passes the explicit
  output size `faster_repeat_interleave` accepts. A patched build is
  measured only if a reviewer asks, labelled "not the official release".

## Sequential-encoder follow-ups

- **Atomic `_resume.pt` write**: it is written in place, so a crash during
  the ~45 s write destroys the only resume state. A temporary file plus
  `os.replace` doubles its peak on disk.
- **Window val-day clicks** for `train_on_val` users with more than 200 of
  them, as `train.parquet` is windowed
  ([validation](validation.md#gates)).
- **`--resume` of a `train_on_val` run on the GPU** has never run.
- **Launch overhead / CUDA graphs** in the training step.
- **goodreads d64 R@100**: −0.0039 against the bar, the one miss of the
  success rule ([final models](validation.md#final-models-the-e1c-recipe)).
- **Goodreads trainer-input tiebreak**: `cmd_prep` sorts by `(user_id,
  ts)` with no tiebreak, so the order inside timestamp ties changes run to
  run ([validation](validation.md#trainer-inputs-with-timestamps-data-gates-not-citable)).
  A tiebreak moves the output off the Hub copy.
- **yambda val/test row order**: the `uid` joins in `cmd_prep` give a
  fresh row order each run; nothing may align to it by position.

## Defects that block no remaining run

- **A finer manifest `match`** (seed and a `params` subset, or an ordered
  list of code_versions per entry). Phase P found ~241 cells whose old
  records pass the reuse rule cell by cell, but only 36 are reachable
  through entries that cannot name a seed or params. Worth ≈ 3.5 GPU-h of
  perf (mostly YFCC seed 0) plus the quality pass of ~200 arXiv `deep`
  cells whose timing reruns anyway; not built (one more report mechanism
  for ~2 % of the budget). [reuse](artifacts/campaign-v2/README.md#reuse-entries)
- **`scripts/check_doc_links.py` cannot parse `for … in f(...)` in a CLI
  module** (it reads `node.iter.func.value`); cv2-harness-core shaped
  `bench/cli.py`'s `_children` around it instead of editing the checker.
- **`eval-data pubmed plan` trips NCBI's throttle.** Its 16 parallel HEAD +
  Range requests draw HTTP 503 on most of a burst (12 of 18 at 16-way; 1-8
  concurrent pass); `_remote_size` turns the error into size 0 and `plan`
  aborts. Related: `_fetch_one` accepts a GET that ends early when its HEAD
  failed (no length to check), so the MedCPT shards then rest on `verify`'s
  structural check alone. The 2026-10-07 restage worked around it with a
  2-connection size check ([artifact](artifacts/pubmed-restage/README.md)).
- `linr_v2` and `linr_v3` diverge from their golden files (recall@100
  4.5e-4 / 1.7e-5), cause unidentified, predating the kernel-opt pass
  ([validation](validation.md#harness-gates)).
- The quality subset is a 10k prefix of the query file, not a seeded
  sample; safe on the current datasets only because their files are
  shuffled.
- Large `.log` / `.txt` dumps remain under `docs/artifacts/` (two
  `cute-dsl-scorer` `kernel_only-*.txt` at ~182 KB each, an
  `e3-openalex` convert log at 176 KB, a `cute-dsl-scorer` diagnostic at
  135 KB): candidates for the Hub-or-drop treatment.

## Unmeasured

- LiNR V2 after the backend-parity fix: the arxiv and bloom cells, and
  V3 stage 2 (the same kernel).
- What the old harness's quality pass did to `linr_v4`'s batch; needs the
  frozen golden worktree.
- `bloom_compact`'s `block_n` has not been retuned for the two-phase
  compaction shape.

## Background reading

Not scheduled work; candidates for citations or a one-time check.

- A warp-ballot (`__ballot_sync`) candidate-selection pattern used across
  recent GPU-IVF kNN kernels: check whether the probe kernel already does
  something equivalent.
- A tunable-vectorization GPU bloom filter (arXiv 2512.15595) and a
  cuckoo-filter alternative (arXiv 2603.15486): comparisons for the
  transposed bloom-index kernel.
- A bucket-based coalesced-access layout for filtered graph search
  (GRAB-ANNS, arXiv 2604.16402): an alternative to the compact CSR-like
  probe layout.
- Recall-bucketed / Pareto-frontier reporting (the cuVS Bench methodology)
  for `bench report`'s recall/latency tables, instead of point
  comparisons.
- For the paper: Meta's public SilverTorch numbers
  (`github.com/meta-recsys/silvertorch`, an Engineering-at-Meta post) as
  target figures for
  [official-vs-reimplementation](paper/official-vs-reimplementation.md);
  two ANN-benchmark trustworthiness critiques (arXiv 2507.00379, a
  YDB.tech write-up) for
  [provenance-and-disclosure](paper/provenance-and-disclosure.md). No
  public LiNR reproduction exists to compare against.
