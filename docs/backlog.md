---
title: backlog
created: 2026-10-07
updated: 2026-10-07
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
  baseline is generic torch (roadmap D5). Possibly done later by a
  separate agent on a separate VM.

## After the paper

- **TF-3 / TF-4 retune**: TF-3 (retune) and TF-4 (`evict_first`, 0-5 %),
  second-order after TF-9 (probe layout) and TF-1 (transposed bloom
  index). Needs a fresh `bench report` head-to-head against the current
  kernels: the kernel-opt gate checked only the bloom kernel-only ratio,
  and the end-to-end comparison in
  [validation](validation.md#official-against-our-triton-reimplementation-citable-contested)
  reflects the code before TF-1 / TF-9.
- **G-b: extended experiments**: a synthetic scale ladder to 240M and 1B
  items (L4, L5), a controlled pass-rate sweep (the LiNR V1/V2
  crossover), co-design ablation depth, V3 bit width, an extended batch
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

- **`eval-data pubmed plan` trips NCBI's throttle.** Its 16 parallel HEAD +
  Range requests draw HTTP 503 on most of a burst (12 of 18 at 16-way; 1-8
  concurrent pass); `_remote_size` turns the error into size 0 and `plan`
  aborts. Related: `_fetch_one` accepts a GET that ends early when its HEAD
  failed (no length to check), so the MedCPT shards then rest on `verify`'s
  structural check alone. The 2026-10-07 restage worked around it with a
  2-connection size check ([artifact](artifacts/pubmed-restage/README.md)).
- `bench/oracle.py`'s `item_embs.t().contiguous()` holds a second full
  fp32 copy of the item table, so the harness's per-dataset limit at
  native width is about half the device memory divided by `4·D` bytes
  (OpenAlex at 768-d: ~11-12 M items, hence the 10 M slice). A view would
  remove the copy; it needs its own check against the existing oracle
  results and golden cells.
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
