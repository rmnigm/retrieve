---
title: decisions
created: 2026-09-26
updated: 2026-09-26
type: summary
tags: [decisions]
sources: [pyproject.toml, evaluation/config/suites.yaml, retrieve/src/retrieve/, evaluation/bench/]
---

# Standing decisions and constraints

Decisions in force, with the reason each was taken. They were made by the
user or measured into place and are not reopened without the user. How the
code implements them is in the [system pages](index.md#system); the open
work is the [roadmap](roadmap.md). The two process contracts,
[agent orchestration](contracts/agent-orchestration.md) and
[coding guidelines](contracts/coding-guidelines.md), carry their own
decisions.

## Goal and scheduling

- **The goal is a reproducibility paper** on SilverTorch (Meta) and LiNR
  (LinkedIn): our Triton reimplementation from the papers' text, Meta's
  official kernels as the reference, one correct harness, public datasets
  up to the papers' scale.
- **No dates.** Every roadmap step is done, in order; nothing is optional
  or conditional on a deadline. The GPU is the bottleneck, so GPU steps
  are ordered first and CPU steps run beside them.
- **Nothing is citable until its gate passed** (CLAUDE.md rule 2).
  [validation.md](validation.md) says which gates have passed.

## Library

- **Meta's `meta-recsys/silvertorch` ops are the reference backend**,
  `backend="official"`, installed by the `official` extra and pinned to one
  commit (`21aa35e`, see [pyproject.toml](../pyproject.toml)). Bumping the
  pin is a deliberate change that reruns the official parity gate.
- **Triton is our implementation and gets the kernel effort.** The
  hand-written CUDA C++ and CuTe DSL SilverTorch backends were deleted once
  the official backend's parity gate was green; the git tag
  `cuda-cute-backends-final` holds them.
- **Meta's shape**: `retrieve.modules` (the `nn.Module`s and builders) and
  `retrieve.ops` (registered kernels, one namespace per backend: `triton`,
  `reference`, `official`), plus `retrieve.indexing` and
  `retrieve.functional`. Meta's modules and ops are imported and wired in,
  never copied. See [architecture](system/architecture.md).
- **LiNR V1-V4 are library modules**; the harness keeps only a name to
  class table.
- **Op names, module names, buffer names and op schemas are stable.** A
  state dict written by an earlier release loads into the current modules.
- **k-means++ is opt-in** (`kmeans_init="random"` is the default) until
  campaign numbers say otherwise. Seeding cost at 3M × 128 with 8,192
  lists is 9.5 s for k-means++ against 0.8 s random.
- **Stream compaction is deterministic**: survivors come out in ascending
  item order on both backends, and a rerun is byte-identical on the
  `[:counts]` prefix, the only part a kernel writes
  ([kernels](system/kernels.md)). Chosen over a wider quality tolerance
  because two golden cells could not reproduce themselves.
- **LiNR's exact scorers store items fp16 and return fp32 scores**
  (`PostfilterKNN`, `PrefilterKNN`, every backend): fp16 storage is the
  LiNR paper's, and keeps the item table at `N × D × 2` bytes (PubMed 10M
  × 768: 14.3 GiB, where an fp32 table is 28.6 GiB on top of the
  harness's own fp32 copy). fp32 scores are what the exact-algorithm gate
  needs: fp16 scores gave `recall_oracle@1000` 0.956 on YFCC-10M, fp32
  scores 0.993 (gate 0.99), an fp32 table 1.0. Cost: the `[B, N]` score
  buffer doubles (+610 MiB at B=16 over 10M items). An fp32 table is the
  next step if a dataset's storage rounding alone breaks the gate
  ([kernels](system/kernels.md#score-conventions),
  [artifact](artifacts/l1-l2/README.md)).
- **Int8 quantization uses one global scale**, as the SilverTorch paper
  does.
- **Bloom hashes are keyed on `(clause_idx, value)`**, a deviation from
  the paper that stops equal values in different clauses from colliding
  ([filtering](system/filtering.md#bloom-hash-keys-clause_idx-value)).
- **No library change while campaign records accumulate.** The records'
  resume key includes the library tree hash, so any edit under
  `retrieve/src/retrieve/` invalidates the campaign. The two scorer
  improvements TF-1 (transposed bloom index in Triton) and TF-9 (CSR or
  capped-pad probe layout) wait for the campaign to end.

## Harness

- **Three packages, one dependency direction**: `bench` → `training` →
  `eval_datasets`, enforced by `tests/test_dependency_direction.py`. The
  library retrieves; the harness measures.
- **One backend per algorithm in the campaign grid.** `triton` everywhere,
  because it is the fastest arm on every algorithm measured (eager median
  torch/triton 4.16× on V1, 7.27× on V2, 10.19× on V3). `silvertorch`
  runs `[triton, official]`, because that comparison is the paper. The
  `torch` floor is not swept again.
- **`linr_v4` is out of the grid.** It is ours, not LiNR's: the paper
  defines V1-V3.
- **No unfiltered `quality` suite.** Its datasets were Yambda's, which
  left the study; unfiltered cells return with the new datasets (roadmap
  E5). The report's no-filter table emits a placeholder until then.
- **Modes: `eager` everywhere, `graph` on `triton`.** Graph-only was
  rejected: the official ops cannot be captured, and eager is the papers'
  comparable mode. A run with a narrowed mode set records
  `status: partial`, which the report treats as not citable; whether a
  deliberate, recorded narrowing should read differently is open (see the
  roadmap).
- **Clocks cannot be locked** in the container. Records carry the SM clock
  sampled under load after every timing window and an `unstable` flag
  (window spread over 5 %). A batch-size-1 comparison narrower than about
  21 % is noise.
- **Timed official forwards run with `OfficialConfig(cache_plans=False)`**,
  so Meta's plan cache does not flatter repeated identical queries.
- **Results storage** (user, 2026-09-26): no results in git. A run appends
  JSONL to a local, gitignored results tree (one `write` + `fsync` per cell,
  and resume reads it back without the network); a finished leg is
  aggregated into Parquet (`results.parquet`, one row per perf entry — the
  table every report reads) and published with its JSONL and samples to the
  private Hub repo `pinkmeme/eval-results` (`bench upload`, `bench fetch`
  back). Raw outputs behind a documented finding go to the same repo under
  `artifacts/<plan>/`; what is neither cited nor needed for re-derivation is
  dropped. Git keeps code, prose, gate reports, the golden cells (a test
  fixture) and the sha256 of each Hub manifest. The `.git` history still
  holds the removed files; rewriting it is a separate decision.

## Datasets

- **The study's datasets**: arXiv and Goodreads (current), YFCC-10M,
  PubMed + MedCPT, Semantic Scholar SPECTER2 (OpenAlex if the API key is
  refused) and KuaiRand-27K. Each has real filters and an open or local
  query encoder.
- **Dropped**: Amazon Reviews 2023; Yambda-full and Cohere Wikipedia (scale
  without meaningful filters, closed query encoder). Existing Yambda
  unfiltered runs and checkpoints stay as they are.
- **No PCA.** Every dataset runs at its encoder's native width (YFCC 192,
  PubMed 768); each dataset contributes one width, and the dim ablation is
  dropped.
- **PubMed runs as a 10M slice**, not the full ~36M: the harness holds
  items fp32 on the device (110 GB for the full catalog), while at 10M the
  items, codes and attributes fit in about 40 GB, and 10M matches the
  papers' pool and YFCC's size.
- **YFCC runs clause filters only**, no bloom, so bloom false positives
  cannot spoil the cross-check against the shipped ground truth.

## Environment

- **One GPU box, serialized** (CLAUDE.md rule 1). One GPU job at a time;
  each concurrent GPU job needs its own `TORCHINDUCTOR_CACHE_DIR`.
- **Disks**: the repository on the persistent network volume, everything
  large on the ephemeral local disk ([storage](system/storage.md)). The
  pod image's defaults put data and the Hub cache on `/workspace`
  instead; storage.md marks this contested.
- **`ncu` is blocked**; kernel attribution uses `torch.profiler`.
