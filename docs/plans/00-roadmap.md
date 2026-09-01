# Plans roadmap — ordered work queue

Read this first. It orders the other documents in [docs/plans/](.) and
records what is actually done.

**Conventions used here.** A plan lives in this directory while it is
either a live instruction set or a record of intent whose validation
hasn't signed off. Once neither is true it moves to
[archive/](archive/). System behaviour is documented in
[../system/](../system/), never here — if you want to know how something
*works*, those are the maintained references; a plan only tells you why
it was built that way.

---

## Open work

Everything in this section is unfinished. Everything below it is history.

### 1. GPU validation of the refactor track — **blocking**

Branch `refactor/kernels-eval` carries the whole K1–K9 / E1–E8 refactor
plus the new CUDA backend, and **none of it has run on a GPU**. Two
ordered runbooks, both with empty sign-off blocks:

- [refactor-validation-handoff.md](refactor-validation-handoff.md) — 8
  steps covering the library + harness refactor, with pass criteria,
  behaviour deltas, and six named fallbacks.
- [cuda-silvertorch-handoff.md](cuda-silvertorch-handoff.md) — build,
  parity, and benchmark runbook for `SilverTorch(backend="cuda")`.

Until these pass, no result produced on this branch is trustworthy and
the two implemented plan docs stay in place.

### 2. Publication blocker — goodreads oracle rerun

Goodreads d128 + d256 filter sweeps must be re-run against a fresh
oracle before their numbers can be cited. The *cause* is fixed (E6: the
oracle cache is now keyed by a content fingerprint, so same-shape stale
caches recompute themselves), but the affected runs predate the fix.
Do this after item 1 passes; the first run rebuilds each sweep's oracle
once.

### 3. Deferred kernel optimizations

Stages 1+2 took the big levers. Two remain, no committed timeline:

1. Hardware popcount in `oporp_1bit_match_topk` (PTX dump → maybe swap
   SWAR for `popc.b64`).
2. Allocator hygiene in `oporp_1bit_match_topk` /
   `fused_masked_knn_topk` (drop the `torch.full(-inf)` pre-fill;
   replace `cat`-to-pad with pre-allocate-and-slice).

### 4. Thesis results expansion

> **Note on references.** This section was written against
> `docs/thesis/*.md` chapter drafts and `docs/thesis/results-data/` CSVs
> that no longer exist — the thesis moved to LaTeX, and `docs/thesis/`
> now holds only `main.tex`, `references.bib`, and `figures/`. The work
> items are still valid; resolve chapter and figure references against
> the LaTeX sources.

**4a — Harness schema extensions** (library untouched). New latency /
memory statistics are additive fields on `PerfStats` in
[measure.py](../../evaluation/retrieval/measure.py); pass-level items
land in [passes.py](../../evaluation/retrieval/passes.py). Making these
one-field changes was E5.2's purpose.

1. `throughput_qps` — removes the rate-statistic ambiguity in the Pareto
   plots.
2. `mean_ms` alongside `median_ms`.
3. `p99_ms` (ideally `p999_ms`) — required for any serving-SLO claim.
4. `build_time_s`, ideally split per phase (encode / quantize / cluster /
   assemble).
5. `topk_ids_jaccard_vs_torch` — a stronger cross-backend parity
   statistic than `recall_abs_diff`.
6. Per-kernel timing + occupancy via Nsight Compute. Heaviest item;
   manual, one-off per kernel.
7. Per-query latency vectors, not just quantiles — enables violins and
   paired-permutation tests without re-running.

Roughly ordered by cost: 1–3 cheap, 4–5 next, 6–7 need dedicated
profiling.

**4b — Measurement gap re-runs** (no schema change, just more cells):

1. yambda-5b-d256 quality config.
2. `SimHashKNN` `k_bits` deep sweep on goodreads-d128.
3. Multi-seed re-run on one representative cell (goodreads-d128, seeds
   0–4).
4. Extended batch-size grid {1, 4, 16, 64, 256, 1024} on arxiv-d128,
   perf-only (quality is batch-size invariant).
5. K=10 quality runs on the existing quality configs — a one-line YAML
   edit.
6. arXiv per-clause selectivity CSV extraction, mirroring the goodreads
   one. CPU-bound.
7. Cross-dataset deep-sweep coverage: only one deep-sweep exists per
   algorithm (`linr_v3` on goodreads-d128, `silvertorch` on
   arxiv-d128), so each figure is locked to a single dataset — an
   obvious "is this dataset-specific?" reviewer question. Add
   `deep_sweep_linr_v3_arxiv_d128.yaml` and
   `deep_sweep_silvertorch_goodreads_d128.yaml`, then re-facet the
   corresponding plots to overlay or grid by dataset.

Item 7 is recommended before submission; the rest unblock specific
figures.

### 5. Deferred feature plans

Both are **not started**, and both have stale anchors — each carries a
banner listing what rotted. Re-scope before executing.

- [torch-export-refactor.md](torch-export-refactor.md) — layer-side
  `mode=` flags / sibling `forward_*` methods so every layer traces
  under `torch.export`. The kernel-side surface is already in shape.
  Revisit when there is a concrete consumer for `.pt2` artifacts.
- [live-update-api.md](live-update-api.md) — upsert / delete for the
  LiNR family + filters via a `LiveIndexMixin`. Independent of the
  kernel work. Its `RetrievalModule` dependency is already satisfied.

### 6. Research catalog

[future-work-and-research.md](future-work-and-research.md) — a vetted
menu, not a queue: engineering items (GPU CI, fused top-k selection,
persistent kernels, RaBitQ / int4, AOTI serving demo) and research
directions with prior-art citations (filter-aware IVF,
selectivity-adaptive planning, MoL re-ranking, GPU NOT-predicates,
streaming freshness, 100M–1B scaling). Includes an impact × cost table.

---

## Done

### CUDA SilverTorch backend — implemented 2026-07-06, **not yet GPU-run**

A second implementation of SilverTorch's Algorithm 1 phases 2+3 in CUDA
C++, selected by `SilverTorch(backend="cuda")`. Where the Triton kernel
fuses a row-wise bloom read into scoring, this follows the paper: a
transposed, cluster-major bloom index evaluated into 1-bit-per-item
masks, then masked `__dp4a` scoring. Scores are bit-identical to the
Triton backend by construction; ids match up to permutation within tied
scores.

**Phase 2 (2026-09-01, also not GPU-run)** closed the three gaps that
first pass left, per
[cuda-silvertorch-phase2.md](cuda-silvertorch-phase2.md). `filter_mode=
"exact"` now runs on cuda too — as a *second phase-2 mask kernel*
(`cps_clause_mask_kernel`, one warp per output word, two ballots) rather
than a third scoring kernel, since the scorer is filter-agnostic and any
filter is a 1-bit-per-item mask. The scorer gained a config-gated
`UNROLL ∈ {1, 2, 4}` knob (items in flight per segment; `1` is the
original loop, and the arithmetic — hence bit-exactness — is identical at
every value). And two static reviews, one reading the kernels and
launchers like a compiler, one fact-checking the toolchain and API
assumptions against upstream docs, were triaged and applied: an explicit
`nvcc`-major check before the JIT build, `ToolchainMissing` split from a
real build failure so a compile error can no longer masquerade as a test
skip, `if constexpr` where a dead ternary arm was doing out-of-range
pointer arithmetic, and the tie-tolerant id gate above.

Design and constraints:
[../system/kernels.md](../system/kernels.md#codesigned_probe_score_cuda--the-cuda-c-backend).
Phase-2 plan: [cuda-silvertorch-phase2.md](cuda-silvertorch-phase2.md).
Validation runbook:
[cuda-silvertorch-handoff.md](cuda-silvertorch-handoff.md).

This also made `Backend` three-valued (`"torch" | "triton" | "cuda"`),
which matters beyond SilverTorch: every other layer dispatches
`triton`-or-torch, so `"cuda"` silently resolves to the torch path
there. See
[../system/architecture.md](../system/architecture.md#backend-dispatch).

### Refactor track — implemented 2026-07-06, pending validation

Two structure-preserving cleanup plans written from a full audit of both
packages. They change no measured numbers and no op schemas. All code
phases plus the `filter=` → `filter_mode=` rename and the K9/E8.2 doc
sweeps are committed on `refactor/kernels-eval`.

- [kernels-layers-design.md](kernels-layers-design.md) — library-side
  dedup and API consistency: fix the broken `tune-kernels` subcommand
  (K1); shared host-wrapper prep/epilogue so validation reaches the
  production path, and the last four `@custom_op` kernels migrated to
  `@triton_op` (K2); shared `@triton.jit` predicate helpers in
  `kernels/common.py` (K3); `masked_topk` plus the `_PackedBitsKNN` base
  that absorbs `SimHashKNN`'s near-copy of `OneBitKNN` (K4); public
  bloom-hash core in `layers/filters/bloom_hash.py` (K5);
  `FilterModule.register_index` kw-only alignment and the minimal
  `RetrievalModule` ABC (K6); the `KernelTuneSpec` registry (K7); new
  tests (K8); doc sweep (K9).
- [evaluation-refactor.md](evaluation-refactor.md) — harness leanness:
  delete the dead `torch_knn` algo, the unreachable CPU-timing path and
  three unused dependencies; quarantine the upload script behind
  `--repo-id`; rename `datasets` → `eval_datasets` (E1); replace the
  deep kwarg threading in `sweep.py` with `SweepContext` /
  `FilterAssets` (E2); typed `RetrievalAlgo` protocol and a declarative
  eligibility table instead of ValueError-as-control-flow (E3);
  `AlgoBase` for the duplicated compile tail (E4); split `bench_tools.py`
  into `measure.py` / `encode.py` / `passes.py` with `PerfStats` /
  `QualityStats` dataclasses (E5); **oracle cache keyed by content
  fingerprint** — the fix for the blocker in Open work item 2 (E6);
  config/loader hygiene (E7); unit tests and the `evaluation.md` rewrite
  (E8).

Both stay in this directory only until validation signs off; archive
them after.

### Stage 1 — Autotune separation (2026-05-22)

Every kernel exposes a `<Name>Config` dataclass + a single curated
`DEFAULT_CONFIG` next to its `@triton.jit` body. Overrides go through a
private `_<name>_impl(..., *, config=None)` so the public op keeps a
fixed schema. Offline tuning via
[tune.py](../../retrieve/src/retrieve/tune.py). `bloom_match` keeps a
hard-coded tile — its per-call width is dictated by `N`. Convention:
[../system/kernels.md → Autotune separation](../system/kernels.md#autotune-separation).

### Stage 2 — `torch.compile` fixes (2026-05-22)

Replaced `@torch._dynamo.disable` on every kernel host wrapper with
proper op registration, so each algo's
`compile(dynamic=True, mode="reduce-overhead")` captures one
cudagraph_trees graph across filter + index + cascade. The compact
kernels were refactored to return full-width `[B, N]` `(ids, counts)`,
removing the `counts.max().item()` host sync.

### Stage 2b — `custom_op` → `triton_op` (2026-05-23, completed by K2)

`@custom_op` is opaque: inductor and `torch.export` cannot see the
underlying `@triton.jit` kernel. Stage 2b flipped the silvertorch and
bit-KNN wrappers to `@torch.library.triton_op` with a textually-inline
`wrap_triton(...)` launch. The filter/compact kernels initially stayed
behind because their shape-branching prep didn't trace cleanly; K2 moved
that branching into the eager `_impl`s and finished the migration.

**All ten Triton kernel ops are `@triton_op`.** The CUDA backend later
added three ops on `@torch.library.custom_op` — deliberately, since a C++
extension has no `@triton.jit` body for inductor to see. Current totals:
13 registered ops across 8 kernel files. (An earlier version of this
roadmap claimed nothing remains on `@custom_op`; that stopped being true
when the CUDA backend landed.)

The host-side `if actual_k < k: pad` tail was eliminated rather than
moved caller-side: `oporp_1bit_match_topk_indirect` widens its score
buffer so `topk(k)` always has ≥ k lanes, and the silvertorch wrappers
rely on index-build asserts. The pad branch was dead in production.

---

## Archived

[archive/](archive/) holds plans whose work fully landed and which are no
longer instructions for anyone. Currently: the SilverTorch reverse-clause
wrapper fix (shipped in `cc85d8f`; the remaining sweep rerun is tracked
as Open work item 2).

Also deleted along the way, subsumed by the system docs: the prototype
custom_op migration doc, the per-item research doc, the deferred
mask-compact-kernel doc, the Stage 1 autotune-separation plan, the Stage
2 `02-triton-op-migration.md` plan, and the `plans-silvertorch-backup/`
tree (which held the `ShardedSilverTorch` sketches and a shelved
native-CUDA experiment — that experiment is no longer shelved, it
shipped as `backend="cuda"`).
