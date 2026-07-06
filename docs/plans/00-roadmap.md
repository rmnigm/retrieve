# Plans roadmap — ordered work queue

This file orders the other plans in [docs/plans/](.) by priority. Read this first; the three main-thread stages have their own docs.

The main thread is a three-stage kernel cleanup: **separate autotune from kernel call sites → wrap kernels in `triton_op` / `custom_op` so `torch.compile(dynamic=True, mode="reduce-overhead")` stops graph-breaking → revisit per-kernel optimizations on the clean base.** Each stage unblocks the next; doing them in order means the optimization work in stage 3 lands on kernels that already have stable, traceable call shapes.

Everything else — `torch.export` readiness, the standalone feature plans — is deferred behind the main thread.

> **Scope note (2026-05-19, revised 2026-05-23):** silvertorch (IVF + codesigned probe) is back in scope for kernel-decoration cleanup only — its three `triton_op` migrations (`codesigned_probe_score`, `codesigned_probe_score_bloom`, `codesigned_probe_score_exact`) shipped under Stage 2b alongside the linr work. IVF-side sharding and `ShardedSilverTorch` remain shelved; the historical plans for those (and the shelved native-CUDA `codesigned_probe_score` experiment) are preserved in [../plans-silvertorch-backup/](../plans-silvertorch-backup/). Active main-thread work below targets the **linr** family (`PostfilterKNN` / `PrefilterKNN` / `OneBitKNN`), the filter layers (`ExactAttributeFilter`, `BloomFilter`), and silvertorch's three probe-score kernels.

---

## Main thread

### Stage 1 — Autotune separation — ✅ **done (2026-05-22)**

Shipped: every in-scope kernel now exposes a `<Name>Config` dataclass +
`DEFAULT_CONFIG` (single per-arch curated default, no REGISTRY/lookup)
next to the `@triton.jit` body. Config overrides go through a private
`_<name>_impl(..., *, config=None)` companion so the public op keeps
its fixed schema (the ops were `@custom_op` at the time; all are
`@triton_op` as of 2026-07-06 — see the Stage 2b note). Offline tuning via
[retrieve/src/retrieve/tune.py](../../retrieve/src/retrieve/tune.py)
(`uv run tune-kernels <kernel-subcommand>`) sweeps a hard-coded grid on
real-eval shapes (read from `evaluation/data/<dataset>/item_attrs_narrow.pt`
+ `evaluation/retrieval/config.py`) and emits a pasteable
`DEFAULT_CONFIG = ...` line. The filter-kernel subcommands also accept
repeatable `--regime N,B,C,A_MAX` / `--regime N,B,W` flags so downstream
consumers can re-tune against their own catalog shapes. Covered: `clause_mask`, `clause_compact`,
`bloom_compact`, `fused_masked_knn_topk`, `oporp_1bit_match_topk`,
`codesigned_probe_score`, `codesigned_probe_score_exact`. (`bloom_match`
retains its hard-coded tile — the per-call width is dictated by `N`.)
See
[../system/kernels.md → Autotune separation](../system/kernels.md#autotune-separation)
for the convention. The original plan doc has been deleted as superseded.

### Stage 2 — `torch.compile` fixes via `custom_op` — ✅ **done (2026-05-22)**

Replaced `@torch._dynamo.disable` on every in-scope kernel host wrapper with `@torch.library.custom_op` + explicit `register_fake`, so each linr algo's `self.compile(dynamic=True, mode="reduce-overhead")` captures a single cudagraph_trees graph across the whole filter + index + cascade — no graph breaks per kernel call. Shipped on `clause_mask`, `clause_compact`, `bloom_compact`, `fused_masked_knn_topk`, `bloom_match`, and `oporp_1bit_match_topk`. The compact kernels were refactored to return full-width `[B, N]` `(ids, counts)` along the way, removing the `counts.max().item()` host sync. Plan doc deleted as superseded. (2026-07-06: `@custom_op` has since been replaced by `@triton_op` everywhere — see the Stage 2b note.)

### Stage 2b — `custom_op` → `triton_op` for the silvertorch + onebitknn family — ✅ **done (2026-05-23)**

`@custom_op` is opaque — inductor and `torch.export` can't see the underlying `@triton.jit` kernel. Stage 2b flipped the silvertorch + onebitknn wrappers to `@torch.library.triton_op` + textually-inline `wrap_triton(_kernel)[grid](...)` so that (a) inductor can fuse around the kernel for Stage 3 optimizations, (b) `torch.export` preserves the kernel reference in the exported program.

Migrated: `bloom_match` (already done), `codesigned_probe_score` + `codesigned_probe_score_bloom`, `codesigned_probe_score_exact`, `oporp_1bit_match_topk_full` + `oporp_1bit_match_topk_indirect`. At the time, the `@custom_op`-wrapped filter/compact kernels (`clause_mask`, `clause_compact`, `bloom_compact`, `fused_masked_knn_topk`) stayed on `@custom_op` because their shape-branching prep (P-bucketing, `p == 0` early-return, pad tails) didn't trace cleanly under `triton_op`'s make_fx-traceable requirement. **Superseded (2026-07-06):** the kernels-layers refactor (K2, branch `refactor/kernels-eval`) moved that shape-branching into the eager `_impl`s only (explicit `bucket=`/`pad_to_k=` flags on shared prep/finish helpers) and migrated the remaining four to `@triton_op` — **all ten registered ops across the seven kernel files are now `@triton_op` + inline `wrap_triton`**; nothing remains on `@custom_op`. See [../system/kernels.md → Graph-break behavior](../system/kernels.md) for the as-built shape (pending GPU validation, [refactor-validation-handoff.md](refactor-validation-handoff.md)).

The host-side `if actual_k < k: pad` tail (formerly in every silvertorch/onebitknn wrapper) was eliminated rather than moved caller-side: `oporp_1bit_match_topk_indirect` widens the score buffer to `max(_bucket_n(n_loop), _bucket_n(k))` so `torch.topk(., k)` always has ≥ k lanes; the silvertorch wrappers and `oporp_1bit_match_topk_full` rely on layer-construction asserts (`SilverTorch.register_index` checks `k <= n_probe * max_cluster_size`; `OneBitKNN.register_index` checks `k <= corpus_size`). The pad branch was dead in production anyway. See [../system/kernels.md → Graph-break behavior](../system/kernels.md) for the as-built shape.

### Stage 3 — Kernel optimizations on the clean base — deferred

Stages 1+2 subsumed the big-lever items (the `counts.max().item()` host sync across the compact family; the `.item()` guard in `OneBitKNNTriton.forward`; the `out_indices[:, :p].contiguous()` slice). Two items remain but are deferred (no committed timeline):

1. Hardware popcount in `oporp_1bit_match_topk` (PTX dump → maybe swap SWAR for `popc.b64`).
2. Allocator hygiene in `oporp_1bit_match_topk` / `fused_masked_knn_topk` (kill the `torch.full(-inf)` pre-fill; replace `cat`-to-pad with pre-allocate-and-slice).

The detailed plan doc (`03-kernel-optimizations.md`) has been deleted.

### Stage 4 — Thesis results expansion

> **Stale references (verified 2026-07-06):** the `docs/thesis/*.md` chapter/notes files and `docs/thesis/results-data/` CSVs this stage links to no longer exist in the tree — `docs/thesis/` now holds only `main.tex`, `references.bib`, and `figures/` (the thesis moved to LaTeX; the markdown chapter drafts were removed after commit `438acca`). There is also no repo-root `thesis/figures/scripts/` directory. The *work items* below remain valid; resolve the plot catalog / chapter references against the LaTeX sources or the thesis workspace before executing them.

Driven by the Chapter 6 plot catalog at ~~[docs/thesis/results-data/recipes/plot_catalog.md](../../docs/thesis/results-data/recipes/plot_catalog.md)~~ (missing) and the planned schema extensions documented in ~~[docs/thesis/06-eval-protocol.md](../../docs/thesis/06-eval-protocol.md)~~ (missing) §5.7. Both unblock figures B3/B4/E1/E2/F3/F4/G2 in Ch.6 (results notes at ~~[docs/thesis/07-results.md](../../docs/thesis/07-results.md)~~, missing).

**Stage 4a — Schema extensions in the eval harness** (library `retrieve/` is untouched). Post-refactor home: latency/memory fields extend `PerfStats` in [evaluation/retrieval/measure.py](../../evaluation/retrieval/measure.py) (additive dataclass fields — no call-site churn, that was E5.2's point); pass-level items land in [evaluation/retrieval/passes.py](../../evaluation/retrieval/passes.py). (`bench_tools.py` no longer exists — it was split into `measure.py` / `encode.py` / `passes.py`.)

1. `throughput_qps` (= `n_queries_total / total_wall_clock_s` over the timing window) — eliminates the rate-statistic ambiguity in current Pareto plots and unblocks A2 / A4 / B4 in throughput form.
2. `mean_ms` alongside `median_ms`.
3. `p99_ms` (ideally `p999_ms` too) — needed for any online-serving SLO claim.
4. `build_time_s` — wall-clock for `build_algorithm`, ideally split per phase (encode / quantize / cluster / assemble) — unlocks F4 stacked bar.
5. `topk_ids_jaccard_vs_torch` — index-set Jaccard between Triton and torch backends (target >0.999) — stronger parity guarantee than `recall_abs_diff` for F1 / G4.
6. Per-Triton-kernel timing + occupancy via Nsight Compute — Stage 4a's heaviest item; manual one-off per kernel.
7. Per-query latency vector (not just median / quantiles) — enables E1 violins and paired-permutation significance tests without re-running.

Each is independent; ordering is roughly by cost (1-3 cheapest; 4-5 next; 6-7 require dedicated profiling / harness re-architecture).

**Stage 4b — Measurement gap re-runs** (no schema change; just additional cells):

1. yambda-5b-d256 quality config (closes the only data-coverage hole in §6.2.3 / §6.6.1).
2. SimHashKNN `k_bits` deep sweep on goodreads-d128 (new YAML; populates the B2 bit-budget axis added in commit `4f9b1a6`).
3. Multi-seed re-run on 1 representative cell (E2; recommend goodreads-d128 with seeds {0..4}).
4. Extended batch-size grid {1, 4, 16, 64, 256, 1024} on arxiv-d128 (perf-only pass; quality is bs-invariant per the `quality_pass_cached` docstring in [passes.py](../../evaluation/retrieval/passes.py)).
5. K=10 quality runs on the existing quality configs (single-line YAML edit; unblocks B3 and G2).
6. arXiv per-clause selectivity CSV extraction (mirror of `goodreads_clause_c*.csv`; CPU-bound; closes the D3 / E3 arxiv-side gap noted in [04-datasets-notes.md §5.6](../../docs/thesis/04-datasets-notes.md)).
7. **Pre-publication blocker** — goodreads-d128 + d256 filter sweeps rerun against a fresh oracle (stale-cache bug was documented at the top of the removed `docs/thesis/07-results.md`). **Code fix implemented (2026-07-06, branch `refactor/kernels-eval`, E6):** the oracle disk cache is now a `gt_topk_v3_<sweep>.pt` dict blob keyed by a content fingerprint of `(item_embs, queries, qa_narrow, K_GT)` — same-shape stale caches recompute automatically ([oracle.py](../../evaluation/retrieval/oracle.py)). **The rerun itself remains**: after GPU validation passes, re-run the goodreads d128/d256 filter configs; the first run rebuilds each sweep's oracle once (old `gt_topk_v2_*` files are ignored and can be deleted).
8. Cross-dataset deep-sweep coverage for §6.6 figures. Currently only one deep-sweep CSV exists per algorithm: `deep_sweep_linr_v3_goodreads_d128.csv` (used by fig-6-6, Recall vs candidate_pool) and `deep_sweep_silvertorch_arxiv_d128.csv` (used by fig-6-7, Recall vs latency for `n_lists × n_probe`). Each figure is therefore locked to one dataset, which is hard to defend in §6.6 ("is the trend dataset-specific or general?"). Run the mirror sweeps so each algorithm has both datasets and the figures can show either two-curve overlays or a 2×2 grid:
   - `deep_sweep_linr_v3_arxiv_d128.yaml` — V3 `candidate_pool` × filter-sweep on arXiv (mirrors the goodreads layout already in [deep_sweep_linr_v3_goodreads_d128.csv](../thesis/results-data/deep_sweep_linr_v3_goodreads_d128.csv)).
   - `deep_sweep_silvertorch_goodreads_d128.yaml` — IVF `n_lists × n_probe` Pareto on Goodreads (mirrors [deep_sweep_silvertorch_arxiv_d128.csv](../thesis/results-data/deep_sweep_silvertorch_arxiv_d128.csv)).

   After both land, update [thesis/figures/scripts/plot_6_6_linr_v3_pc_sweep.py](../../thesis/figures/scripts/plot_6_6_linr_v3_pc_sweep.py) and [thesis/figures/scripts/plot_6_7_silvertorch_pareto.py](../../thesis/figures/scripts/plot_6_7_silvertorch_pareto.py) to facet by dataset (or overlay), and rewrite §6.6 captions.

Stage 4b items 1–6 are nice-to-haves that unblock specific Ch.6 figures. **Item 7 is a publication blocker** — the goodreads-d128/d256 filter numbers cannot be cited until the rerun completes. **Item 8 is recommended before submission** — current single-dataset deep-sweep figures invite a "is this dataset-specific?" reviewer question.

---

## Refactor track (proposed 2026-07-03 — implemented 2026-07-06, pending GPU validation)

Two structure-preserving cleanup plans written from a full audit of both packages; they change
no measured numbers and no op schemas. **Status: both plans are implemented on branch
`refactor/kernels-eval` (2026-07-06) — all code phases (K1–K8, E1–E7 + E8.1 tests, plus the
`filter` → `filter_mode` rename) are committed, and the K9/E8.2 doc sweeps are done. The branch
is pending GPU validation; the ordered runbook, pass criteria, deviations, and fallback recipes
live in [refactor-validation-handoff.md](refactor-validation-handoff.md).** The two plan docs
below stay intact (not trimmed per the usual "Cleanup status" convention) until that validation
passes; note that one Stage 2b-era claim in this file was corrected above (all kernels are now
`@triton_op` — nothing remains on `@custom_op`).

- [kernels-layers-design.md](kernels-layers-design.md) — library-side dedup + API consistency:
  fix the broken `tune-kernels codesigned-probe-score` subcommand (K1, called the public op with
  kwargs removed in Stage 2b); shared host-wrapper prep/epilogue so validation reaches the
  production `triton_op` path — and the four remaining `@custom_op` kernels migrated to
  `@triton_op` along the way (K2); shared `@triton.jit` predicate helpers in
  `kernels/common.py` (K3); `masked_topk` util + fold `SimHashKNN`'s ~95% copy of `OneBitKNN`
  into the shared `_PackedBitsKNN` base (K4); bloom-hash core made public in
  `layers/filters/bloom_hash.py` for `SilverTorch` (K5); `FilterModule.register_index` kw-only
  alignment + reintroduce the minimal `RetrievalModule` ABC that live-update-api assumes (K6);
  tune.py `KernelTuneSpec` registry with seven subcommands incl. the new
  `codesigned-probe-score-exact` (K7); new tests (K8); system-doc drift sweep (K9, done).
- [evaluation-refactor.md](evaluation-refactor.md) — harness readability/leanness: delete the
  broken-and-unused `torch_knn` algo, the unreachable CPU-timing path, and three unused
  dependencies, quarantine the campaign-bound upload script behind `--repo-id`, and rename the
  `datasets` package to `eval_datasets` (E1); replace the 15–20-kwarg × 5-level parameter
  threading in `sweep.py` with `SweepContext`/`FilterAssets` dataclasses (E2); typed
  `RetrievalAlgo` protocol + declarative cell-eligibility table instead of
  ValueError-as-control-flow (E3); `AlgoBase` for the duplicated compile/bookkeeping tail (E4);
  split `bench_tools.py` into `measure.py`/`encode.py`/`passes.py` + `PerfStats`/`QualityStats`
  dataclasses that make Stage 4a's schema extensions one-field changes, emitting the additive
  `precision@k`/`mrr@k` columns (E5); **oracle disk cache keyed by content fingerprint instead
  of shape** — the concrete fix for Stage 4b item 7's stale-cache blocker (E6); config/loader
  hygiene (E7); CPU-only unit tests (E8.1, done) + `docs/system/evaluation.md` rewrite (E8.2,
  done).
- [future-work-and-research.md](future-work-and-research.md) — vetted idea catalog beyond the
  refactors: engineering (GPU CI, fused top-k selection vs the host-`torch.topk` rule,
  persistent kernels, RaBitQ/int4 quantization, AOTI serving demo, Pareto/energy/selectivity
  harness columns) and research directions with prior-art citations (filter-aware /
  label-centric IVF, selectivity-adaptive query planning, RaBitQ vs Sign-OPORP under fixed
  kernel budget, MoL OverArch re-ranking stage, GPU NOT-predicate filters, streaming index
  freshness, 100M–1B single-GPU scaling study, filtered-eval methodology write-up). Includes
  an impact × cost prioritization table.

---

## Later (deferred behind the main thread)

- [torch-export-refactor.md](torch-export-refactor.md) — make every linr + silvertorch layer `torch.export`-ready. Kernel-side surface (Configs, `triton_op` / `custom_op`, full-width `(ids, counts)`, no `.item()` on host, no `pad` tail) is in shape; what remains is layer-side `mode: Literal[...]` flags on `PrefilterKNN` / `OneBitKNN`, sibling-method `forward_*` per filter mode on `SilverTorch` (plus `forward_candidates` extraction), algo wiring across the three algos, and the `evaluation/retrieval/build_export.py` scaffold. AOTI wiring (`torch>=2.5` + `aoti_compile_and_package`) and `ShardedSilverTorch` stay separate later efforts. Revisit when there's a concrete consumer for `.pt2` artifacts.
- [live-update-api.md](live-update-api.md) — upsert/delete API for V1/V2/V3 + filters. Standalone feature work, independent of stages 1-3 (no kernel surgery; just a `LiveIndexMixin` on the layers).
- [silvertorch-reverse-clause-wrapper-fix.md](silvertorch-reverse-clause-wrapper-fix.md) — wire `clause_is_reverse` through `SilverTorch.register_index` + the codesigned exact-clause kernel so silvertorch can run on reverse-clause sweeps (was producing ≈0 recall on goodreads `c1_lang_reverse / c0c1 / all4`). *Status note (2026-07-06): the wiring shipped — `SilvertorchAlgo` threads `clause_is_reverse` end-to-end (commit `cc85d8f`) and a CPU test covers it (`tests/test_silvertorch_algo_reverse.py`); the referenced `docs/thesis/07-results.md` no longer exists (see the Stage 4 stale-references note). What remains is rerunning the affected sweeps.*

---

## Cleanup status

The prototype custom_op migration doc, the per-item research doc, the deferred mask-compact-kernel doc, the Stage 1 autotune-separation plan, and the Stage 2 `02-triton-op-migration.md` plan have all been deleted (subsumed by the active plans above and the system docs). `torch-export-refactor.md` was trimmed to the remaining layer-side and `build_export.py` scaffold work — its per-phase historical execution brief is gone. `ShardedSilverTorch` and the shelved native-CUDA experiment stay in [../plans-silvertorch-backup/](plans-silvertorch-backup/).
