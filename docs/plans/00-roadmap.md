# Plans roadmap — ordered work queue

This file orders the other plans in [docs/plans/](.) by priority. Read this first; the three main-thread stages have their own docs.

The main thread is a three-stage kernel cleanup: **separate autotune from kernel call sites → wrap kernels in `triton_op` / `custom_op` so `torch.compile(dynamic=True, mode="reduce-overhead")` stops graph-breaking → revisit per-kernel optimizations on the clean base.** Each stage unblocks the next; doing them in order means the optimization work in stage 3 lands on kernels that already have stable, traceable call shapes.

Everything else — `torch.export` readiness, the standalone feature plans — is deferred behind the main thread.

> **Scope note (2026-05-19, revised 2026-05-23):** silvertorch (IVF + codesigned probe) is back in scope for kernel-decoration cleanup only — its three `triton_op` migrations (`codesigned_probe_score`, `codesigned_probe_score_bloom`, `codesigned_probe_score_exact`) shipped under Stage 2b alongside the linr work. IVF-side sharding and `ShardedSilverTorch` remain shelved; the historical plans for those (and the shelved native-CUDA `codesigned_probe_score` experiment) are preserved in [../plans-silvertorch-backup/](../plans-silvertorch-backup/). Active main-thread work below targets the **linr** family (`SimilarityMasking` / `PrefilterKNN` / `OneBitKNN`), the filter layers (`ExactAttributeFilter`, `BloomFilter`), and silvertorch's three probe-score kernels.

---

## Main thread

### Stage 1 — Autotune separation — ✅ **done (2026-05-22)**

Shipped: every in-scope kernel now exposes a `<Name>Config` dataclass +
`DEFAULT_CONFIG` (single per-arch curated default, no REGISTRY/lookup)
next to the `@triton.jit` body. The `@custom_op`-wrapped kernels
delegate to a private `_<name>_impl(..., *, config=None)` so the public
op keeps its fixed schema. Offline tuning via
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

[02-triton-op-migration.md](02-triton-op-migration.md)

Replaced `@torch._dynamo.disable` on every in-scope kernel host wrapper with `@torch.library.custom_op` + explicit `register_fake`, so each linr algo's `self.compile(dynamic=True, mode="reduce-overhead")` captures a single cudagraph_trees graph across the whole filter + index + cascade — no graph breaks per kernel call. Shipped on `clause_mask`, `clause_compact`, `bloom_compact`, `fused_masked_knn_topk`, `bloom_match`, and `oporp_1bit_match_topk`. The compact kernels were refactored to return full-width `[B, N]` `(ids, counts)` along the way, removing the `counts.max().item()` host sync.

### Stage 2b — `custom_op` → `triton_op` for the silvertorch + onebitknn family — ✅ **done (2026-05-23)**

`@custom_op` is opaque — inductor and `torch.export` can't see the underlying `@triton.jit` kernel. Stage 2b flipped the silvertorch + onebitknn wrappers to `@torch.library.triton_op` + textually-inline `wrap_triton(_kernel)[grid](...)` so that (a) inductor can fuse around the kernel for Stage 3 optimizations, (b) `torch.export` preserves the kernel reference in the exported program.

Migrated: `bloom_match` (already done), `codesigned_probe_score` + `codesigned_probe_score_bloom`, `codesigned_probe_score_exact`, `oporp_1bit_match_topk_full` + `oporp_1bit_match_topk_indirect`. The `@custom_op`-wrapped filter/compact kernels (`clause_mask`, `clause_compact`, `bloom_compact`, `fused_masked_knn_topk`) stay on `@custom_op` — their bodies have shape-branching prep work that doesn't trace cleanly under `triton_op`'s make_fx-traceable fake-impl requirement, and they don't sit on the kernel-fusion or kernel-export critical path.

The host-side `if actual_k < k: pad` tail (formerly in every silvertorch/onebitknn wrapper) was eliminated rather than moved caller-side: `oporp_1bit_match_topk_indirect` widens the score buffer to `max(_bucket_n(n_loop), _bucket_n(k))` so `torch.topk(., k)` always has ≥ k lanes; the silvertorch wrappers and `oporp_1bit_match_topk_full` rely on layer-construction asserts (`SilverTorch.register_index` checks `k <= n_probe * max_cluster_size`; `OneBitKNN.register_index` checks `k <= corpus_size`). The pad branch was dead in production anyway. See [../system/kernels.md → Graph-break behavior](../system/kernels.md) for the as-built shape.

### Stage 3 — Kernel optimizations on the clean base

[03-kernel-optimizations.md](03-kernel-optimizations.md)

Stages 1+2 subsumed the big-lever items (the `counts.max().item()` host sync across the compact family; the `.item()` guard in `OneBitKNNTriton.forward`; the `out_indices[:, :p].contiguous()` slice). Two items remain:

1. Hardware popcount in `oporp_1bit_match_topk` (PTX dump → maybe swap SWAR for `popc.b64`).
2. Allocator hygiene in `oporp_1bit_match_topk` / `fused_masked_knn_topk` (kill the `torch.full(-inf)` pre-fill; replace `cat`-to-pad with pre-allocate-and-slice).

---

## Later (deferred behind the main thread)

- [torch-export-refactor.md](torch-export-refactor.md) — the broader plan to make every linr + silvertorch kernel + layer `torch.export`-ready. **Significantly shrunk after Stages 1+2+2b (2026-05-23 reassessment).** All kernel-side work (Configs, `triton_op`/`custom_op` decoration, full-width `(ids, counts)` API, no `.item()` on host, host-side `pad` tail eliminated) shipped as a side effect of Stages 1+2+2b. The `_build_query_signatures_compiled` / `_project_oporp_1bit_query_compiled` wrappers the original plan needed to bypass were deleted entirely; `oporp_1bit_match_topk` is now two custom_ops (`_full` + `_indirect`); `codesigned_probe_score` is now two `triton_op`s (`_probe_score` + `_probe_score_bloom`) plus the separate `_exact` variant. **Remaining: 1–2 PRs** — `mode: Literal[...]` flag on `PrefilterKNN` and `OneBitKNN`, sibling-method `forward_*` per filter mode on `SilverTorch` (plus `forward_candidates` extraction), algo wiring across the three algos, `evaluation/retrieval/build_export.py` scaffold. See [the doc's "Remaining work — consolidated" section](torch-export-refactor.md#remaining-work--consolidated-2026-05-23). AOTI wiring (`torch>=2.5` + `aoti_compile_and_package`) and `ShardedSilverTorch` stay separate later efforts. Revisit when there's a concrete consumer for `.pt2` artifacts.
- [live-update-api.md](live-update-api.md) — upsert/delete API for V1/V2/V3 + filters. Standalone feature work, independent of stages 1-3 (no kernel surgery; just a `LiveIndexMixin` on the layers).
- [yambda-hf-migration.md](yambda-hf-migration.md) — data migration, blocked on a host with the yambda data + checkpoints locally. Code is already in place.

---

## Cleanup status

Cleanup tasks finished as of 2026-05-22: the prototype custom_op migration doc, the per-item research doc, and the deferred mask-compact-kernel doc have all been deleted (subsumed by the active plans above and the system docs). The `torch-export-refactor.md` plan's per-phase bodies are preserved as historical execution briefs; the 2026-05-23 status block at the top of that doc supersedes them — Phases 1 and 2 are fully shipped, Phases 3, 4, and 5 have their kernel work shipped (only layer-side `mode` flag / sibling-method surgery and `build_export.py` scaffold remain). Phase 3 (silvertorch) was restored alongside the 2026-05-23 scope-note revision; `ShardedSilverTorch` and the shelved native-CUDA experiment stay in [../plans-silvertorch-backup/](plans-silvertorch-backup/).
