# Plans roadmap — ordered work queue

This file orders the other plans in [docs/plans/](.) by priority. Read this first; the three main-thread stages have their own docs.

The main thread is a three-stage kernel cleanup: **separate autotune from kernel call sites → wrap kernels in `triton_op` / `custom_op` so `torch.compile(dynamic=True, mode="reduce-overhead")` stops graph-breaking → revisit per-kernel optimizations on the clean base.** Each stage unblocks the next; doing them in order means the optimization work in stage 3 lands on kernels that already have stable, traceable call shapes.

Everything else — `torch.export` readiness, the standalone feature plans — is deferred behind the main thread.

> **Scope note (2026-05-19):** silvertorch (IVF + codesigned probe + IVF-side sharding) is **no longer a planning target**. Historical plans covering silvertorch, the shelved native-CUDA `codesigned_probe_score` experiment, and `ShardedSilverTorch` are preserved in [../plans-silvertorch-backup/](../plans-silvertorch-backup/). Active work below targets the **linr** family only (`SimilarityMasking` / `PrefilterKNN` / `OneBitKNN`) plus the filter layers (`ExactAttributeFilter`, `BloomFilter`).

---

## Main thread

### Stage 1 — Autotune separation — ✅ **done (2026-05-22)**

Shipped: every in-scope kernel now exposes a `<Name>Config` dataclass +
`DEFAULT_CONFIG` (single per-arch curated default, no REGISTRY/lookup)
next to the `@triton.jit` body. The `@custom_op`-wrapped kernels
delegate to a private `_<name>_impl(..., *, config=None)` so the public
op keeps its fixed schema. Offline tuning via
[evaluation/scripts/tune_kernels.py](../../evaluation/scripts/tune_kernels.py)
(`uv run tune-kernels --kernel <name>`) sweeps a hard-coded grid on
real-eval shapes (read from `evaluation/data/<dataset>/item_attrs_narrow.pt`
+ `evaluation/retrieval/config.py`) and emits a pasteable
`DEFAULT_CONFIG = ...` line. Covered: `clause_mask`, `clause_compact`,
`bloom_compact`, `fused_masked_knn_topk`, `oporp_1bit_match_topk`,
`codesigned_probe_score`, `codesigned_probe_score_exact`. (`bloom_match`
retains its hard-coded tile — the per-call width is dictated by `N`.)
See
[../system/kernels.md → Autotune separation](../system/kernels.md#autotune-separation)
for the convention. The original plan doc has been deleted as superseded.

### Stage 2 — `torch.compile` fixes via `triton_op` / `custom_op` — 🟡 **in progress (4 of 5 shipped, 2026-05-22)**

[02-triton-op-migration.md](02-triton-op-migration.md)

Replace every `@torch._dynamo.disable` on a kernel host wrapper with `@torch.library.custom_op` + explicit `register_fake`. End state: each linr algo's `self.compile(dynamic=True, mode="reduce-overhead")` captures a single cudagraph_trees graph across the whole filter + index + cascade — no graph breaks per kernel call.

Shipped on `clause_mask`, `clause_compact`, `bloom_compact`, `fused_masked_knn_topk`, and (outside the original five-kernel scope) `bloom_match`. The compact kernels were refactored to return full-width `[B, N]` `(ids, counts)` along the way, removing the `counts.max().item()` host sync.

**Remaining**: `oporp_1bit_match_topk`. The Optional-to-dummy-tensor caller-side refactor in `OneBitKNN` is still to land. After that, Stage 2 closes and Stage 3 items 1 + 2 are unblocked.

Silvertorch kernels (`codesigned_probe_score`, `codesigned_probe_score_exact`) keep `@torch._dynamo.disable` — silvertorch is explicitly out of Stage 2 scope per the scope note above.

### Stage 3 — Kernel optimizations on the clean base

[03-kernel-optimizations.md](03-kernel-optimizations.md)

Stages 1+2 subsumed the big-lever items (the `counts.max().item()` host sync across the compact family; the `.item()` guard in `OneBitKNNTriton.forward`; the `out_indices[:, :p].contiguous()` slice). Two items remain:

1. Hardware popcount in `oporp_1bit_match_topk` (PTX dump → maybe swap SWAR for `popc.b64`).
2. Allocator hygiene in `oporp_1bit_match_topk` / `fused_masked_knn_topk` (kill the `torch.full(-inf)` pre-fill; replace `cat`-to-pad with pre-allocate-and-slice).

---

## Later (deferred behind the main thread)

- [torch-export-refactor.md](torch-export-refactor.md) — the broader plan to make every linr kernel + layer `torch.export`-ready. After stage 1 lifts the KernelConfig pattern out, what remains is layer-side: per-layer `mode` flag (Optional collapse), `forward_candidates` extraction, `_build_query_signatures_eager` / `_project_oporp_1bit_query_eager` export-path bypasses, `evaluation/retrieval/build_export.py` scaffold, AOTI wiring. Revisit when there's a concrete consumer for `.pt2` artifacts.
- [live-update-api.md](live-update-api.md) — upsert/delete API for V1/V2/V3 + filters. Standalone feature work, independent of stages 1-3 (no kernel surgery; just a `LiveIndexMixin` on the layers).
- [yambda-hf-migration.md](yambda-hf-migration.md) — data migration, blocked on a host with the yambda data + checkpoints locally. Code is already in place.

---

## Cleanup status

Cleanup tasks finished as of 2026-05-22: the prototype custom_op migration doc, the per-item research doc, and the deferred mask-compact-kernel doc have all been deleted (subsumed by the active plans above and the system docs). The `torch-export-refactor.md` plan retains its Phase 1 KernelConfig section as a back-reference — the structure shipped as part of Stage 1 — and the rest of its phases are still pending; Phase 3 (silvertorch) is stubbed out per the scope note above.
