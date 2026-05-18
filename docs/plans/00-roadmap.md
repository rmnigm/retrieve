# Plans roadmap — ordered work queue

This file orders the other plans in [docs/plans/](.) by priority. Read this first; the three main-thread stages have their own docs.

The main thread is a three-stage kernel cleanup: **separate autotune from kernel call sites → wrap kernels in `triton_op` / `custom_op` so `torch.compile(dynamic=True, mode="reduce-overhead")` stops graph-breaking → revisit per-kernel optimizations on the clean base.** Each stage unblocks the next; doing them in order means the optimization work in stage 3 lands on kernels that already have stable, traceable call shapes.

Everything else — `torch.export` readiness, the shelved CUDA experiment, the standalone feature / scaling plans — is deferred behind the main thread.

---

## Main thread

### Stage 1 — Autotune separation

[01-autotune-separation.md](01-autotune-separation.md)

Move every Triton kernel's tuning surface (`@triton.autotune` configs, the `bloom_match` BLOCK_N if-ladder, the `fused_masked_knn_topk` `_bucket_p` ladder, `do_not_specialize` hints) out of the kernel host-call site into a per-kernel `Config` dataclass + arch-keyed `REGISTRY` + `lookup(device, problem_hint)` helper. Stand up an offline `tune-kernels` script that benchmarks the config grid on the local arch and emits REGISTRY-pasteable rows.

Structural, not perf. Acceptance: parity tests pass, eager perf within ±5% of the prior autotune-selected configs.

Lifts the KernelConfig pattern out of [torch-export-refactor.md](torch-export-refactor.md) Phase 1; strips all other Phase 1-5 concerns (mode flags, `.item()` removal, `build_export.py`).

### Stage 2 — `torch.compile` fixes via `triton_op` / `custom_op`

[02-triton-op-migration.md](02-triton-op-migration.md)

Replace every `@torch._dynamo.disable` on a kernel host wrapper with `@torch.library.custom_op` + explicit `register_fake`. End state: each algo's `self.compile(dynamic=True, mode="reduce-overhead")` captures a single cudagraph_trees graph across the whole filter + index + cascade — no graph breaks per kernel call.

Extends the 3-kernel work in [migrate-clean-triton-custom-op.md](migrate-clean-triton-custom-op.md) to all 7 kernels by absorbing two caller-side refactors that the original plan deferred:
- Compact kernels (`bloom_compact`, `clause_compact`) return full-width `[B, N]` `(ids, counts)` instead of `[B, P]` (removes the `counts.max().item()` host sync).
- Kernels with `Optional[Tensor]` args (`oporp_1bit_match_topk`, `codesigned_probe_score`) require pre-allocated dummy tensors from the caller + a Python `bool` for the constexpr flag.

Preserves the measurement protocol from `migrate-clean-triton-custom-op.md` verbatim (before / after timing per algo path; cudagraph-warning grep; algo × backend × filter smoke; top-K parity against eager reference).

### Stage 3 — Kernel optimizations on the clean base

[03-kernel-optimizations.md](03-kernel-optimizations.md)

The surviving items from [kernel-optimization-research.md](kernel-optimization-research.md) after stages 1+2 subsume the big-lever ones (the `counts.max().item()` host sync across the compact family; the `.item()` guard in `OneBitKNNTriton.forward`; the `out_indices[:, :p].contiguous()` slice). Three small items remain:

1. Hardware popcount in `oporp_1bit_match_topk` (PTX dump → maybe swap SWAR for `popc.b64`).
2. Allocator hygiene in `oporp_1bit_match_topk` / `fused_masked_knn_topk` (kill the `torch.full(-inf)` pre-fill; replace `cat`-to-pad with pre-allocate-and-slice).
3. Archive [mask-compact-kernel.md](mask-compact-kernel.md) — stage 2 eliminates its only hot caller, both "When to actually do this" triggers should now be unmet.

---

## Later (deferred behind the main thread)

- [torch-export-refactor.md](torch-export-refactor.md) — the broader 5-phase plan to make every kernel + layer `torch.export`-ready. After stage 1 lifts the KernelConfig pattern out, what remains is layer-side: per-layer `mode` flag (Optional collapse), `forward_candidates` extraction, `_build_query_signatures_eager` / `_project_oporp_1bit_query_eager` export-path bypasses, `evaluation/retrieval/build_export.py` scaffold, AOTI wiring. Revisit when there's a concrete consumer for `.pt2` artifacts.
- [silvertorch-cuda-shelved.md](silvertorch-cuda-shelved.md) — dead. Native-CUDA `codesigned_probe_score` was tried, lost to Triton on paper-like shapes, archived. No action unless a fundamentally new approach (`mma.sync` int8 or similar) is on the table.
- [live-update-api.md](live-update-api.md) — upsert/delete API for V1/V2/V3 + filters. Standalone feature work, independent of stages 1-3 (no kernel surgery; just a `LiveIndexMixin` on the layers).
- [sharding.md](sharding.md) — `ShardedSilverTorch` scatter-gather wrapper. Standalone scaling work; not implemented.
- [yambda-hf-migration.md](yambda-hf-migration.md) — data migration, blocked on a host with the yambda data + checkpoints locally. Code is already in place.

---

## Cleanup once the main thread lands

After stages 1-3 ship, several existing plans are partly or fully subsumed and can be annotated or archived:

- [migrate-clean-triton-custom-op.md](migrate-clean-triton-custom-op.md) → annotate "superseded by 02-triton-op-migration.md" (the 3-kernel migration is the prototype that stage 2 generalizes).
- [kernel-optimization-research.md](kernel-optimization-research.md) → annotate "items 1-2 done by stage 2; items 4c / 5 / 6 subsumed by stages 1-2; items 3 / 4a / 4b / 4d are stage 3" with pointers.
- [mask-compact-kernel.md](mask-compact-kernel.md) → archive (stage 3 action item).
- [torch-export-refactor.md](torch-export-refactor.md) → keep, but trim Phase 1's KernelConfig+REGISTRY section (now lifted into stage 1) and note that the remainder is what's left.
