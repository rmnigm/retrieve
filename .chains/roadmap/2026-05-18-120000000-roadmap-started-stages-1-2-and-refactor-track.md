---
chain: "roadmap"
branch: "main"
nextStep: "Validate the refactor track (K1-K9 / E1-E8) on an A100 via the refactor-validation-handoff runbook."
created: "2026-05-18T12:00:00Z"
---

# Roadmap history, part 1: the early "Done" entries (May-July 2026)

Source: the "Done" and "Archived" sections of `docs/plans/00-roadmap.md` (created 2026-05-18). The roadmap was the single ordered work queue; this chain holds its history, and the live queue is now the public page `docs/system/roadmap.md`.

## Stage 1: autotune separation (2026-05-22)
Every kernel exposes a `<Name>Config` dataclass and one curated `DEFAULT_CONFIG` next to its `@triton.jit` body; overrides go through a private `_<name>_impl(..., *, config=None)` so the public op keeps a fixed schema; tuning is offline via `tune.py`. `bloom_match` keeps a hard-coded tile (per-call width is dictated by N).

## Stage 2: `torch.compile` fixes (2026-05-22)
Replaced `@torch._dynamo.disable` on every kernel host wrapper with op registration, so each algo's `compile(dynamic=True, mode="reduce-overhead")` captures one cudagraph_trees graph across filter + index + cascade. The compact kernels return full-width `[B, N]` `(ids, counts)`, removing the `counts.max().item()` host sync.

## Stage 2b: `custom_op` -> `triton_op` (2026-05-23, completed by K2)
`@custom_op` is opaque to inductor and `torch.export`; the silvertorch and bit-KNN wrappers moved to `@torch.library.triton_op` with a textually-inline `wrap_triton(...)` launch; K2 finished the filter / compact kernels by moving their shape-branching into the eager `_impl`s. All ten Triton ops became `@triton_op` (later C4 turned `clause_compact` / `bloom_compact` back into opaque `custom_op`s so compiled V2 / V3 capture). The host-side `if actual_k < k: pad` tail was eliminated: `oporp_1bit_match_topk_indirect` widens its score buffer so `topk(k)` always has >= k lanes, and the SilverTorch wrappers rely on index-build asserts.

## Refactor track (implemented 2026-07-06; library gates passed 2026-09-02)
Two structure-preserving plans from a full audit, no measured number or op schema changed, all phases plus the `filter=` -> `filter_mode=` rename on `refactor/kernels-eval`: kernels-layers-design (K1-K9: tuner fix, shared prep / epilogue, shared `@triton.jit` helpers, `masked_topk` and `_PackedBitsKNN`, public bloom-hash core, kw-only filter signatures and the minimal `RetrievalModule`, the `KernelTuneSpec` registry, tests, doc sweep) and evaluation-refactor (E1-E8: dead weight, `datasets` -> `eval_datasets`, context objects, typed protocol and eligibility table, `AlgoBase`, `bench_tools` split with stats dataclasses, the content-fingerprinted oracle cache, config hygiene, tests and the evaluation doc rewrite).

## Archived / deleted along the way
The SilverTorch reverse-clause wrapper fix (landed `cc85d8f`; the rerun tracked as "Stage 4b item 7", later D1). Deleted, subsumed by the system docs: the prototype custom_op migration doc, the per-item research doc, the deferred mask-compact-kernel doc, the Stage 1 and Stage 2 plans, and a `plans-silvertorch-backup/` tree (a `ShardedSilverTorch` sketch and a shelved native-CUDA experiment that later shipped as `backend="cuda"` and was deleted at B4).

## CUDA and CuTe SilverTorch backends
CUDA C++ implemented 2026-07-06, validated and tuned on the A100 2026-09-02 (47/47 parity; the memory-level-parallelism fix took bloom scoring from 147.8 to 58.6 µs kernel-only against Triton's 124.6 µs at B=16, P=58k, D=128). CuTe DSL port implemented 2026-09-02, kernel-for-kernel parity with C++, the cost being DSL launch overhead, gone under CUDA-graph replay. Both deleted at B4 (2026-09-06) after the official ops passed the B2 parity gate; citable as `retrieve@cuda-cute-backends-final`. Records: the cuda-silvertorch and cute-dsl-scorer chains.
