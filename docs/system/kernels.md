---
title: kernels
created: 2026-09-26
updated: 2026-10-15
type: concept
tags: [kernels, library]
sources: [retrieve/src/retrieve/ops/]
---

# `retrieve` kernels

Every kernel of ours is Triton: the LiNR kernels and the filter primitives, flat in
[`ops/triton/`](../../retrieve/src/retrieve/ops/triton/), one file per kernel, registered as
`torch.ops.retrieve.*` when the package is imported (its `__init__.py`); every op has a pure-torch
twin with the same name and signature in
[`ops/reference/`](../../retrieve/src/retrieve/ops/reference/) (the `"torch"` backend and the
parity oracle). `SilverTorch` runs no kernel of ours: its phases 2+3 are Meta's SilverTorch code, adapted
(`backend="cuda"`), behind the adapter in
[`ops/official/`](../../retrieve/src/retrieve/ops/official/adapter.py)
([SilverTorch on Meta's ops](#silvertorch-on-metas-ops)). Our earlier Triton reimplementation of
SilverTorch is deleted ([Deleted backends](#deleted-backends)).

By domain:

- LiNR — `fused_masked_knn_topk` (`PrefilterKNN`), `oporp_1bit_match_topk` (`OneBitKNN` /
  `SimHashKNN`), `gemv_scores` (`PostfilterKNN` at `B = 1`; its dense matmul + top-K is otherwise
  pure torch, there being no fusion to win over cuBLAS LtGemm + CUB).
- filters — `clause_mask` / `clause_mask_scores` and `clause_compact`
  (`ExactAttributeFilter`). The mask/compact split mirrors the filter API split
  documented in [filtering.md](filtering.md).

Cross-kernel `@triton.jit` building blocks live in
[`common.py`](../../retrieve/src/retrieve/ops/triton/common.py)
([Shared kernel helpers](#shared-kernel-helpers-opstritoncommonpy)) and the plain-Python launch
scaffold they share in [`_host.py`](../../retrieve/src/retrieve/ops/triton/_host.py):
`compact_finish` (the compaction op's scan + scatter), the boundary checks and `wide`, and
`grid_batch_tiles` (the 3-D grid split of the filter kernels).

All kernels follow the same conventions.

- One launch per `forward()` call. No persistent threads, no streams.
- Inputs are CUDA tensors. Every `_<name>_prep` checks its boundary
  (`_host.check_contiguous`). An item-side table (the
  index, and the `[B, N]` candidate lists) must arrive contiguous, and a
  view raises `ValueError`, because a per-call `.contiguous()` would copy
  the whole index on every forward. Query-side tensors are small and are
  still `.contiguous()`-copied. Gated by
  [`test_op_boundary.py`](../../retrieve/tests/correctness/test_op_boundary.py).
  A `tl.arange` over the embedding or word width (`D`, `W`) runs at the
  next power of two, masked back to the true width (§ Padding), so any
  width is accepted.
  The prep then passes `.stride(i)` for every axis, so a kernel body
  never hard-codes a layout. The strides are always the contiguous ones.
- `@triton.jit` helpers are imported **by name**
  (`from retrieve.ops.triton.common import clause_pass`), never called as
  `common.clause_pass(...)`. `torch.compile` re-compiles a `@triton_op`'s
  kernel through inductor, which rebuilds the kernel's global namespace
  from the `JITFunction`'s globals and captures `@triton.jit` callees by
  name; a module object is not captured, so `common.<fn>` raises
  `NameError('common is not defined')` at `ast_to_ttir` time. Eager Triton
  resolves the attribute and never sees it; `tests/compile` catches it.
- Top-K selection is **not** in-kernel. Each kernel writes a `[B, ·]` score
  buffer and the host calls `torch.topk` on it. CUB's top-K (under torch)
  is faster than anything we can implement in pure Triton without a
  warp-level radix-select primitive.
- Tile config (`block_n`, `num_warps`, `num_stages`) is offline-tuned per kernel and shipped
  as a single `DEFAULT_CONFIG` constant on the kernel module. No runtime
  `@triton.autotune`. Callers who want a non-default tile pass
  `config=<Kernel>Config(...)` to the private
  `_<name>_impl(..., config=)` companion — the public op has a fixed
  schema and always uses the shipped tile. The `tune-kernels` CLI
  (shipped with the library at `retrieve.ops.tune:main`) sweeps the
  candidate grid on a given arch and prints the line to paste into the
  kernel file. The same convention applies to the LiNR and the filter kernels — see
  [Autotune separation](#autotune-separation) below for the rationale.
- Host-wrapper shape. Every kernel file follows one pattern: a shared
  `_<name>_prep(...)` does validation + contiguity + output-buffer
  allocation + the full launch-kwarg dict (returned in a small frozen
  `_<Name>Launch` dataclass), a shared `_<name>_finish(...)` owns the
  topk/gather/sentinel epilogue where one exists, and every entry point
  — the eager `_impl` and the `@triton_op` public op(s) — is prep →
  one launch line → finish. Validation therefore runs on the
  production ops, not just `_impl`. (`gemv_scores` is the exception: one op at a fixed tile, no
  `_impl`, no Config, no prep.)
- Graph-break behavior. Seven ops are registered across the five Triton kernel files
  (`SilverTorch` registers no op of ours; it calls `torch.ops.st.*`), in two flavours.

  **Six on `@torch.library.triton_op`**, each with a textually-inline
  `wrap_triton(_kernel)[grid](**launch.kwargs)` launch: `clause_mask`, `clause_mask_scores`,
  `fused_masked_knn_topk`, `gemv_scores`, `oporp_1bit_match_topk_full`,
  `oporp_1bit_match_topk_indirect`. The
  decorator stops dynamo from graph-breaking at the wrapper boundary, so
  each layer's full forward captures into one cudagraph_trees graph, and
  lets inductor see the underlying `@triton.jit` kernel (preserves the
  reference under `torch.export` — gated by
  [`tests/compile/test_export_kernel_ref.py`](../../retrieve/tests/compile/test_export_kernel_ref.py)). The `wrap_triton` call
  must stay textually inside each decorated body — dedup targets the
  code around it, never the launch line itself.

  **One on `@torch.library.custom_op`** (`mutates_args=()`, `device_types="cuda"`, plus a
  `register_fake`), opaque to inductor and calling the shared `_impl`: the **stream-compaction**
  kernel `clause_compact`. Under `triton_op`, inductor
  analyses the kernel's TTIR for mutated pointers, and its provenance
  walk follows a store address back through every argument of the
  `tt.call` that produced it. `compact_store`'s address is
  `row_base + intra` — data-dependent on the pass mask, which is the
  result of the `clause_pass` call whose arguments
  include the index buffers — so `item_attrs_ptr` / `query_attrs_ptr` /
  `is_reverse_ptr` (graph inputs) are reported as mutated and
  cudagraph trees skip the *whole* forward with "skipping cudagraphs due
  to mutated inputs", silently falling back to compiled-eager. `clause_mask` stores at a data-independent
  address and is unaffected, which is why it stays a `triton_op`. As a
  custom op the launch is opaque, the op is functional as declared, and the forward captures — gated by
  [`tests/compile/test_linr_compile.py`](../../retrieve/tests/compile/test_linr_compile.py),
  which asserts `cudagraph_skips == 0` for compiled `linr_v1`/`v2`/`v3`. The cost is one custom-op dispatch instead of an
  inlined launch, and the exported graph holds the op node rather than a
  Triton HOP (export and its bit-exact round-trip still work; the kernel
  is simply reached through the op).

## Autotune separation

No kernel uses runtime `@triton.autotune` or a hand-picked module
constant:

- `@triton.autotune` re-tunes at every cache-key shape change, leaking
  compile pressure into the cudagraph-trees capture for
  `torch.compile(dynamic=True, mode="reduce-overhead")`.
- A hard-coded `_BLOCK_N = 256` is a guess, not a measurement against
  real-eval shapes.

The pattern, applied uniformly to every kernel in this tree:

1. A `@dataclass(frozen=True) class <Name>Config` next to the
   `@triton.jit` body holds `block_n`, `num_warps`, `num_stages`.
2. A `DEFAULT_CONFIG` module constant holds the single curated
   default for the current arch (sm_80 / A100 in this repo).
3. The host wrapper takes `config: <Name>Config | None = None`;
   `cfg = config if config is not None else DEFAULT_CONFIG` resolves it.
4. A private `_<name>_impl(..., *, config: ...Config | None = None)`
   lives next to the public op(s) in every file. Both share the same
   `_<name>_prep` / `_<name>_finish` helpers, so their bodies differ
   only in the launch line (`_kernel[grid](**kwargs)` eager vs
   `wrap_triton(_kernel)[grid](**kwargs)` in the `@triton_op` body —
   the launch must appear textually in the decorated function's source,
   which is what `torch.export`'s kernel registry walks to preserve the
   kernel reference; the `custom_op`-registered compaction kernel
   has no launch line of its own and just calls `_impl`) and, where policies intentionally diverge, in
   explicit prep/finish flags (e.g. `fused_masked_knn_topk`'s
   `bucket=` / `pad_to_k=`). The op schema doesn't carry the config
   dataclass; tests and the tuner reach `_impl` directly to pass an
   override.
5. Tuning is offline: [`ops/tune.py`](../../retrieve/src/retrieve/ops/tune.py) (`uv run
   tune-kernels <kernel-subcommand>`) is a declarative registry — one
   `KernelTuneSpec` per kernel in the `KERNELS` tuple, from which the
   four click subcommands are generated
   (`fused-masked-knn-topk`, `oporp-1bit-match-topk`, `clause-mask`, `clause-compact`;
   [`test_tune_smoke.py`](../../retrieve/tests/correctness/test_tune_smoke.py)
   pins the list), all driven by one
   generic `_sweep` + `_print` pair. Each spec sweeps its `(block,
   num_warps)` grid against built-in shape regimes mirroring real-eval
   workloads (goodreads N≈800k, arxiv N≈3M, synth-15M; batch sizes from
   the eval configs). The two clause kernels accept
   repeatable `--regime N,B,C,A_MAX` flags; the
   dimension-swept kernels expose `--d`/`--b`/`--w` flags and derive
   regimes by crossing them with the module's bucket/P axes. The shipped
   `DEFAULT_CONFIG` is always swept too, and it stays unless a
   candidate's geometric-mean time over the regimes is more than
   `NOISE_BAND` (3 %) below it, **and** the candidate is at most
   `WORST_CAP` (5 %) slower in every single regime (`tune._choose`, pinned by
   `test_tune_smoke.py`). A B = 1 regime whose own winner beats the pick
   by ≥ `B1_GAP` (10 %) is flagged, which is the evidence a separate batch-1
   config would need. The CLI emits a pasteable `DEFAULT_CONFIG = ...` line.
   `--json-out` dumps the per-regime sweep (`per_regime`), the rule's
   ratios (`rule`), and `env`: the SM clock sampled after each regime, the
   commit, and the torch / triton versions and device. Re-run once per new
   arch; commit the line.
   (`gemv_scores` has no subcommand: a fixed tile.) A CUDA-gated
   smoke test
   ([`test_tune_smoke.py`](../../retrieve/tests/correctness/test_tune_smoke.py))
   runs one tiny regime per spec so schema drift between tune.py and
   the kernel `_impl`s breaks CI instead of a tuning session.

The compaction kernel has no accumulating state (per-tile counts and offsets are plain stores,
so a repeated launch rewrites the same values); the tuner calls its host wrapper, which allocates
fresh buffers per call either way.

## Addressing

Triton's `program_id` is int32, and so is every integer argument below 2³¹, so a row base
`row * stride` is an int32 product and wraps once an address passes 2³¹ elements. Two
classes of product reach that inside the paper's scale ladder:

- **output / candidate axis**, `B·N ≥ 2³¹`: every `[B, N]`-shaped buffer a kernel indexes
  by batch row: the `clause_mask` mask, the compaction scratch and
  `[B, N]` result, the `[B, P]` candidate lists and score buffers of
  `fused_masked_knn_topk` and `oporp_1bit_match_topk` (B = 144 at N = 15M).
- **item axis**, `N·row_stride ≥ 2³¹`: item tables read by contiguous tile with a row
  stride above 1: OPORP bits at `N ≥ 2³¹/W` (1.07B at `W = 2`), clause attrs at
  `N ≥ 2³¹/(C·A_max)` (268M at `C·A_max = 8`).

**Rule.** Every kernel takes `WIDE: tl.constexpr`, set by
[`_host.wide`](../../retrieve/src/retrieve/ops/triton/_host.py) when any tensor the kernel
addresses holds ≥ 2³¹ elements. Batch-row bases go through `common.row_base(ptr, row,
stride, WIDE)`, item tiles through `common.tile_rows(ptr, row0, lane, stride, WIDE)`.
Under `WIDE` the scalar row product is int64, via `tl.assume(row, stride ≥ 0)`, so it
lowers to one `mul.wide.u32`. The lane offsets (`tl.arange`) stay int32, relative to that
base. Narrow, both helpers reproduce the int32 addressing the kernels always had.
Not widened: query rows (`B·D`, `B·W`, `B·C`) and tile-count rows (`B·N/BLOCK`), which
stay far below 2³¹ at any launchable `B`, and gathered item ids, which are loaded as
int64 already.

**Why a constexpr, not int64 everywhere.** Always-int64 was built first and measured slower: the scorer's 32-lane programs
pay a full 64-bit multiply per row base and the compaction kernels pay the registers the int64
bases cost. With the constexpr, launches below 2³¹ time within noise of the int32 kernels, and
only launches past 2³¹ pay.

**Bounds that remain.** `N`, `P` and every per-row count stay below 2³¹: the compaction
scratch holds ids as int32, and `tile_id · BLOCK` is int32. The batch-first grids cap `B`
at 65,535.

**Gate.**
[`tests/correctness/test_large_offsets.py`](../../retrieve/tests/correctness/test_large_offsets.py)
runs one case per class with a planted answer: `B = 144, N = 16M` through
`clause_mask`, `clause_compact`, `fused_masked_knn_topk` and
`oporp_1bit_match_topk_indirect`, and one `[140M, 16]` int64 table read as clause attrs and
OPORP bits. It is skipped below 48 / 24 GiB of free
device memory. Every widening is mutation-checked: without its int64 cast the case
raises an illegal address or returns wrong ids.

## Padding

`tl.arange` needs a power-of-two extent, and the datasets do not all have one (pubmed and
openalex D = 768; OPORP's default `k_bits = D` gives `W = 3` and `12`).
Every kernel with a lane axis over the embedding or word width takes the true width and
its `triton.next_power_of_2` as two constexprs (`D` / `D_PAD`, `W` / `W_PAD`), aranges over
the padded one and masks lanes `≥ D` (`≥ W`) out of both the query and the item load with
`other = 0`. The tables are not padded: no extra index memory or bandwidth, nothing at
`register_index`, the same stored layout on every backend.

Why a zero lane changes nothing, per kernel:

- **`fused_masked_knn_topk`** (fp32 `tl.sum` of products): each masked lane adds an exact
  `+0.0`. The reduction tree over `D_PAD` lanes differs from any tree over `D`, as the tree
  already differs from cuBLAS's `bmm`; the parity gate is the file's existing tolerance
  (`atol = 1e-6` on unit-norm data), which holds at D = 192 and 768.
- **`oporp_1bit_match_topk`**: masked words load `0` on both sides, `0 ^ 0` has no set bits,
  so the Hamming distance and `D_TOTAL - 2·hamming` (`D_TOTAL = 64·W`, the true width) are
  unchanged.


**At a power-of-two width the code is the same.** With `D == D_PAD` the mask is an all-true
constant, which Triton folds, so the power-of-two build is the pre-padding kernel: every kernel of
this section, compiled at the power-of-two widths the study runs, is SASS-identical to the tree
before padding.

## Shared kernel helpers (`ops/triton/common.py`)

[`ops/triton/common.py`](../../retrieve/src/retrieve/ops/triton/common.py)
holds the `@triton.jit` building blocks the kernel bodies call (Triton
inlines them). Every helper is a pure function of already-loaded tiles
or takes fully-resolved addressing from the caller — no helper decides
its own tile shape, launch grid, or masking policy:

- `row_base(ptr, row, stride, WIDE)` / `tile_rows(ptr, row0, lane, stride, WIDE)` — the
  row-base arithmetic of [Addressing](#addressing); every kernel's `[B, ·]` row and item
  tile goes through one of the two.
- `popcount_int64(x) → int32` — the hardware `POPC` (`libdevice.popc` on
  int64, `__nv_popcll`); used by `oporp_1bit_match_topk`. Its torch twin
  [`functional.py::popcount_int64`](../../retrieve/src/retrieve/functional.py)
  is SWAR; popcount is exact integer math, so the two agree bit for bit
  (see [Numerics](#numerics)).

- `clause_pass(item_attrs_ptr, is_reverse_ptr, query_attrs_ptr, ids,
  load_mask, bid, strides..., C, A_MAX) → [BLOCK] int1` — the exact
  AND-of-OR clause predicate (inner OR over `A_MAX` slots, outer AND
  over `C`, reverse XOR, `q_c == -1` inactive override), already ANDed
  with `load_mask`. **Inactive clauses** are not read: a clause the query
  leaves at `-1` passes whatever the item holds, so a uniform branch
  (`if q_c != -1`, one value per program) skips its loads and compares.
  Unrolled over every clause, the predicate paid `C · A_MAX` loads per item
  and batch row however few clauses the query used. The 10-clause synth
  tables made the V1 / V2 clause cells slower by the width, flat in p and
  growing with the batch, and cost `clause_compact` registers; with the
  branch it is back to its narrow-table count. **Known cost:** with every
  clause active the branch skips nothing and the kernel is a little slower
  than without it. Masked loads and 4 warps were measured and are not used.
  Used by `clause_mask` / `clause_compact`
  (`ids=n_offsets`, `load_mask=n_valid`).
  `is_reverse_ptr` must point at int8 storage (host preps do
  `.to(torch.int8)`; Triton can't load native torch.bool).
- `compact_store(pass_mask, ids, base, out_ptr, bid, ...)` — the
  stream-compaction store (`cumsum` intra-tile rank + masked store at
  the caller-supplied `base`, cast to the pointee type).
- `compact_stash(pass_mask, ids, tile_counts_ptr, scratch_ptr, bid,
  tile_id, ..., BLOCK_N)` — phase 1 of the compaction op: the
  tile's survivor count, and `compact_store` of its ids into the tile's
  own slot range `scratch[bid, tile_id * BLOCK_N :]` (int32).
- `compact_scatter_kernel` — the one *launched* kernel in this file:
  phase 3 of the compaction op, one program per `(row, tile)` on the
  predicate launch's grid, moving the tile's stashed run to the scanned
  row offset. Launched by
  [`_host.compact_finish`](../../retrieve/src/retrieve/ops/triton/_host.py),
  which also does the `cumsum`.


Perf caveat: helpers with many pointer params (`clause_pass`) can
perturb register allocation. The documented fallback if a kernel
regresses > 5% on the tune-kernels gate is to revert *that kernel body*
to the inlined predicate with a
`# keep in sync with ops/triton/common.py::clause_pass` breadcrumb.

## Score conventions

- Real-valued similarity (V1, V2): plain dot product of the fp16 inputs
  (items stored fp16, as the LiNR paper does; the query cast to fp16),
  accumulated in fp32 and **returned as fp32 on every path**: cuBLAS
  (V1, V2 dense, V2 `torch`) through `out_dtype=torch.float32`,
  `fused_masked_knn_topk` (V2 `triton`) by its fp32 `tl.sum`. On CPU
  (the harness suite's plumbing tests) `aten::mm.dtype` / `bmm.dtype`
  have no kernel, so those three call sites branch on `is_cuda` and
  multiply fp32 casts of the fp16 operands there; a cast-then-`mm` on
  CUDA is not bit-identical to `out_dtype=` (different reduction order,
  |Δ| ≈ 3e-5), so the CUDA path keeps `out_dtype=`. Higher is
  better. `-inf` marks masked-out / padded positions. An fp16 *score*
  would round runs of near-tied items to one value: on YFCC-10M the
  top-1000 spans about fifteen fp16 quanta (2⁻¹¹ near 0.8), so fp16 scores
  cost the exact algorithms recall at k 1000 that fp32 scores over the same
  fp16 table recover, up to the fp16 *storage* rounding
  The cuBLAS path's
  tensor-core accumulator truncates more than a plain fp32 sum; the bound is
  gated in [`test_accumulation.py`](../../retrieve/tests/parity/test_accumulation.py).
- 1-bit Sign-OPORP (V3): `D - 2 * popcount(query_bits ^ item_bits)`, fp32.
  `D = 64 * W`. This is the standard Hamming-to-dot-product relation for
  sign-quantized vectors. Higher is better. Same `-inf` sentinel.
- The **id** returned alongside a `-inf` score is always `-1`, on every
  backend — `interfaces.py` names `-1 / -inf` as the paired "no item"
  sentinels. Every torch-side path gets there through `masked_topk`; the adapted SilverTorch
  code's fused epilogue writes the same sentinels. So a caller can read either tensor to find
  the dead slots,
  and ids from two backends compare directly on rows with fewer than K
  survivors.

## OPORP layout

Used only by V3. Built once at index time by [`quantize_oporp_1bit`](../../retrieve/src/retrieve/indexing/quantize.py):

- `signs[D]` int8 ∈ {-1, +1} — Rademacher (random sign vector).
- `perm[D]` int64 — permutation of `[0, D)`.
- `item_bits[N, W]` int64, `W = D // 64`. Bit `b` of word `w` is set iff
  `(items * signs)[perm][..., 64*w + b] > 0`.

Queries land in the same bit space via [`project_oporp_1bit_query`](../../retrieve/src/retrieve/indexing/quantize.py):
`query_bits = pack_signs(((query * signs)[perm]) > 0)`.

The projection is **deterministic**: the seed determines `signs` and `perm`,
so loading the same checkpoint produces the same bits. At the default
`k_bits = D` each bin holds one coordinate, so the projection is a signed
permutation, and `popcount(q ^ x)` is invariant under it: the seed changes the
bits but never the Hamming score, apart from an exactly-0 coordinate, which packs
as bit 0 under either sign. V3's results are therefore the same at every seed.
At `k_bits < D` the bins sum several coordinates and the seed does change the
scores. V3 store (`item_bits`,
`signs`, `perm`) as buffers; both backends call the same projection helper
on every forward, so the torch reference and the Triton kernel see byte-for-
byte identical bits.

## PostfilterKNN dense path — cuBLAS, and `gemv_scores` for one query

`PostfilterKNN`'s forward is `torch.mm(query_fp16, item_embs_t,
out_dtype=torch.float32)` (fp32 operands on CPU, § Score conventions) + optional `masked_fill(-inf)` + `torch.topk`
over the fp32 scores (§ Score conventions) — implemented directly in
[`PostfilterKNN`](../../retrieve/src/retrieve/modules/knn.py). The one
exception is a single query on the triton backend, below.

### `gemv_scores` — the single-query GEMV ([file](../../retrieve/src/retrieve/ops/triton/gemv_scores.py))

At B = 1 cuBLAS picks its kernel by shape. At D 128 / 256 (0.8–3 M items)
that is `gemv2N`, a CUDA-core kernel at about half the memory bandwidth,
most of a LiNR V1 forward at 3 M d128. At D 768 / 10 M
and some other shapes it is a cutlass tensor-core GEMM. Its fp32
accumulation is one FMA per k, k = 0 … D−1 in order (found by trying
split, interleaved, FMA / mul-add orders against it).
`gemv_scores` runs that order with one item per lane and `BLOCK_N` = 1024
items a program. That gives `torch.equal` scores at about twice the speed
(goodreads 287 → 137 µs, arXiv d128 1041 → 478, d256 1886 → 937).

Because the kernel choice depends on (B, N, D), `register_index` checks
the GEMV against `torch.mm` on 8 item rows taken as queries, and sets
`gemv_exact`; a kernel's accumulation order does not depend on the data.
The triton backend's `score()` takes the GEMV only at B = 1 with
`gemv_exact`, and cuBLAS otherwise. So scores are cuBLAS's bit for bit
whatever the shape: PubMed d768 / 10 M keeps cuBLAS. `register_index`
first creates the cuBLAS handle with a 1×1 `mm`. The handle allocates
outside the caching allocator, and the check would otherwise make the
process's first cuBLAS call at the build's memory peak
(`CUBLAS_STATUS_NOT_INITIALIZED` on PubMed 10 M d768 in a two-layer
process). A module loaded from a state dict has `gemv_exact` False
(cuBLAS) until `register_index`. `interfaces.DISPATCH` keeps the label
`cublas`: the two backends' scores are still equal, so the harness's
collapse of `linr_v1` triton / torch stays valid.

## `fused_masked_knn_topk` — PrefilterKNN sparse path

[`ops/triton/fused_masked_knn_topk.py`](../../retrieve/src/retrieve/ops/triton/fused_masked_knn_topk.py).

Scores only the items in a precompacted `positive_indices` buffer (gather
+ dot + write). Returns `(ids[B, K], scores[B, K])` with `-1` / `-inf`
padding when fewer than `K` candidates pass.

```
inputs:   query             [B, D]    fp16 or fp32
          item_embs         [N, D]    fp16 or fp32
          positive_indices  [B, P]    int64
          counts            [B]       int64
output:   scores            [B, P]    fp32  (kernel-internal)
return:   ids               [B, K]    int64
          scores            [B, K]    fp32
```

(`_fmkt_prep` validates the dtypes: parity tests feed fp32, the
production `PrefilterKNN` path feeds fp16; scores are always fp32.)

**Accumulation is fp32 whatever the input dtype.** The kernel widens
`q` and `emb_rows` to fp32 before the multiply, so an fp16 × fp16 product
is exact and `tl.sum` reduces in fp32 (the compiled PTX: `add.f32` /
`fma.rn.f32`, no `f16` arithmetic). This has to be explicit: Triton's
`tl.sum` reduces in its *operand* dtype (`_pick_sum_dtype` promotes only
sub-32-bit ints). Reduced in fp16, on goodreads d128 (`|score|` up to 31,
partial sums of the same order) the error against an fp64 dot is large
enough to swap boundary pairs on a few percent of rows against the
`torch` backend. In fp32 it is ~1e-5; the `torch` side returns fp32 too
(§ Score conventions), so the two backends differ only in reduction order. The
accumulation-width parity file
[`test_accumulation.py`](../../retrieve/tests/parity/test_accumulation.py)
pins every scoring kernel against an fp64 oracle, because the other
parity files compare Triton against `ops.reference` at the same input
dtype and cannot see this class of defect.

**Launch grid** `(G, B)`, `G = min(cdiv(P, BLOCK_N), cdiv(programs, B))`,
tile programs on `grid_x`, batch on `grid_y` (≤ 65535). A program runs two
loops over its row. The scoring loop strides over the row's counted tiles
`g, g + G, …` below `cdiv(counts[b], BLOCK_N)`; each gathers `BLOCK_N`
item rows by indirect load:

```
item_ids[BLOCK_N] = pos_indices[b, p_off]
emb_rows[BLOCK_N, D] = item_embs[item_ids]
dots[BLOCK_N] = sum(emb_rows * q[None, :], axis=1)
```

The fill loop then stores `-inf` over `[cdiv(counts[b], BLOCK_N) ·
BLOCK_N, P)` in `FILL_N` = 1,024-lane chunks, strided over the same
programs, so a mostly empty row costs a few wide stores per program, not
one `BLOCK_N`-lane tile per iteration.

The grid is sized by the config, not by `P`. The compact family returns
full-width `[B, N]` candidates, so a grid of `cdiv(P, BLOCK_N)` one-tile
programs launched hundreds of thousands of 8-warp programs at goodreads N,
bs 16, nearly all only storing `-inf`: bound by program count, not bytes. The stride
does not change which program reduces a lane or how: each lane is still
one `tl.sum` over D in the same layout, so scores are bit-identical to
the one-tile grid.
`programs` is a field of the tile config. The default 864 is one wave on
A100 (108 SMs × 8 resident 8-warp programs); swept over {864, …, 13824},
it was the best or within 1 % of the best at every batch size and pass
rate measured, and at bs 16, p 1 fewer programs gather faster.

**Per-cell scoring is elementwise**, not `tl.dot`. Different `(b, p)` cells
gather different rows, so a true GEMM would re-load each row across the
block-of-queries axis. For LinR's cell shapes (small B, sparse P) the
elementwise reduction is bandwidth-bound and tensor cores wouldn't help.
If `P` ever grows large enough that bandwidth becomes saturated *and* the
gather is dense in id-space, this can revisit a tiled `tl.dot`.

**Counts handling**: positions past `counts[b]` get score `-inf` via
`tl.where(in_count, dots, -inf)`. The `pos_indices` load is also gated
by `mask=in_count` — lanes past `count[b]` are never dereferenced, so the
caller's `positive_indices` buffer doesn't need any padding past
`counts[b]` (and the host wrapper doesn't materialize a padded copy).
Positions past `P_real` (block tail) are masked at store time. The host
then `torch.topk(scores, min(k, P_real))` and gathers global ids from
`positive_indices`.

**Score buffer is uninitialized.** The host wrapper allocates `[B,
P_real]` via `torch.empty` — every in-bounds lane is overwritten by the
kernel (real dot or `-inf`), so the post-topk `where(isfinite(scores),
…, -1)` mask sees deterministic values without a `torch.full(-inf)`
pre-fill kernel launch.

**Wide D** (`SPLIT`, `D_PAD > 256`). Holding the query and the
`[BLOCK_N, D_PAD]` tile across the tile loop costs 98-112 registers a
thread at D 768: 2 resident 8-warp CTAs an SM instead of the straight-line
body's 4, and 1.43× the time of that body at p ≈ 1 (V2 at PubMed 10 M; the
grid size does not matter). Past 256 the scoring loop therefore reloads
the query per tile (an L1 hit). `WIDE_CONFIG` (`block_n` 8, `num_warps` 4,
`programs` 3456) then holds no spill and needs no register cap. The larger `programs`
covers skewed batches: PubMed `all5` puts most candidates in one or two
rows of a bs-16 batch, and at 864 programs (54 a row) such a row ran slower
than the one-loop body; at 3,456 it runs well under it, at a small cost on
a uniform p 1 batch. A `maxnreg` cap also worked eagerly, but inductor's
launcher drops it, so it was rejected. Each lane's
score is the same `tl.sum` over D, so scores are bit-identical to the
one-loop body.

**Count width** (`INT32_COUNT`, `D_PAD ≤ 128`). At `D_PAD ≤ 256` the
query is held across the scoring loop. The loop's bound comes from the
int64 `counts`, so its induction variable and every tile offset are 64-bit
unless the count is narrowed (`counts[b] ≤ P < 2³¹`, index arithmetic
only). Narrowed, the body is shorter and at `D_PAD` 64-128 compiles to
25-32 registers. At `D_PAD` 256 it compiles to 40, and so did the one-loop
body before it: 6 resident 8-warp programs an SM instead of 8, so the
default 864 programs take a second, one-third-full wave. Left 64-bit at
256, the same body compiles to 32 registers and fits one wave. So the count
is narrowed only at `D_PAD ≤ 128`. This rests on the compiler's register
allocation; a toolchain change should re-check the register counts.

**Tile config.** `FusedMaskedKnnTopkConfig(block_n, num_warps,
num_stages, programs)` — shipped as `DEFAULT_CONFIG` (`D_PAD ≤ 256`) and
`WIDE_CONFIG` (past it), picked by `config_for_width`; pass
`config=` to override. Re-tune on a new arch via `uv run tune-kernels
fused-masked-knn-topk` and paste the printed
`DEFAULT_CONFIG = ...` line.

**Bucketing.** The eager `_fused_masked_knn_topk_impl` (tune sweeps,
parity tests — callers that see many distinct widths per process)
rounds `P` up via `_bucket_p` to one of `{256, 2048, 16384, 131072,
1048576}` and passes it as `tl.constexpr`, so the JIT cache compiles
once per bucket × D regardless of how `counts.max()` shifts across
calls; it also early-returns full `-1`/`-inf` padding at `p == 0` and
pads short rows back out to `k` columns. The **public op skips all
three** (`bucket=False`, `pad_to_k=False` in the shared prep/finish):
candidate widths are static per deployment (the compact family returns
full-width `[B, N]`; linr_v3's stage-2 width is the fixed
`candidate_pool`, with `LiNRV3` rejecting `k > candidate_pool`), and `P < k`
raises `ValueError` on the op and its torch twin alike. `PrefilterKNN` pads a
short candidate list to `k` columns of `-1` (its lanes past `counts` are never
read), so the layer always returns `[B, k]`.
When bucketed, the score buffer is allocated at `[B, P_BUCKET]`; lanes
in `[P_real, P_BUCKET)` get `-inf` automatically because
`count[bid] <= P_real`, the caller-supplied `positive_indices` stays at
width `P_real`, and the post-topk `gather` clamps `topk_local` to
`P_real - 1` before indexing (the `where(isfinite, …, -1)` mask then
overwrites those slots with the `-1` sentinel; on the unbucketed public
op the clamp is an identity).

## `oporp_1bit_match_topk` — V3 (all paths)

[`ops/triton/oporp_1bit_match_topk.py`](../../retrieve/src/retrieve/ops/triton/oporp_1bit_match_topk.py).

Computes Hamming-similarity scores from packed sign bits. Single kernel
covers both the full-scan and the candidate / masked path via a
`HAS_INDICES: tl.constexpr` flag — they differ only in how items are
addressed.

```
inputs:   query_bits        [B, W]     int64
          item_bits         [N, W]     int64
          positive_indices  [B, P]     int64    (HAS_INDICES=True only)
          counts            [B]        int64    (HAS_INDICES=True only)
output:   scores            [B, n]     fp32     (n = N if full, P if indices)
return:   ids               [B, K]     int64
          scores            [B, K]     fp32
```

**Inner op**: per (b, n) cell, `tl.sum(_popcount_int64(qb ^ item_row),
axis=W) → hamming`, then `score = D_TOTAL - 2 * hamming`.

**Popcount** is the hardware `POPC`
([`common.popcount_int64`](../../retrieve/src/retrieve/ops/triton/common.py),
`libdevice.popc`). It replaced a five-step SWAR bit-twiddle, a clear win on
the indirect op and a small one on the full scan, where the top-k
dominates. The torch
reference [`popcount_int64`](../../retrieve/src/retrieve/functional.py) is
still SWAR, because this torch has no `bitwise_count`. Popcount is exact, so torch and Triton
produce **bit-exact** identical scores.

**Launch grid** `(G, B)`, tile programs on `grid_x`, batch on `grid_y`
(up to 65535). Full scan: `G = cdiv(N, BLOCK_N)`, one tile per program,
every lane real. Indirect: `G = min(cdiv(P, BLOCK_N), cdiv(programs,
B))` and a program strides over tiles `g, g + G, …`; a tile at or past
`counts[b]` stores `-inf` with no load. This is the same fix as
[`fused_masked_knn_topk`](#fused_masked_knn_topk--prefilterknn-sparse-path):
V3's filtered stage 1 gets full-width `[B, N]` candidates from the compact
family. Scores are integers computed by the same lane body, so the stride
is exact. `programs` defaults to 1728 here (4-warp programs); it was not
swept.

**HAS_INDICES path**: replaces the contiguous `n_off` load with an
indirect `pos_indices[b, n_off]` lookup; otherwise identical. Used for
candidate-set rerank and for the masked V3 path (after `compact_mask`).

**Tile config.** `Oporp1BitMatchTopkConfig(block_n, num_warps,
num_stages, programs)` — shipped as `DEFAULT_CONFIG` (`D_PAD ≤ 256`) and
`WIDE_CONFIG` (past it), picked by `config_for_width`; pass
`config=` to override. Re-tune on a new arch via `uv run tune-kernels
oporp-1bit-match-topk`.

**Bucketing.** The eager `_oporp_1bit_match_topk_impl` (tune sweeps,
parity tests) runs the `HAS_INDICES=True` path at score-buffer width
`n_kernel = _bucket_n(positive_indices.shape[1])` (buckets
`{4096, 65536, 1048576, 16777216}`), passed as `tl.constexpr N`: the
same JIT-cache invariant as `fused_masked_knn_topk`. The **public
indirect op does not bucket** (`bucket=False`, `n_kernel = P`), like
`fused_masked_knn_topk`'s: its candidate width is static per deployment,
and the bucket sized the buffer, the grid and the top-k by the ladder's
next rung. At arxiv N = 2,988,996 that rung is 16,777,216, 5.6 × N. The candidate path
returns **`min(k, P)` columns** (`torch.sym_min`, symbolic under
`dynamic=True`), with no pad. That is the convention of the torch twin and of
the other candidate-path layers, and `P = 0` gives an empty `[B, 0]`.
`positive_indices` stays at its caller width; the kernel's indirect
`pos_indices` load is gated by `in_count = (n_off < count[bid])` (not
`n_valid = (n_off < N)`) so it never reads OOB when `N > n_loop`. The
post-topk `safe_local.clamp_max(n_loop - 1)` + `where(isfinite(scores),
ids, -1)` tail handles the per-row "ran short" case (rows where
`counts[b] < min(k, P)` get `-1` sentinels in the bottom slots). For
`HAS_INDICES=False`, `N = item_bits.shape[0]` is fixed per registered
index — `OneBitKNN.register_index` asserts `k <= N`, no bucketing
needed.

## `clause_compact` — fused clause eval + stream compaction

[`ops/triton/clause_compact.py`](../../retrieve/src/retrieve/ops/triton/clause_compact.py).

Powers `ExactAttributeFilter.evaluate_indices`. Avoids materializing the
dense `[B, N]` bool that `evaluate_mask` would otherwise produce, then doing
a host-side argsort to compact it. One op call produces the
`(positive_indices, counts)` pair that `PrefilterKNN` and `OneBitKNN`'s
sparse paths consume.

```
inputs:   item_clause_attrs   [N, C, A_max]  int64
          clause_is_reverse   [C]            bool
          query_clause_attrs  [B, C]         int64
return:   positive_indices    [B, N]         int64  (full width, written on [:counts] only)
          counts              [B]            int64
```

The returned index buffer is **full-width** `[B, N]` — only the first
`counts[b]` entries per row are written; the tail is whatever the
`torch.empty` allocation held (no host-side `counts.max().item()` sync,
no narrow slice, no prefill and no tail store). **Every reader bounds
its reads by `counts`**, and none gathers through a tail id: the Triton
consumers load ids under `n < counts[b]`, the reference ops index through
`where(n < counts[b], id, 0)` and mask the rest to `-inf`, and the top-k
epilogues turn any `-inf` slot's id into `-1`. A path that takes padded
ids without `counts` (`FullScanKNN` / `SilverTorch` candidates) needs the
caller to mask the tail to `-1` first. So the op's full output is a
function of its inputs only under `torch.use_deterministic_algorithms(True)`,
which fills `torch.empty` (the tail then reads `2⁶³ − 1`); `opcheck` runs
in that mode. Pinned by
[`test_compact_order.py`](../../retrieve/tests/parity/test_compact_order.py)
(`test_consumers_ignore_the_tail_past_counts`: every consumer returns the
same tensors with the tail poisoned to `-1` and to an out-of-range id).

**Launch grid** `(B, tiles_y, tiles_x)` — batch on `grid_x` so adjacent
dispatched programs share the same item tile (good L2 reuse on the
`[N, C, A_MAX]` item-attrs read); the `cdiv(N, BLOCK_N)` tile count is
split across `grid_y × grid_z` to dodge the 65,535 cap on a single
axis (which would otherwise overflow at N>~16M with `block_n=256`). The
kernel reconstructs `tile_id = tile_x * tiles_y + tile_y`. Each program
owns one `(query, n-tile)` cell. The op is **two launches around a
scan**. The predicate launch's epilogue is `cumsum` intra-tile ranks and a
masked store into the tile's own fixed slot range in an int32 scratch,
plus a store of the tile's count; the scan and a
predicate-free second launch then move each tile's run to its row offset:

```
predicate launch (b, tile):                     _clause_compact_kernel
    pass_mask = AND over clauses of (any item attr == query attr) ^ reverse
    tile_counts[b, tile] = sum(pass_mask)                     # common.compact_stash
    intra = tl.cumsum(pass_mask) - 1
    tl.store(scratch[b, tile * BLOCK_N + intra], item_id, mask=pass_mask)
host, inside the op:                            _host.compact_finish
    tile_ends    = tile_counts.cumsum(1)          # [B, T] int64, T = grid tiles
    tile_offsets = tile_ends - tile_counts        # exclusive scan
    counts       = tile_ends[:, -1].clone()
scatter launch (b, tile), same grid:            common.compact_scatter_kernel
    ids = scratch[b, tile * BLOCK_N : +tile_counts[b, tile]]
    positive_indices[b, tile_offsets[b, tile] : +count] = ids   # nothing past counts[b]
```

Why this shape and not the two alternatives (re-evaluate the predicate in
the second launch, or a packed bitmask): both were built and measured.
Re-evaluation costs a full second pass of the predicate. The bitmask — and
even a bare count-only epilogue — makes the *predicate* launch far slower
at `B = 1`, where the grid is under one wave and per-program latency is the
kernel time; the `cumsum` + masked-store epilogue keeps it near the one-pass
speed. The scratch traffic is the survivors only (4 B in, 8 B out per id),
and so is the output's: with no tail store the scatter writes nothing for a
rejected item. Its allocation is
`4 · B · T · BLOCK_N` bytes, half the `[B, N]` int64 result. `counts` is a
fresh tensor, not a view into the scan: inductor asserts custom-op outputs
are 16-byte aligned, and at `B = 1` the scan's last column is a contiguous
view at element offset `T - 1` (pinned by
`test_compiled_batch_of_one_v2`).

**Clause loop** is the shared
[`common.clause_pass`](../../retrieve/src/retrieve/ops/triton/common.py)
helper, fully unrolled (`C` and `A_MAX` are `tl.constexpr`): inner OR
over the `A_MAX` attribute slots per clause, outer AND over the `C`
clauses, with the reverse flag XORed in per-clause and `q_c == -1`
overriding to "always passes." The tile's count and id run go out
through `common.compact_stash`; `common.compact_scatter_kernel` moves the
run to the scanned base.

**Output ordering** within a row is **ascending item order** — the order
`ops.reference.clause_compact` (`compact_mask`'s stable argsort) emits, so
the two backends agree `torch.equal` on ids and counts, and a call
reproduces itself launch after launch and process after process
([`test_compact_order.py`](../../retrieve/tests/parity/test_compact_order.py)).
The order is fixed on purpose: a
row base claimed with `tl.atomic_add` orders a row by tile completion,
which `PrefilterKNN`'s top-k and `OneBitKNN`'s heavily tied Hamming
ranking turn into run-to-run quality noise. The price is the scan, the
stash round trip and two extra
launches.

**Op registration** is `@torch.library.custom_op`, not `@triton_op` — this
kernel is the one exception in the tree. The
data-dependent `base + intra` store address above is exactly what makes
inductor's TTIR mutation analysis report the index buffers as mutated, which
makes cudagraph trees skip the compiled forward; the "Graph-break behavior"
bullet in the conventions list at the top of this file has the full mechanism
and names the regression test.

**Tile config.** `ClauseCompactConfig(block_n, num_warps, num_stages)`
— shipped as `DEFAULT_CONFIG` on the kernel module (not tuned for the
two-phase shape, not retuned); tests/tuner
override via `_clause_compact_impl(..., config=)`. Re-tune on a new arch via
`uv run tune-kernels clause-compact`.

## `clause_mask` — fused clause eval emitting `[B, N]` bool

[`ops/triton/clause_mask.py`](../../retrieve/src/retrieve/ops/triton/clause_mask.py).

**`clause_mask_scores(scores, item_clause_attrs, clause_is_reverse, query_clause_attrs)`** is
the same kernel with `HAS_SCORES`: it loads the `[B, N]` fp32 score at the lane it just tested
and stores `score` or `-inf` into a new fp32 buffer, instead of the bool. `LiNRV1` calls it
(through `ExactAttributeFilter.mask_scores`) on the cuBLAS scores, so the old mask kernel →
`[B, N]` bool → `torch.where` chain is one pass, the bool never reaches memory, and `topk` gets
the bit-identical tensor (ids and scores `torch.equal` to the mask + `where` path). It is
functional, not in place: under `torch.compile` inductor has no mutation to turn into a copy of
the score matrix (a profile of V1's graph showed that copy). The score matrix stays cuBLAS's,
because any other fp32 reduction re-orders the D products and flips k-th-place ties.

Powers `ExactAttributeFilter.evaluate_mask` on CUDA. Same inner loop as
`clause_compact` minus the count → scan → write compaction — emits the
`[B, N]` bool directly, without the `[B, N, C, A_max]` bool (`C·A_max`×
the output mask) a pure-torch broadcast materializes before reducing.

```
inputs:   item_clause_attrs   [N, C, A_max]  int64
          clause_is_reverse   [C]            bool
          query_clause_attrs  [B, C]         int64
return:   mask                [B, N]         bool
```

**Launch grid** `(B, tiles_y, tiles_x)`, identical shape to
`clause_compact`. Each program loads `query_clause_attrs[b, :]` once and
reduces over the `C × A_max` clause-attribute grid in registers. No
`[B, N, C, A_max]` intermediate ever materializes.

**Inner loop** is the same shared `common.clause_pass` helper as
`clause_compact`'s: inner OR over `A_MAX` slots, outer AND over `C`
clauses, reverse XOR, inactive override. The epilogue is a single
`tl.store` of the `pass_mask` tile — no cumsum, no atomics.

**Batched rows** (`_clause_mask_scores_batched_kernel`, `clause_mask_scores`
at B > 1). The per-row kernel is one program per (row, item tile). Each
row reloads the tile's int64 `[BLOCK_N, C, A_max]` attrs and redoes the
compares, so a selective multi-clause batch cost B× the attr traffic:
LiNR V1 `all4` bs 16, mask 1336 µs of a 2.0 ms forward on goodreads and
5226 of 7.4 ms on arXiv. The batched kernel is one program per item tile
and `BLOCK_B` = 16 rows. It loads each clause's attrs once, compares them
with every row's value (`[BLOCK_B, BLOCK_N]`), skips a clause no row of
the block queries, and writes each row's masked scores. The predicate is
the same boolean, so the output is `torch.equal`. Mask: goodreads `all4`
bs 16 1336 → 253 µs, arXiv 5226 → 877 µs. At B = 1 the per-row kernel
runs unchanged.

**Tile config.** `ClauseMaskConfig(block_n, num_warps, num_stages)` —
shipped as `DEFAULT_CONFIG` (the bool form) and `SCORES_CONFIG` (the
`clause_mask_scores` form, 128 lanes on 4 warps: the bool tile ran the
fp32 form far slower) on
the kernel module; tests/tuner override via `_clause_mask_impl(...,
config=, scores=)`. Re-tune on a new arch via
`uv run tune-kernels clause-mask`.

## Layer dispatch

Each LinR Triton subclass is a thin router from the `forward()` signature
to one of the three kernels above. The dispatch is **design-time** —
`PostfilterKNN` is always dense, `PrefilterKNN` is always sparse,
`OneBitKNN` always uses popcount — not a runtime sparsity heuristic. Pick
the variant that matches your expected mask shape, not the one that benches
best on a given input.

Each LinR class accepts `backend="torch" | "triton"` on `__init__`; the
table below describes the `backend="triton"` path. With `backend="torch"`
each module runs the same op chain in pure torch (no kernels), eager.

| layer                       | path         | kernel(s) called                                  |
|-----------------------------|--------------|---------------------------------------------------|
| [`PostfilterKNN`](../../retrieve/src/retrieve/modules/knn.py) | always dense | none — pure torch `(q @ x.T)` + `masked_topk` |
| [`PrefilterKNN`](../../retrieve/src/retrieve/modules/knn.py)         | candidates   | `fused_masked_knn_topk` over caller-precompacted `(candidate_ids, counts)` |
|                                                                | unmasked     | none — pure torch dense path (nothing to pre-filter) |
| [`OneBitKNN`](../../retrieve/src/retrieve/modules/bit_knn.py) / [`SimHashKNN`](../../retrieve/src/retrieve/modules/bit_knn.py) | full         | `oporp_1bit_match_topk_full` (HAS_INDICES=False)       |
|                                                                | candidates   | `oporp_1bit_match_topk_indirect` (HAS_INDICES=True)        |

None of the layers takes a raw mask on these kernel paths — callers
with a `[B, N]` bool mask compact it first (`compact_mask`, or a
filter's `evaluate_indices`) and pass `(candidate_ids, counts)`. For
the bit-KNNs that trade is always right: popcount is cheap enough that
the gather penalty never crosses the dense-fallback break-even point.
`PostfilterKNN`'s dense path keeps the inline mask (no compaction)
because cuBLAS + `torch.topk` already handle it at the same cost a
tile-fused kernel would.

## Helpers

### [`compact_mask`](../../retrieve/src/retrieve/functional.py) (`retrieve.functional`)

Bool `[B, N]` → `(positive_indices[B, N], counts[B])` — **full width**,
the same contract as the triton `clause_compact` kernel, so torch and triton paths stay
interchangeable. Implementation:
`mask.sum(1)` for counts, then
`mask.float().argsort(descending=True, stable=True)` for the indices —
no `.item()` sync, no narrow slice. The tails differ across
implementations (arbitrary argsort-tail ids here, unwritten memory in
the kernel), so consumers must bound reads by `counts` either way.

### [`popcount_int64`](../../retrieve/src/retrieve/functional.py) (`retrieve.functional`)

Torch-side SWAR popcount. Required because this PyTorch (2.10.0+cu128)
lacks `Tensor.bitwise_count`. Returns int32 to keep the downstream sum
narrow. The kernel side uses the hardware `POPC`; both are exact, so the
OPORP/SimHash parity tests stay bit-exact.

### [`quantize_int8`](../../retrieve/src/retrieve/indexing/quantize.py) (`retrieve.indexing`)

Per-row symmetric int8 of the queries, called at query time by the candidate re-rank and the
adapter's reference path; `search`'s prep kernel computes the same codes and scales
([cuda](#cuda--the-adapted-forward)). Codes are `round(x / abs_max * 127)`, so an
element whose quotient sits on a half-integer flips a code when the
division is off by one ulp. Eager uses PyTorch's correctly rounded fp32
division; under `torch.compile`, inductor emits Triton's `/`, which lowers to
the approximate `div.full.f32` (≤ 2 ulp). So the code quotient `x / abs_max` is
computed in fp64 and rounded to fp32, in eager and compiled alike (no branch:
inside a `triton_op` inductor traces the implementation with make_fx, where
`torch.compiler.is_compiling()` is false). fp64 has 53 ≥ 2·24 + 2 mantissa bits,
so that double rounding returns the correctly rounded fp32 quotient, and Triton's
fp64 `/` is correctly rounded: graph-mode codes equal eager's bit for bit, and
eager's codes equal the pre-fix fp32 ones. Cost: two extra elementwise casts in eager.
The scales stay `abs_max / 127.0`: eager PyTorch evaluates a division by a
Python scalar as a multiply by fp32(1/127), and inductor emits the same multiply,
so they already agree (a correctly rounded division would not).

### [`quantize_oporp_1bit`](../../retrieve/src/retrieve/indexing/quantize.py) (`retrieve.indexing`)

Build-time only. `O(D)` parameter cost — the `signs` vector and `perm`
permutation are the entire projection. Apply with one elementwise multiply
and one `index_select`, no matmul needed. The `_pack_signs_to_int64`
helper packs the sign-quantized output into `[..., W]` int64 words using
`<<` and `sum(-1)`; same packing used both at index time and at query time.

**Chunked build.** Every step of the chain (sign-flip, permute, bin, row
norm, pack) is row-local, so `quantize_oporp_1bit` and
`quantize_simhash_1bit` (`x @ R^T` then the same pack) run it
`_CODE_CHUNK_ROWS` = 65,536 rows at a time into a preallocated `[N, W]`
int64 output (`_pack_chunked`, the shape of `quantize_int8_global`). The
one-shot chain held about seven full-corpus fp32/int64 temporaries, about
200 GiB at pubmed's 10M × 768 for a 0.9 GiB output, which is why
`LiNRV3` could not build there; the transient is now about eight projected
chunks (`[65,536, k_bits]` fp32, 192 MiB at 768 bits). The bits are
`torch.equal` to the one-shot projection (`test_quantize.py`, and on the
PubMed tables).
The query side (`project_*_query`) stays one-shot: `B` rows.

### KMeans centroid update

[`KMeans._cluster_sums`](../../retrieve/src/retrieve/indexing/kmeans.py) computes the
per-cluster sums with a reduction order fixed by the shapes, so two identical builds are bit
for bit equal. The obvious `torch.zeros(n_lists, D).index_add_(0, assignments, embs)` is not
reproducible on CUDA: `index_add_` reduces with float atomics, a cluster's summation order
follows block scheduling, and one assignment flipped early cascades through the rest. Instead
`bincount` gives the counts (integer atomics are exact in any order), and the sums are a
one-hot GEMM `onehot[n_lists, W] @ embs[W, D]` in float64 over panels of `W` items,
accumulated into `sums` in panel order (`addmm_`, issued sequentially). A GEMM's reduction
order is a function of its shapes alone. float64 keeps the accumulation exact well past
float32 and cannot be demoted to TF32 by a caller's
`torch.set_float32_matmul_precision("high")` (which would cost 13 mantissa bits); it costs
8 % more time on the A100, where DMMA runs at the float32 SIMT rate. The extra matmul per
Lloyd iteration has the FLOPs of the `cdist` assignment step, so `fit` takes at most ~2x the
`index_add_` version's time (measured well under that). The panel
width is capped so the one-hot buffer stays at 64 MiB whatever `n_lists` is.

### Compile on the V3 torch reference

The pure-torch hot bodies on V3's reference path —
`project_oporp_1bit_query`
([`indexing/quantize.py`](../../retrieve/src/retrieve/indexing/quantize.py))
and the reference op's `_hamming_scores`
([`ops/reference/oporp_1bit_match_topk.py`](../../retrieve/src/retrieve/ops/reference/oporp_1bit_match_topk.py)) —
are pure tensor-flow free of `.item()` and Python control flow, so the
harness's `torch.compile(mode="reduce-overhead", dynamic=False,
fullgraph=True)` of each algo (`evaluation/bench/measure.py`) traces them
into its cudagraph capture cleanly:

- **`project_oporp_1bit_query`** ([`quantize.py`](../../retrieve/src/retrieve/indexing/quantize.py)).
  Per-query OPORP projection: `multiply → index_select → sign-pack`
  (~5 small kernels in eager). Under the outer cudagraph_trees the
  launch tax collapses, a measured multiple at small batches.
- **`_hamming_scores`** ([`ops/reference/oporp_1bit_match_topk.py`](../../retrieve/src/retrieve/ops/reference/oporp_1bit_match_topk.py)).
  `xor → popcount → reduce` over the full corpus. The win here is
  *fusion*, not launch elision: eager materializes the `[B, N, W]` xor
  once and re-streams it through six SWAR popcount ops; Inductor fuses
  those into a single elementwise triton kernel. Measured ~3× at B=8,
  ~43× at B=64, N=50k — by far the largest compile win in the repo.
  ``d_total`` is derived from `item_bits.shape[1]` inside the body so it
  stays symbolic under `dynamic=True` (one graph across all `(B, N, W)`).

The matmul-bearing references (`PostfilterKNN`, `FullScanKNN`) have no
dedicated compile wrappers: cuBLAS + CUB already win the heavy op, and
the cudagraph capture + mandatory output clone (to escape the
`reduce-overhead` buffer pool) cost more than they save.

## Numerics

V1's pure-torch dense path uses cuBLAS for the matmul, so the torch and
Triton-backend classes go through identical kernels and produce
bit-identical fp32 scores. Against V2's kernel the two differ only in
reduction order (cuBLAS's truncating tensor-core accumulator vs an fp32
`tl.sum`), so the cross-module and cross-backend tests in
[`test_linr.py`](../../retrieve/tests/correctness/test_linr.py) compare
at `atol=2e-6`. V2's sparse path scores per-cell with
`tl.sum(emb_rows * q[None, :], axis=1)` — elementwise multiply +
reduction, not `tl.dot` — so it doesn't share the tensor-core
tile-reduction order quirks. Its parity test uses
[`assert_topk_matches`](../../retrieve/tests/parity/conftest.py) at
`atol=1e-6`: the kernel's `tl.sum` and the reference's `bmm` reduce in
different orders (measured drift ≤ 6e-8 at D ≤ 128).

**OPORP popcount is bit-exact.** V3's torch reference (SWAR) and the
Triton kernel (hardware `POPC`) count the same packed bits exactly, so torch
and Triton scores agree exactly, and
[`test_oporp_1bit_match_topk.py`](../../retrieve/tests/parity/test_oporp_1bit_match_topk.py)
pins it with `assert_topk_equal` (`torch.equal` scores, ids up to ties).
A divergence means a popcount or packing bug.

**The SilverTorch dequant is one expression in three places.** The bit-exact contract between
the torch oracle, the adapter's reference path and `search` rests on the left-associated
`dot.float() * q_scale * global_scale` being written the same way at every site that computes it:
the adapter's epilogue `_dequant` ([`ops/official/adapter.py`](../../retrieve/src/retrieve/ops/official/adapter.py)),
`search`'s fused epilogue (`fork_search_cuda.cu`, the same fp32 multiplies in the same
order) and `SilverTorch._forward_candidates`
([`modules/silvertorch.py`](../../retrieve/src/retrieve/modules/silvertorch.py)). Reassociating
any one of them (`dot * (q_scale * global_scale)`) changes the last bit and breaks `torch.equal`
in `test_official.py`; it is enforced by those tests, not by the code. Meta's `fused_kmean_ann`
int8 kernels accumulate with `IDP.4A.S8.S8` into an exact int32 (`cuobjdump -sass` of the pinned
build), so the dot itself is exact.

## Top-k

`torch.topk` over a `[B, P]` score buffer costs ∝ `B · P`: four 8-bit radix passes and a gather,
though on a filtered row most slots are `-inf`. From `functional.TOPK_MIN_SLOTS` = 2²³ (8.4 M)
slots, where a row has at least `TOPK_MIN_RATIO` = 8 blocks of `TOPK_BLOCK` = 256 slots per top-k
slot, [`functional.two_level_topk`](../../retrieve/src/retrieve/functional.py) runs the top-k in
two levels:

1. each whole block's max (one `amax` over a `[B, blocks, 256]` view, a single read of the
   buffer);
2. `torch.topk` over the maxima, picking each row's k blocks (unsorted);
3. a gather of those blocks' contiguous slots, plus the fewer than 256 slots past the last whole
   block (the ragged tail, always candidates);
4. the final `torch.topk` over the candidates.

**Why it is exact.** If an item x is in the row's top-k, its block's max is at least x. If that
block were not among the k largest maxima, k other blocks would each hold an item at least as
large as x. So the candidates hold the top-k multiset: scores are `torch.equal` to the whole-width
top-k, and ids differ only within the group tied at the k-th score. Rows with fewer than k finite
items fill with `-inf` / `-1` as before.

`masked_topk` routes every torch-side top-k through it, so LiNR V1 takes it over its unpadded
`[B, N]` scores, well under the plain top-k's time on the small datasets and a little under it
on PubMed. Below the thresholds it is plain
`torch.topk`. `SilverTorch` has the same
two-level cut and its own radix select inside `search`
([cuda](#cuda--the-adapted-forward)).

## SilverTorch on Meta's ops

`SilverTorch` runs phase 1 of Algorithm 1 in torch (k-means, the cluster-sorted CSR layout, int8
quantization, the centroid top-`n_probe`) and phases 2+3 on Meta's SilverTorch code, adapted:
`backend="cuda"`, its only backend, the `silvertorch/` workspace member
([CHANGES](../../silvertorch/CHANGES.md); [§ cuda](#cuda--the-adapted-forward)). The layer
registers the index (`indexing.ivf.csr_layout`): `item_codes` in cluster-sorted order,
`cluster_offsets [n_lists + 1]`, `cluster_sizes`, `sort_perm` (sorted position → original id) and
`inv_perm`, with the filter buffers in the same sorted order. A query row's probed clusters are
laid out back to back, so the scorer's output is `[B, width]` with `width =
indexing.ivf.probe_width(cluster_sizes, n_probe)`, the sum of the `n_probe` largest clusters: a
static bound on any row's item count, cached as the Python int `SilverTorch._probe_width` and
rederived by `set_query_params` and by the load hook. `register_index` / `set_query_params` raise
unless `width ≥ k`.

### Meta's ops

[`ops/official/`](../../retrieve/src/retrieve/ops/official/adapter.py) (`__init__.py`: loader,
`OfficialConfig`, the upstream constants and `st = torch.ops.st`; `adapter.py`: the rest) is an
adapter over the ops of Meta's [meta-recsys/silvertorch](https://github.com/meta-recsys/silvertorch)
(vendored at `21aa35e`) as the `silvertorch/` member carries them: the six upstream-shaped ops
below, called unchanged by the gates' reference path (`official_probe_score`,
`official_scores_full`, `fused_scores`, the bloom searches over CPU plans), and the adapted
forward's own ops (`search`, `probe_prep`, the `*_prepared` searches, `bloom_index_build_fixed`;
[§ cuda](#cuda--the-adapted-forward)). The upstream facts in this section were measured on Meta's
code as published; they hold for the adapted
ops wherever § cuda does not say otherwise, and the parity gate of the adapted code against the
published code held bit-exact at int32 up to the tag `meta-published-parity-final`, where the
published code left the workspace.

**Layout.** Phase 1 is ours: the same `KMeans`, the same `quantize_int8_global` codes, the same
centroid top-`n_probe`. Meta's scorer reads the cluster-sorted CSR the layer registers
([above](#silvertorch-on-metas-ops)). It takes `cluster_ids = probe_ids [B, n_probe]`,
`cluster_length = cluster_sizes[probe_ids]` and `max_tensor_size_per_row` equal to the compact
probe width `_probe_width`. Its output is therefore `[B, width]` (rounded up to 32). Returned
indices are sorted positions and map back through `sort_perm`. Slots the op never writes (pads,
filtered docs) carry `indices == -1` and become `-inf` / `-1`.

**Ops called**, with their schemas as registered upstream
(`TORCH_LIBRARY_FRAGMENT(st, …)` at `21aa35e`; the member registers them under the same strings):

- `fused_kmean_ann(Tensor cluster_offsets, Tensor cluster_ids, Tensor cluster_length,
  Tensor embeddings, Tensor queries, int max_tensor_size_per_row,
  Tensor? filtering_bit_mask=None, int invalid_index_value=-1, int divisor_for_int8=-1,
  Tensor? filtering_bit_index=None, Tensor? per_embedding_scale=None) -> (Tensor,
  Tensor)` — `fused_kmean_ann.cpp:400-413`. `cluster_offsets` int64 `[n_lists+1]`,
  `cluster_ids` / `cluster_length` int64 `[B, P]`, `embeddings` int8 `[N, D]`
  **cluster-sorted**, `queries` int8 `[B, D]`; returns `scores` (int32 when
  `divisor_for_int8 == -1`, fp16 otherwise) and `indices` int32, both `[B,
  round_up(max_tensor_size_per_row, 32)]`, pads at the dtype minimum /
  `invalid_index_value`.
- `fused_kmean_ann_with_partial_masks(…same five tensors…, int max_tensor_size_per_row,
  Tensor partial_mask_column_counts_cumsum, Tensor partial_mask_first_item_offset_in_column,
  Tensor partial_mask_column_results, int invalid_index_value=-1, int divisor_for_int8=-1,
  Tensor? filtering_bit_index=None, Tensor? per_embedding_scale=None, Tensor?
  cluster_warp_size=None, Tensor? cluster_warp_rounded_length_cumsum=None, Tensor?
  cluster_remaining_length_cumsum=None, Tensor? cluster_warp_size_cumsum=None, int
  total_cluster_rounded_warps=0, int total_cluster_remaining_warps=0) -> (Tensor,
  Tensor)` — `fused_kmean_ann.cpp:415-436`.
- `bloom_index_build(Tensor feature_ids, Tensor feature_offsets, Tensor feature_values,
  float b_multiplier, int k, bool fast_build=False) -> (Tensor, Tensor)` —
  `bloom_indexer.cpp:115-125`; int32 `[F]`, int64 `[N·F+1]`, int64 `[nnz]`,
  `b_multiplier > 1.0`; returns `bloom_index` int64 `[W]` and
  `bundle_b_offsets` int64 `[n_bundles+1]`.
- `parse_expression_query_batch(str[] expressions, Tensor silvertorch_ks, int
  bloom_hash_k, bool return_query_plan=True, int max_sub_queries=5) -> (int, Tensor[])`
  — `expression_query_parser.cpp:456-465`; CPU-only (the CUDA registration forwards
  to it); `silvertorch_ks` is unused upstream; returns `(max_stack_size,
  [plans_data int8, plans_offsets int64])`.
- `bloom_index_search_batch(Tensor bloom_index, Tensor bloom_bundle_b_offsets, Tensor
  bloom_query_plans_data, Tensor bloom_query_plans_offsets, int k, int hash_k, bool
  return_bool_mask=True) -> Tensor` — `bloom_index_search.cpp:549-558`; bool `[B,
  n_bundles·2048]` or packed int64 `[B, n_bundles·32]`.
- `bloom_index_search_batch_return_partial_response(…same four…, Tensor
  selected_cluster_offsets, Tensor selected_cluster_lengths, int k, int hash_k, Tensor?
  query_plan_index=None) -> (Tensor, Tensor, Tensor)` — `bloom_index_search.cpp:560-571`;
  int64 `[B, P]` doc-start / length per probe; returns `column_counts_cumsum` int32
  `[B·P]`, `first_item_offset_in_column` int8 `[B·P]`, `column_mask_response`
  int64 `[Σ columns]`.

Nothing else upstream is used by the library — no `BloomIndexSearchModule` (Meta's own
modules are re-exported for users by `retrieve.modules.official`), no `is_topk`, no
`*_multiple` sharded variants.

**Scores** (`OfficialConfig.score_path`). `"int32"`:
`divisor_for_int8=-1` returns the raw int32 dot; the epilogue
`(dot.float() · q_scale[b]) · global_scale` is the same two
left-associated fp32 multiplies as the torch oracle and as `search`'s fused epilogue, so scores are
**bit-identical** (the parity contract, [Numerics](#numerics)). `"fp16"`
(default — the instantiation Meta ships for int8 serving, hence the
timed path): `divisor_for_int8 = default_divisor(D)`, the smallest power
of two with `127²·D / divisor ≤ 65504` (16 / 32 / 64 at D = 64 / 128 /
256); the kernel writes `fp16(dot / divisor)` — exact division, one fp16
rounding (relative ≤ 2⁻¹¹) — and the epilogue multiplies by `divisor ·
q_scale · global_scale`. `per_embedding_scale` is not used: upstream
casts the int32 dot to fp16 *before* dividing by it, which overflows for
any realistic `D`.

**Filter modes.** `none` → the scorer alone. `exact` → the clauses tested inside `search` over
the probed docs ([§ cuda](#cuda--the-adapted-forward)); the gates' reference path packs our
`clause_mask` into the scorer's `filtering_bit_mask` instead (int64 `[B, ceil(N/64)]`, doc `d`
at bit `63 − d % 64` of word `d // 64` — `MASK_BIT_ORDER`, see below), the full-`N` form
`fused_kmean_ann` reads. `bloom` → **Meta's bloom**, not ours: at `register_index`,
`attrs_to_features` turns the sorted `[N, C, A_max]` attrs into the jagged `(feature_ids int32
[C], feature_offsets int64 [N·C+1], feature_values int64)` layout — the clause index is the
feature id — and `bloom_index_build(b_multiplier, k)` (or `bloom_index_build_fixed` at the
configured width, § cuda) builds `bloom_index [W]` + `bundle_b_offsets`; in `prepare_queries`
(once per batch, outside `forward`), `queries_to_expressions` renders each `[C]` query row as
`"0:v0 AND 1:v1"` (`NOT c:v` for reverse clauses, `""` = match all) and `parse_plans` runs Meta's
CPU parser. Meta's searches decode the plans on the host per call
(`bloom_index_search_cuda.cu:1045`), so on the reference path the plans stay on the CPU; the
forward uploads them once instead (§ cuda). The search `k` is our `k_hash` (must be ≤ 10:
`MAX_K_V2`, a fixed array size the upstream kernel never checks), `OfficialConfig.n_stored_hashes`
(default 7; the ops' `hash_k` argument, renamed because `k_hash` is the library-wide name of the
search `k`) is the number of raw hashes stored per term, `build_k` defaults to the search `k`. A
bloom index registered without attributes is empty and a later query with attributes raises.
Meta's partial-response search over the probed clusters feeding
`fused_kmean_ann_with_partial_masks` (the paper's co-design) and its full-`N` search feeding
`fused_kmean_ann(filtering_bit_mask=…)` return the same scores (T5); the forward evaluates the
bloom over the probed clusters only (§ cuda), and the partial-vs-full ablation (S9 / C5) ran on
Meta's code as published (records `official`, `params.bloom_path`).

**Bit order — measured, high-first.** Meta's scorer reads mask
words high-first ("lower doc id put at higher bits",
`bloom_index_util.cuh`), and the packed bloom output is stored the same
way; both are constants in the adapter (`MASK_BIT_ORDER`,
`BLOOM_OUTPUT_BIT_ORDER`) and `bloom_filtering_mask` bit-reverses per
word if they ever differ. The scorer's order is measured on the A100
three independent ways — the packed
search output for a doc-0 predicate is `0x8000000000000000`, a
hand-built mask with bit 63 set scores doc 0 (bit 0 scores doc 63), and
the packed output round-trips as a scorer mask — so both constants are
`"high_first"`. `test_official.py` pins that value as
`OFFICIAL_BIT_ORDER` and T3 asserts the adapter constants equal it,
with the other order kept as a negative control (nothing scored).

**Host syncs in Meta's code as published.** Every published op syncs the host
(`repeat_interleave` without an output size, `.item()` on cumsums, a per-call host decode and
pageable upload of the plans), and the partial-response output shape is data-dependent, so the
published code had no `torch.compile` or CUDA-graph path. Measured on the A100
(`torch.profiler` + fd-2 sync counting): `fused_kmean_ann` was
**19 launches and 3 host syncs** per unfiltered forward (only 2 of the
19 the scoring kernels; the rest payload prep — scans, a
`repeat_interleave`, fills), `fused_kmean_ann_with_partial_masks` 19 / 4,
the partial-response bloom search 13 / 2 with two H2D plan uploads, so a
bloom forward ≈ 32 launches and ≥ 5 syncs. The adapted code's forward has none
([§ cuda](#cuda--the-adapted-forward), C3 / C4: 9-12 launches). The sync sites, in the
upstream-shaped ops the gates still call: `fused_kmean_ann` reads `remaining_docs` back
(`fused_kmean_ann_cuda.cu:465`), the partial scorer its warp totals (`:2054, :2072`), the partial
search its chunk sums (`bloom_index_search_cuda.cu:1495`), both searches decode the plans on the
host per call (`:1045`); C3 / C4 removed them from the member's build. The CPU expression parse
(≈ 59 µs per call at B=16) is query preparation: it runs in `prepare_queries`, outside the timed
forward, and the harness records it as `query_prep_ms`
([evaluation](evaluation.md#query-preparation)). Meta's released `setup.py` compiles the host side
of its `.cu` files at gcc `-O0` (OF-11); the member's compiles at `-O3`, bit-identical and no
slower.

**Epilogue (OFFICIAL-REWORK).** Everything query-side runs in `prepare_queries`, outside the
timed forward: bloom → the expressions rendered and parsed, one plan per distinct expression plus
a device row → plan index (`query_plan_index`, which Meta's partial search accepts); exact → the
int32 query attrs. The forward is the probe (centroid matmul + `topk`), Meta's search and scorer,
then a top-k on the scorer's raw `[B, M]` (int32 dots or `fp16(dot / divisor)`, pads at the
dtype's minimum) with only the `[B, k]` winners dequantized and mapped through `sort_perm`
(`official_probe_score`, the reference path; fused into `search` on the forward). Dequantizing
multiplies by positive scales, so the winners and their scores are bit-identical to dequantizing
every slot first, ties aside. Upstream has no top-k inside the scorer: its tests sort the compact
output themselves. `is_topk` is a separate op, a full row sort plus a `≥ kth` membership mask
whose ties exceed k, so it is not used.

### `cuda` — the adapted forward

[`silvertorch/`](../../silvertorch/CHANGES.md) is a uv workspace member: Meta's
`meta-recsys/silvertorch` vendored at `21aa35e`, synced to upstream `22c2007`, Apache-2.0 with
Meta's `LICENSE` and headers and one "Modified by the retrieve authors" line per changed file,
under Meta's own names (distribution and package `silvertorch`, ops `torch.ops.st.*`, C++
namespace `st::ops`). `CHANGES.md` lists every change, each gated when the
member was `official-fork/` (package `silvertorch_fork`, ops `torch.ops.stfork.*`) and loaded
beside Meta's published package for the parity gates; the published code left the workspace at
RENAME and the member took Meta's names back ([Deleted backends](#deleted-backends)).
`SilverTorch(backend="cuda")`, the default and only value, runs it.

**Forward: one op.** `SilverTorch` runs phase 1 (ours) and then one `torch.ops.st.search` call
(`adapter.fork_search`; `fork_search_cuda.cu`). In order:

1. **Prep, one launch** (`probe_prep_kernel`; `st::probe_prep` exposes it): per query row the
   int8 codes and scale exactly as eager torch computes our `quantize_int8` (the code quotient in
   fp64 rounded to fp32; the scale `abs_max · fp32(1/127)`, as torch divides by a CPU scalar), and
   one block for the probed clusters' ids, lengths and offsets, the warp sizes and the scorer's
   three cumsums.
2. **Filter, into the payloads.** Bloom: the payload kernels evaluate the
   prepared plan (one per distinct expression) on the ≤ 2 bloom columns a payload's docs span and
   cut the same mask bits upstream's payload kernels cut from `column_mask_response`, so no column
   search runs; the scorer reads no other bit of Meta's full-`N` mask, so the result is the one Meta's full
   path returns, without its scan of all `N`. The S9 / C5 partial-vs-full ablation ran on Meta's
   code as published (records `official`), which had both paths.
   Each filter's full-warp and remainder payload kernels are one launch (the blocks split
   between the two), so their latencies overlap at small batches. `exact`: the clauses are
   tested over the probed docs only, one warp per 32-doc payload with `__brev(__ballot_sync(…))`
   as its mask, an inactive clause a warp-uniform skip. The attrs are the layer's
   `fork_clause_attrs` layout (C14), not the `[N, C, A_max]` int64 table the LiNR filter
   reads: int32 `[R, N]`, clause `c` in slot rows `clause_slots[c]:clause_slots[c + 1]` (as many
   as its widest doc has values, so no all-pad slot is stored), each doc's values sorted
   descending so a `-1` ends them. A warp's 32 docs read one coalesced 128-byte row per slot,
   and a lane stops at its match or its first pad. `prepare_queries` casts the query attrs to
   int32 (`fork_query_attrs`: a value outside `[-1, 2^31)` becomes `INT32_MIN`, which no doc
   holds; `register_index` raises on doc values outside that range). A given full-`N`
   `filtering_bit_mask` is read by the scorer as upstream's.
3. **Scorer**: upstream's payload scorer (`process_cluster` / `_remaining`), except that for
   D ≥ 128 full warps go to `process_cluster_coop` (fixed in `search`; no switch):
   the warp reads its 32 consecutive rows linearly (≤ 256 bytes a row a chunk) into a padded
   shared tile and each lane takes its row's `dp4a` dot from there; a payload with < 16 passing
   docs keeps upstream's per-lane loads, and the int32 sum and `post_process` are upstream's, so
   scores are equal. Filtered, a warp loads up to 32 consecutive payloads at once (as many as keep
   ≥ 64 warps an SM) and walks them by shuffle, since most filtered payloads only write sentinels. On `search`'s path its `[B, width]` buffers are not pre-filled: lanes failing
   the filter write the sentinels themselves and each row's written width (from the prep kernel)
   bounds what the select reads. Payload kernels use a warp-parallel cluster search, or one warp
   per cluster at wide probes (≥ 40 full warps a cluster and ≥ 8192 probed clusters).
4. **Top-k on the raw scores, ours (k ≤ 1024)**: a radix select per row chunk of 8192 slots in a
   512-thread CTA (keys that order like the scores, held in registers; four 8-bit warp-aggregated
   histogram passes; keys equal to the k-th taken in index order, as `torch.topk` takes them, so
   the result is deterministic and graph replays equal eager), repeated over the chunks' winners
   until one CTA holds them, then the final select with a register bitonic sort and the `[B, k]`
   epilogue (`official_probe_score`'s, its fp32 multiplies in its order) fused. Rows with ≥ 8
   blocks of 256 slots per top-k slot are first cut to their k best blocks, whose maxima the
   scorer folds as it writes. Above k 1024, `topk` (two-level from `B · width ≥ 2²³`) and a
   one-kernel epilogue.

It returns `(ids int64 [B, k], scores fp32 [B, k])` with `-1` / `-inf` pads.
`official_probe_score` is the gates' reference path over the upstream-shaped scorer.

**Bloom width (C12).** `OfficialConfig.bloom_width`: `"configured"` (the default) builds Meta's
bloom with `bloom_index_build_fixed`, every 2048-doc bundle `m_bits` bits per doc (the layer's
`m_bits` is then required), so its index is `N · m_bits / 8` bytes; `"upstream"` keeps Meta's
`max terms · k · b_multiplier` per bundle (OF-10), the width Meta's code as published had and
the parity gate against it used (`m_bits` is then optional).

**Exact recheck (C13).** `OfficialConfig.recheck` (the default; Meta's code has none):
`prepare_queries` also keeps the query attrs and the layer keeps the cluster-sorted exact attrs
(C14's layout, above) beside the bloom index, and `search` tests the bloom's survivors exactly (partial path: a warp takes up
to 32 consecutive payloads, as many as keep ≥ 64 warps an SM, lane l evaluating payload l's bloom
columns; the warp then walks its payloads, skipping those without bloom survivors, each lane
testing one bloom-passing doc). Bloom ∧ exact is the exact
filter, so the bloom returns no false positive; Meta's bloom semantics are
`OfficialConfig(bloom_width="upstream", recheck=False)`, the setting of the parity gate against
the published code and of the study's timed cells.

**Upstream-shaped ops.** The member still registers the same six ops as upstream, under `st::`
with upstream's schema strings (`test_official.py` pins them, with `search`'s), and the adapter's
reference path calls them unchanged: `ensure_loaded()` / `is_available()` are memoized once
(`PACKAGE`, `NAMESPACE`). So everything the [Meta's ops section](#metas-ops) says (ops, scores,
filter modes, bit order) holds for them until a change below replaces it. Builds: the member's
`setup.py` compiles host code at `-O3` (OF-11) and adds `is_topk` / `fresh_index_post_processing`
to the sources; `env.silvertorch_build` records its `.so`.

| op (adapter caller) | state |
|---|---|
| `fused_kmean_ann` (`fused_scores`) | changed: no host sync (C3); its scorer, which `search` also runs, takes the prep kernel's hints (C6) and gains no pre-fill, block maxima, the cooperative scorer from D 128 and filtered payload batches (K fixes 3, 10, 12, 14, 16; `silvertorch/CHANGES.md`) |
| `fused_kmean_ann_with_partial_masks` (`fused_scores(partial=…)`) | changed: no host sync (C3) |
| `bloom_index_build` (`build_bloom_index`) | upstream's (22c2007: CCCL 3 guards, old cub path on CUDA 12.8) |
| `parse_expression_query_batch` (`parse_plans`) | upstream's |
| `bloom_index_search_batch` (`bloom_full_mask`) | upstream's; the forward calls `bloom_index_search_batch_prepared` instead (C4) |
| `bloom_index_search_batch_return_partial_response` (`bloom_partial_masks`) | upstream's; the forward calls `bloom_index_search_batch_return_partial_response_prepared` instead (C4) |

**No host sync (C3, C4).** `fused_kmean_ann` and `_with_partial_masks` find each warp
slot's probed cluster by a binary search over the device warp cumsum instead of a host
`repeat_interleave`, size their payload buffers by static capacities (a row holds at most
`max_tensor_size_per_row / 32` full warps and `min(31·P, max_tensor_size_per_row)` remainder docs) and
read the totals on the device. Bloom plans are prepared once: `prepare_queries` runs Meta's parser and
uploads its encoding (`adapter.prepare_plans`, `DevicePlans`: an int64 blob of 8 byte offsets per plan
slot, then the data, padded to `slots · plan_capacity_bytes(C, n_stored_hashes)` so a batch shape has
one blob shape); the `*_prepared` searches build each plan per thread from it. The partial search's
`column_mask_response` has the static length `max_partial_columns(B, P, width) = B · (width / 64 + 2P)`
with a zero tail. `silvertorch/silvertorch/ops/_fake.py` gives every `st` op's shape function (16,
`opcheck` green), so the forward is `capturable` and compiles.

### Deleted backends

A hand-written CUDA C++ SilverTorch backend and its one-to-one port into
NVIDIA's CUTLASS Python DSL were deleted once the official backend's
parity gate was green; the git tag `cuda-cute-backends-final` holds them. Both ran the paper's two-kernel
design — a phase-2 mask kernel over a *transposed, cluster-major* bloom
index (or a clause-mask kernel for `"exact"`) producing 1-bit-per-item
masks, then a masked `__dp4a` scorer — and were bit-identical to the
Triton kernels on scores.

Their measurements are superseded results.

**Meta's code as published** (`SilverTorch(backend="official")`: the pinned `meta-recsys/silvertorch`
build at `21aa35e`, the `official` extra, the published forward in `silvertorch.py` and the
adapter's per-backend plumbing, the adapted-vs-published parity tests, and our `clause_mask_packed`
kernel that packed the exact predicate into its scorer's bit mask) left the library and the
workspace at RENAME (2026-10-16: the adapted code takes Meta's names, `backend="cuda"` is its only
backend). Its last parity run against the adapted code (T1 / T6 / C13 bit-exact at int32) stands
behind the tag `meta-published-parity-final`.

**The Triton SilverTorch** (`SilverTorch(backend="triton")`, its `backend="torch"` reference and
`bloom_path="full"`: the `codesigned_probe_score*` and `bloom_full_mask` ops, their reference
twins, the transposed bloom index with its per-bit frequency table, the probe scorers' shared
prep / top-k / id-epilogue scaffold and their `tune-kernels` entries) and **our LiNR bloom
filter** (`BloomFilter`, the `bloom_match` / `bloom_compact` / `bloom_match_scores` ops, the
`bloom_hash` builders and the per-clause salt, `functional.bloom_subset_match`) were deleted after
the adapted official backend's review and parity gates were green; the tag
`triton-silvertorch-final` holds them (the study runs LiNR with the exact filter only; the bloom
in the study is Meta's, inside SilverTorch). Their measurements are superseded results.

## Adding an op

A new Triton kernel touches every one of these; missing one is the usual
way a kernel ships half-integrated.

1. `ops/triton/<k>.py` — the `@triton.jit` body, the `<Name>Config`
   dataclass, `DEFAULT_CONFIG`, the shared `_<k>_prep` / `_<k>_finish`
   pair, the `_<k>_impl`, and the public op (`@torch.library.triton_op`
   unless the launch's store address is data-dependent, in which case
   `@torch.library.custom_op` — see "Graph-break behavior" above).
2. `ops/triton/__init__.py` — import the op from the new module (and add
   it to `__all__`) so `import retrieve.ops.triton` registers it.
3. `ops/reference/<k>.py` — the pure-torch twin, same op name and
   signature. This is the parity oracle by default
   ([testing.md](testing.md#oracle-policy)) and the `"torch"` backend.
4. `ops/tune.py` — a `KernelTuneSpec` in `KERNELS` plus a curated entry
   in `DEFAULT_CONFIG`, so `tune-kernels` covers it and
   `test_tune_smoke.py` catches schema drift between `tune.py` and the
   kernel's `_impl`.
5. `retrieve/tests/parity/test_<k>.py` — kernel vs. reference, following
   [testing.md](testing.md#new-parity-test-for-a-triton-kernel).
6. A compile/graph-break gate: a row in `test_linr_compile.py` if a module's forward calls
   the op under `torch.compile`, or a dedicated case in `tests/compile/` otherwise.
7. This file — a kernel section with its inputs/outputs, launch grid,
   tile config and any numerics caveat.
8. [architecture.md](architecture.md)'s op table (`## Kernels`) — one row
   naming the op and its consumer module.
9. A `DISPATCH` row in `interfaces.py` if a new module uses the op —
   [architecture.md](architecture.md#backend-dispatch) is derived from
   it, not maintained separately.
