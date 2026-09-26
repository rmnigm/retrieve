---
title: kernels
created: 2026-09-26
updated: 2026-09-26
type: concept
tags: [kernels, library]
sources: [retrieve/src/retrieve/ops/]
---

# `retrieve` kernels

Every kernel of ours is Triton. The one non-Triton path,
`SilverTorch(backend="official")`, is Meta's own extension behind an
adapter ([its section](#official--metas-torchopsst-kernels-as-the-reference-backend)).
There is no CUDA C++ or CuTe DSL backend ([Deleted backends](#deleted-backends)).

The Triton kernels live flat in
[`ops/triton/`](../../retrieve/src/retrieve/ops/triton/), one file per
kernel, registered as `torch.ops.retrieve.*` when the package is imported
(`_load.py`); every op has a pure-torch twin with the same name and
signature in [`ops/reference/`](../../retrieve/src/retrieve/ops/reference/)
(the `"torch"` backend and the parity oracle). By domain:

- LiNR — kernels used by `PrefilterKNN` / `OneBitKNN` / `SimHashKNN`
  (`fused_masked_knn_topk`,
  `oporp_1bit_match_topk`). `PostfilterKNN`'s dense fp16-input matmul +
  top-K and `PostfilterKNNInt8`'s int8 `_int_mm` + int32 top-K are
  both pure torch — there's no real fusion to win over cuBLAS LtGemm +
  CUB.
- filters — standalone filter primitives consumed by the `FilterModule`
  family: `clause_compact` (powers `ExactAttributeFilter.evaluate_indices`),
  `clause_mask` (powers `ExactAttributeFilter.evaluate_mask`),
  `bloom_match` (powers `BloomFilter.evaluate_mask`) and
  `bloom_compact` (powers
  `BloomFilter.evaluate_indices`). The mask/compact split mirrors the
  filter API split documented in [filtering.md](filtering.md).
- SilverTorch — the two co-designed IVF probe+score kernels
  (`codesigned_probe_score` for the IVF + INT8 + Bloom co-design,
  `codesigned_probe_score_exact` for the IVF + INT8 + exact AND-of-OR
  variant — `SilverTorch.filter_mode` picks one). The adapter over Meta's
  `torch.ops.st.*` ops, selected by `SilverTorch(backend="official")`, is
  its own namespace, [`ops/official/`](../../retrieve/src/retrieve/ops/official/adapter.py).

Cross-kernel `@triton.jit` building blocks live in
[`common.py`](../../retrieve/src/retrieve/ops/triton/common.py) —
see [Shared kernel helpers](#shared-kernel-helpers-opstritoncommonpy) —
and the plain-Python launch scaffold they share in
[`_host.py`](../../retrieve/src/retrieve/ops/triton/_host.py): `probe_prep` +
`probe_topk` (the probe scorers' shared launch arguments and their top-k +
id-epilogue launch), the boundary checks and `wide`, and `grid_batch_tiles`
(the 3-D grid split of the filter and probe kernels).

The LinR kernels are the focus of this doc; the filter primitives
(`clause_compact`, `clause_mask`, `bloom_match`, `bloom_compact`) are
covered next, and the SilverTorch-only kernels at the end for context.

All kernels follow the same conventions.

- One launch per `forward()` call. No persistent threads, no streams.
- Inputs are CUDA tensors. Every `_<name>_prep` checks its boundary
  (`_host.check_contiguous`, `_host.check_pow2`). An item-side table (the
  index, and the `[B, N]` candidate lists) must arrive contiguous, and a
  view raises `ValueError`, because a per-call `.contiguous()` would copy
  the whole index on every forward. Query-side tensors are small and are
  still `.contiguous()`-copied. Every `tl.arange` extent (`D`, `W`) must be
  a power of two: `ValueError` at the boundary instead of a Triton
  `CompilationError` from inside the compiler. Both are gated by
  [`test_op_boundary.py`](../../retrieve/tests/correctness/test_op_boundary.py).
  The prep then passes `.stride(i)` for every axis, so a kernel body
  never hard-codes a layout. The strides are always the contiguous ones.
- Top-K selection is **not** in-kernel. Each kernel writes a `[B, ·]` score
  buffer and the host calls `torch.topk` on it. CUB's top-K (under torch)
  is faster than anything we can implement in pure Triton without a
  warp-level radix-select primitive.
- Tile config (`block_n`, `num_warps`, `num_stages`; `block_p` for the
  silvertorch kernels)
  is offline-tuned per kernel and shipped as a
  single `DEFAULT_CONFIG` constant on the kernel module. No runtime
  `@triton.autotune`. Callers who want a non-default tile pass
  `config=<Kernel>Config(...)` to the private
  `_<name>_impl(..., config=)` companion — the public op has a fixed
  schema and always uses `DEFAULT_CONFIG`. The `tune-kernels` CLI
  (shipped with the library at `retrieve.ops.tune:main`) sweeps the
  candidate grid on a given arch and prints the line to paste into the
  kernel file. The same convention applies uniformly across linr,
  filter, and silvertorch kernels — see
  [Autotune separation](#autotune-separation) below for the rationale.
- Host-wrapper shape. Every kernel file follows one pattern: a shared
  `_<name>_prep(...)` does validation + contiguity + output-buffer
  allocation + the full launch-kwarg dict (returned in a small frozen
  `_<Name>Launch` dataclass), a shared `_<name>_finish(...)` owns the
  topk/gather/sentinel epilogue where one exists, and every entry point
  — the eager `_impl` and the `@triton_op` public op(s) — is prep →
  one launch line → finish. Validation therefore runs on the
  production ops, not just `_impl`. (`bloom_match` is the one
  exception: a single-op ~80-line file with no `_impl`, no Config, no
  prep — see its section.)
- Graph-break behavior. Ten ops are registered across the seven Triton
  kernel files (the official backend registers no op of ours; it calls
  `torch.ops.st.*`), in two flavours.

  **Eight on `@torch.library.triton_op`**, each with a textually-inline
  `wrap_triton(_kernel)[grid](**launch.kwargs)` launch: `clause_mask`,
  `fused_masked_knn_topk`, `bloom_match`, `codesigned_probe_score`,
  `codesigned_probe_score_bloom`, `codesigned_probe_score_exact`,
  `oporp_1bit_match_topk_full`, `oporp_1bit_match_topk_indirect`. The
  decorator stops dynamo from graph-breaking at the wrapper boundary, so
  each layer's full forward captures into one cudagraph_trees graph, and
  lets inductor see the underlying `@triton.jit` kernel (preserves the
  reference under `torch.export` — gated by
  [`tests/compile/test_export_kernel_ref.py`](../../retrieve/tests/compile/test_export_kernel_ref.py)). The `wrap_triton` call
  must stay textually inside each decorated body — dedup targets the
  code around it, never the launch line itself.

  **Two on `@torch.library.custom_op`** (`mutates_args=()`,
  `device_types="cuda"`, plus a `register_fake`), opaque to inductor and
  calling the shared `_impl`: the two **stream-compaction** kernels,
  `clause_compact` and `bloom_compact`. Under `triton_op`, inductor
  analyses the kernel's TTIR for mutated pointers, and its provenance
  walk follows a store address back through every argument of the
  `tt.call` that produced it. `compact_store`'s address is
  `row_base + intra` — data-dependent on the pass mask, which is the
  result of the `clause_pass` / `bloom_subset_pass` call whose arguments
  include the index buffers — so `item_attrs_ptr` / `query_attrs_ptr` /
  `is_reverse_ptr` (graph inputs) are reported as mutated and
  cudagraph trees skip the *whole* forward with "skipping cudagraphs due
  to mutated inputs", silently falling back to compiled-eager (the
  golden baseline's `graph` cells for `linr_v2` / `linr_v3` were taken
  that way and are compiled-eager numbers). `clause_mask` stores at a data-independent
  address and is unaffected, which is why it stays a `triton_op`. As
  custom ops the launches are opaque, the ops are functional as declared,
  and the forward captures — gated by
  [`tests/compile/test_linr_compile.py`](../../retrieve/tests/compile/test_linr_compile.py),
  which asserts `cudagraph_skips == 0` for compiled `linr_v1`/`v2`/`v3`
  × clause/bloom. The cost is one custom-op dispatch instead of an
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
   kernel reference; the two `custom_op`-registered compaction kernels
   have no launch line of their own and just call `_impl`) and, where policies intentionally diverge, in
   explicit prep/finish flags (e.g. `fused_masked_knn_topk`'s
   `bucket=` / `pad_to_k=`). The op schema doesn't carry the config
   dataclass; tests and the tuner reach `_impl` directly to pass an
   override.
5. Tuning is offline: [`ops/tune.py`](../../retrieve/src/retrieve/ops/tune.py) (`uv run
   tune-kernels <kernel-subcommand>`) is a declarative registry — one
   `KernelTuneSpec` per kernel in the `KERNELS` tuple, from which the
   seven click subcommands are generated
   (`fused-masked-knn-topk`, `oporp-1bit-match-topk`,
   `codesigned-probe-score`, `codesigned-probe-score-exact`,
   `clause-mask`, `clause-compact`, `bloom-compact`;
   [`test_tune_smoke.py`](../../retrieve/tests/correctness/test_tune_smoke.py)
   pins the list), all driven by one
   generic `_sweep` + `_print` pair. Each spec sweeps its `(block,
   num_warps)` grid against built-in shape regimes mirroring real-eval
   workloads (goodreads N≈800k, arxiv N≈3M, synth-15M; batch sizes from
   the eval configs). The regime-swept kernels
   (`codesigned-probe-score-exact` and the three filter kernels) accept
   repeatable `--regime N,B,C,A_MAX` (or `N,B,W`) flags; the
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
   (`bloom_match` has no subcommand — see its section.) A CUDA-gated
   smoke test
   ([`test_tune_smoke.py`](../../retrieve/tests/correctness/test_tune_smoke.py))
   runs one tiny regime per spec so schema drift between tune.py and
   the kernel `_impl`s breaks CI instead of a tuning session.

The compact kernels have no accumulating state (per-tile
counts and offsets are plain stores, so a repeated launch rewrites the
same values); the tuner calls their host wrappers, which allocate fresh
buffers per call either way.

## Addressing

Triton's `program_id` is int32, and so is every integer argument below 2³¹, so a row base
`row * stride` is an int32 product and wraps once an address passes 2³¹ elements. Two
classes of product reach that inside the paper's scale ladder:

- **output / candidate axis**, `B·N ≥ 2³¹`: every `[B, N]`-shaped buffer a kernel indexes
  by batch row: the `clause_mask` / `bloom_match` masks, the compaction scratch and
  `[B, N]` result, the `[B, P]` candidate lists and score buffers of
  `fused_masked_knn_topk`, `oporp_1bit_match_topk` and the probe scorers (B = 144 at
  N = 15M).
- **item axis**, `N·row_stride ≥ 2³¹`: item tables read by contiguous tile with a row
  stride above 1: bloom signatures and OPORP bits at `N ≥ 2³¹/W` (134M at `W = 16`,
  1.07B at `W = 2`), clause attrs at `N ≥ 2³¹/(C·A_max)` (268M at `C·A_max = 8`).

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

**Why a constexpr, not int64 everywhere.** Always-int64 was built first and measured
([artifacts](../artifacts/kernel-opt/predictions.md)): `fused_masked_knn_topk` +7 %, since
its 1.5M 32-lane programs pay for a full 64-bit multiply per row base. `bloom_compact` was
+13 %: 40 → 46 registers, one block per SM fewer. `clause_compact` at B = 1 was +3 %, from the
shifted-pointer formulation itself. With the constexpr, launches below 2³¹ time within noise
of the int32 kernels, and only launches past 2³¹ pay.

**Bounds that remain.** `N`, `P` and every per-row count stay below 2³¹: the compaction
scratch holds ids as int32, and `tile_id · BLOCK` is int32. The batch-first grids cap `B`
at 65,535.

**Gate.**
[`tests/correctness/test_large_offsets.py`](../../retrieve/tests/correctness/test_large_offsets.py)
runs one case per class with a planted answer: `B = 144, N = 16M` through
`clause_mask`, `clause_compact`, `fused_masked_knn_topk` and
`oporp_1bit_match_topk_indirect`, and one `[140M, 16]` int64 table read as bloom
signatures, clause attrs and OPORP bits. It is skipped below 48 / 24 GiB of free
device memory. Every widening is mutation-checked: without its int64 cast the case
raises an illegal address or returns wrong ids.

## Shared kernel helpers (`ops/triton/common.py`)

[`ops/triton/common.py`](../../retrieve/src/retrieve/ops/triton/common.py)
holds the `@triton.jit` building blocks the kernel bodies call (Triton
inlines them). Every helper is a pure function of already-loaded tiles
or takes fully-resolved addressing from the caller — no helper decides
its own tile shape, launch grid, or masking policy:

- `row_base(ptr, row, stride, WIDE)` / `tile_rows(ptr, row0, lane, stride, WIDE)` — the
  row-base arithmetic of [Addressing](#addressing); every kernel's `[B, ·]` row and item
  tile goes through one of the two.
- `probe_tile(probe_ids_ptr, offsets_ptr, bid, t, n_probe, NPP, BLOCK_P)` —
  a probe scorer program's slice of the compact CSR layout, the row's
  table built in-kernel ([SilverTorch kernels](#silvertorch-kernels)).
- `probe_ids_kernel` — a *launched* kernel: the probe scorers' id
  epilogue after `torch.topk` (slot → `sort_perm[pos]`, `-1` at `-inf`).
- `popcount_int64(x) → int32` — the hardware `POPC` (`libdevice.popc` on
  int64, `__nv_popcll`); used by `oporp_1bit_match_topk`. Its torch twin
  [`functional.py::popcount_int64`](../../retrieve/src/retrieve/functional.py)
  is SWAR; popcount is exact integer math, so the two agree bit for bit
  (see [Numerics](#numerics)).
- `bloom_subset_pass(qb, sigs) → [BLOCK] int1` — the bloom subset test
  in the `qb & ~sig == 0` OR-reduce form (boolean-identical to, and
  cheaper than, the equality + min-reduce form; all three bloom
  consumers — `bloom_match`, `bloom_compact`,
  `codesigned_probe_score` — use it). Callers AND
  the result with their own validity mask.
- `clause_pass(item_attrs_ptr, is_reverse_ptr, query_attrs_ptr, ids,
  load_mask, bid, strides..., C, A_MAX) → [BLOCK] int1` — the exact
  AND-of-OR clause predicate (inner OR over `A_MAX` slots, outer AND
  over `C`, reverse XOR, `q_c == -1` inactive override), already ANDed
  with `load_mask`. Used by `clause_mask` / `clause_compact`
  (`ids=n_offsets`, `load_mask=n_valid`) and
  `codesigned_probe_score_exact` (`ids=safe_ids`, `load_mask=valid`).
  `is_reverse_ptr` must point at int8 storage (host preps do
  `.to(torch.int8)`; Triton can't load native torch.bool).
- `compact_store(pass_mask, ids, base, out_ptr, bid, ...)` — the
  stream-compaction store (`cumsum` intra-tile rank + masked store at
  the caller-supplied `base`, cast to the pointee type).
- `compact_stash(pass_mask, ids, tile_counts_ptr, scratch_ptr, bid,
  tile_id, ..., BLOCK_N)` — phase 1 of the two compaction ops: the
  tile's survivor count, and `compact_store` of its ids into the tile's
  own slot range `scratch[bid, tile_id * BLOCK_N :]` (int32).
- `compact_scatter_kernel` — the one *launched* kernel in this file:
  phase 3 of both compaction ops, one program per `(row, tile)` on the
  predicate launch's grid, moving the tile's stashed run to the scanned
  row offset. Launched by
  [`_host.compact_finish`](../../retrieve/src/retrieve/ops/triton/_host.py),
  which also does the `cumsum`.
- `or_combine(a, b)` — combine_fn for `tl.reduce` OR-reductions.

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
  `fused_masked_knn_topk` (V2 `triton`) by its fp32 `tl.sum`. Higher is
  better. `-inf` marks masked-out / padded positions. An fp16 *score*
  would round runs of near-tied items to one value: on YFCC-10M the
  top-1000 spans about fifteen fp16 quanta (2⁻¹¹ near 0.8), and fp16
  scores cost the exact algorithms `recall_oracle@1000` 0.956; fp32
  scores over the same fp16 table give 0.993, and the remaining 0.007 is
  the fp16 *storage* rounding (an fp32 table gives 1.0 at twice the
  memory; [decisions](../decisions.md#library),
  [artifact](../artifacts/l1-l2/README.md)). The cuBLAS path's
  tensor-core accumulator truncates: measured ≤ 1.1e-6 from an fp64 dot
  of the same fp16 inputs at D=128, against a plain fp32 sum's ~1e-7;
  gated in [`test_accumulation.py`](../../retrieve/tests/parity/test_accumulation.py).
- 1-bit Sign-OPORP (V3): `D - 2 * popcount(query_bits ^ item_bits)`, fp32.
  `D = 64 * W`. This is the standard Hamming-to-dot-product relation for
  sign-quantized vectors. Higher is better. Same `-inf` sentinel.
- The **id** returned alongside a `-inf` score is always `-1`, on every
  backend — `interfaces.py` names `-1 / -inf` as the paired "no item"
  sentinels. The torch and official backends get there through
  `masked_topk`; the two SilverTorch Triton epilogues apply it themselves
  (see below). So a caller can read either tensor to find the dead slots,
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
so loading the same checkpoint produces the same bits. V3 store (`item_bits`,
`signs`, `perm`) as buffers; both backends call the same projection helper
on every forward, so the torch reference and the Triton kernel see byte-for-
byte identical bits.

## PostfilterKNN dense path — pure torch, no kernel

`PostfilterKNN`'s forward is `torch.mm(query_fp16, item_embs_t,
out_dtype=torch.float32)` + optional `masked_fill(-inf)` + `torch.topk`
over the fp32 scores (§ Score conventions) — implemented directly in
[`PostfilterKNN`](../../retrieve/src/retrieve/modules/knn.py).
There is no Triton kernel here because one that only fuses the matmul
still materializes the full `[B, N]` score buffer and calls the same
host-side `torch.topk`, so its memory traffic and selection cost match
cuBLAS + CUB exactly. `PostfilterKNN` accepts the
`backend=` flag for API symmetry but both values dispatch to this same
pure-torch path.

## PostfilterKNNInt8 dense path — pure torch, no kernel

`PostfilterKNNInt8`'s forward is `torch._int_mm(query_codes,
item_codes_T)` (int8×int8 → int32, cuBLAS LtGemm, IMMA tensor cores on
sm_80+) + optional `masked_fill(int32_min)` + `torch.topk` on the int32
result — implemented directly in
[`PostfilterKNNInt8`](../../retrieve/src/retrieve/modules/knn.py).
Items and queries are int8-quantized with one global scalar scale each
(SilverTorch §3.2); because both scales are global constants per call,
the int32 dot product is a positive monotonic transform of the true
fp32 dot, so topk ordering is exact (modulo per-element int8 rounding)
without rescaling to fp32. Storage is one `[D, N]` int8 buffer — half
of `PostfilterKNN`'s fp16 layout. `torch._int_mm` requires `M >=
17`, so small batches are zero-padded before the matmul and sliced
after; quantization runs **before** padding so the padded zero rows
don't shift the global scale. No Triton kernel; `backend=` is accepted
for API symmetry but both values dispatch here.

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
partial sums of the same order) the error against an fp64 dot is 0.028,
enough to swap one boundary pair on 6.3 % of `c0_genre` rows against the
`torch` backend. In fp32 it is ~1e-5; the `torch` side returns fp32 too
(§ Score conventions), so the two backends differ only in reduction order
([validation](../validation.md#library-gates)). The
accumulation-width parity file
[`test_accumulation.py`](../../retrieve/tests/parity/test_accumulation.py)
pins every scoring kernel against an fp64 oracle, because the other
parity files compare Triton against `ops.reference` at the same input
dtype and cannot see this class of defect.

**Launch grid** `(cdiv(P, BLOCK_N), B)`, tile axis on `grid_x`, batch on
`grid_y`. Each program owns one
`(query, p-tile)` cell and gathers `BLOCK_N` item rows by indirect load:

```
item_ids[BLOCK_N] = pos_indices[b, p_off]
emb_rows[BLOCK_N, D] = item_embs[item_ids]
dots[BLOCK_N] = sum(emb_rows * q[None, :], axis=1)
```

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

**Tile config.** `FusedMaskedKnnTopkConfig(block_n, num_warps,
num_stages)` — shipped as `DEFAULT_CONFIG` on the kernel module; pass
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
`libdevice.popc`). It replaced a five-step SWAR bit-twiddle: −12.5 % on the
indirect op at B=16, P=3M and −2.1 % on the full scan, where top-k(5000)
dominates ([artifacts](../artifacts/kernel-opt/predictions.md)). The torch
reference [`popcount_int64`](../../retrieve/src/retrieve/functional.py) is
still SWAR, because this torch has no `bitwise_count`. Popcount is exact, so torch and Triton
produce **bit-exact** identical scores.

**Launch grid** `(cdiv(n, BLOCK_N), B)`: tile axis on `grid_x` (up to
2³¹), batch on `grid_y` (up to 65535), because `cdiv(N, BLOCK_N)` can
overflow `grid_y` at large N. The body is modeled on `bloom_match`
(int64-word reduction over `W`), which launches batch-first.

**HAS_INDICES path**: replaces the contiguous `n_off` load with an
indirect `pos_indices[b, n_off]` lookup; otherwise identical. Used for
candidate-set rerank and for the masked V3 path (after `compact_mask`).

**Tile config.** `Oporp1BitMatchTopkConfig(block_n, num_warps,
num_stages)` — shipped as `DEFAULT_CONFIG` on the kernel module; pass
`config=` to override. Re-tune on a new arch via `uv run tune-kernels
oporp-1bit-match-topk`.

**Bucketing.** For the `HAS_INDICES=True` path, the score-buffer
width is `n_kernel = _bucket_n(positive_indices.shape[1])` (buckets
`{4096, 65536, 1048576, 16777216}`), passed as `tl.constexpr N`: the
same JIT-cache invariant as `fused_masked_knn_topk`. The candidate path
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
the second launch, or a packed bitmask): both were built and measured
([artifacts](../artifacts/deterministic-compaction/)).
Re-evaluation costs a full second pass of the predicate (1.9–3.1× on the
kernel, +43–58% on a V2/V3 forward). The bitmask — and even a bare
count-only epilogue — makes the *predicate* launch 1.7–1.8× slower at
`B = 1`, where the grid is under one wave and per-program latency is the
kernel time; the `cumsum` + masked-store epilogue keeps it at the one-pass
speed (1.10× at B=1, 1.03× at B=16, full pipeline). The scratch traffic is
the survivors only (4 B in, 8 B out per id), and so is the output's: with
no tail store the scatter writes nothing for a rejected item (−11.7 % on
`clause_compact`, −5.3 % on `bloom_compact` at B=16, 3M items, 1.8 %
pass rate, against a `-1` tail store; [artifact](../artifacts/l1-l2/README.md#timing)). Its allocation is
`4 · B · T · BLOCK_N` bytes, half the `[B, N]` int64 result. `counts` is a
fresh tensor, not a view into the scan: inductor asserts custom-op outputs
are 16-byte aligned, and at `B = 1` the scan's last column is a contiguous
view at element offset `T - 1` (pinned by
`test_compiled_batch_of_one_bloom_v2`).

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
The order is fixed on purpose ([decisions](../decisions.md#library)): a
row base claimed with `tl.atomic_add` orders a row by tile completion,
which `PrefilterKNN`'s top-k and `OneBitKNN`'s heavily tied Hamming
ranking turn into run-to-run quality noise (2e-6 / 7e-5 on `linr_v2` /
`linr_v3`). The price is the scan, the stash round trip and two extra
launches, measured in the
[artifacts](../artifacts/deterministic-compaction/).

**Op registration** is `@torch.library.custom_op`, not `@triton_op` — this
kernel and `bloom_compact` are the two exceptions in the tree. The
data-dependent `base + intra` store address above is exactly what makes
inductor's TTIR mutation analysis report the index buffers as mutated, which
makes cudagraph trees skip the compiled forward; the "Graph-break behavior"
bullet in the conventions list at the top of this file has the full mechanism
and names the regression test.

**Tile config.** `ClauseCompactConfig(block_n, num_warps, num_stages)`
— shipped as `DEFAULT_CONFIG` on the kernel module (not tuned for the
two-phase shape; the retune is roadmap G-a, TF-3); tests/tuner
override via `_clause_compact_impl(..., config=)`. Re-tune on a new arch via
`uv run tune-kernels clause-compact`.

## `clause_mask` — fused clause eval emitting `[B, N]` bool

[`ops/triton/clause_mask.py`](../../retrieve/src/retrieve/ops/triton/clause_mask.py).

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

**Tile config.** `ClauseMaskConfig(block_n, num_warps, num_stages)` —
shipped as `DEFAULT_CONFIG` on the kernel module; tests/tuner override
via `_clause_mask_impl(..., config=)`. Re-tune on a new arch via
`uv run tune-kernels clause-mask`.

## `bloom_match` — Bloom subset test

[`ops/triton/bloom_match.py`](../../retrieve/src/retrieve/ops/triton/bloom_match.py).
A standalone filter primitive: it powers
[`BloomFilter.evaluate_mask`](../../retrieve/src/retrieve/modules/filters.py)
on CUDA. SilverTorch's in-cluster bloom is fused separately into
`codesigned_probe_score`; the two paths are independent.

Implements the conjunctive subset test `(qb & sigs) == qb` reduced over
`W` int64 words. The item-side sigs come from
[`build_signatures`](../../retrieve/src/retrieve/indexing/bloom_hash.py)
(`indexing/bloom_hash.py`, the single home of the bloom hash
math) at `register_index` time; the per-call query bits go through the
loop-free `build_query_signatures` path (same module). The kernel is
purely a bitwise reduction.

**Query-build path.** `build_signatures` is a python `range()`-driven
chunk loop over the index — fine at register time (bandwidth-bound,
~128 ms for N=3M) but a poor fit for the per-forward query build, where
n=B is always small and the loop's trip count makes dynamo specialize on
shape. `build_query_signatures` mirrors the body without the loop
(both wrap the shared `_signature_batch` core, so the chunked and
loop-free paths agree exactly — asserted by
[`test_bloom_hash.py`](../../retrieve/tests/correctness/test_bloom_hash.py));
keeping it purely tensor-flow lets the harness's
`torch.compile(mode="reduce-overhead", dynamic=False, fullgraph=True)` of
each algo (`evaluation/bench/measure.py`, one capture per batch size)
capture it with no graph break. The eager body fires
~15 separate CUDA kernels per call (~0.4 ms wall-clock, **flat in
B**) — pure launch overhead, since the actual work is microseconds;
under the algo-level cudagraph_trees capture the same work collapses
into one replay. Standalone use of `BloomFilter` (not via an Algo
wrapper) runs the function eagerly — wrap externally with
`torch.compile` if you want the cudagraph win there too.

**Hash invariant.** `_signature_batch` (the shared core in
`bloom_hash.py`) keys each hash on `(clause_idx, value)` (paper §4.1:
"for each feature" — a *feature* is a `(key, value)` pair). It does
this by XOR-ing a per-clause salt — `_mix64(clause_id, …)` — into the
post-`_mix64` hash, before the position mask. This is what prevents
value `V` in clause C0 from colliding with the same `V` in clause C3
when clauses share a value vocabulary; without it, single-clause
queries on overlapping vocabularies leak ~25–30% of non-matching items
as false positives. The salt is a `[C]` int64 buffer (`clause_salt`,
built by `generate_clause_salt` and registered at `register_index`
by `BloomFilter` and `SilverTorch`), so the per-forward query build
issues no host→device copy — see
[filtering.md](filtering.md#bloom-hash-keys-clause_idx-value). The kernel itself is untouched: it consumes
`[N, W]` / `[B, W]` int64 buffers as opaque bits. The salt costs one
extra elementwise XOR per chunk (well under measurement noise vs the
existing scatter + word-pack reduction); kernel HBM traffic and launch
shape are unchanged. The hash math is pinned: persisted `bloom_sigs`
buffers must stay bit-valid across refactors, so any change to
`bloom_hash.py` invalidates every stored index (an index persisted
without the `(clause_idx, value)` keying is stale and must be rebuilt).

```
inputs:    qb     [B, W]     int64    packed query bloom signature
           sigs   [N, W]     int64    packed item bloom signatures
output:    mask   [B, N]     bool     (qb & sigs[n]) == qb, AND over W
```

**Launch grid** `(B, tiles_y, tiles_x)` from `_host.grid_batch_tiles`, the
same 3-D split as the clause and compaction kernels: batch on `grid_x`,
the `cdiv(N, 128)` tiles across `grid_y × grid_z`. A 2-D grid with tiles
on `grid_y` capped `N` at 65,535 · 128 = 8,388,480, below the tuner's own
15M regime. It is the structural ancestor of `oporp_1bit_match_topk` (the
popcount step is the only material difference in the body), which launches
`(cdiv(N, BLOCK_N), B)`, tile axis first. Each program loads `qb[b, :]`
once, then a `[BLOCK_N, W]` tile of `sigs`, and emits `[BLOCK_N]` bool to
the output buffer.

`N` stays a `tl.constexpr` (one compile per registered index). Measured
against a runtime `N`: 1269 µs vs 1418 µs at `B = 16, N = 3M`
([artifacts](../artifacts/kernel-opt/predictions.md)). The exact `N` is the
only difference, and the 1-byte mask store is where the time goes.

**Inner op**: the shared
[`common.bloom_subset_pass`](../../retrieve/src/retrieve/ops/triton/common.py)
helper — `qb & ~sig` per word, OR-reduced over `W`, `== 0` at the end
(`(qb & sig) == qb ⇔ qb & ~sig == 0`). This is the cheaper algebraic
form (it saves the int32 cast + min reduction of the equality +
min-reduce form) and is boolean-identical; all three bloom
consumers use it.

**Wins** 10–20× over the pure-torch broadcast `(qb.unsqueeze(1) & sigs)
== qb.unsqueeze(1)).all(-1)` because the broadcast materializes
`[B, N, W]` int64 (24 GiB at `B=64, N=2M, W=16` worst case) before the
reduction; the kernel keeps the tile in registers.

**No autotune, no Config, no `_impl`** — this is the one kernel outside
the `DEFAULT_CONFIG` convention (fixed `BLOCK_N = 128` with a tile-tail
mask for `N < 128`, ~80-line file, single op). Rationale: the per-call
width is dictated by `N`, so a Config would be tuned against exactly
one regime; there is correspondingly no `tune-kernels` subcommand. The
fused `bloom_compact` (next section) shares this kernel's inner
subset-test helper and its 3-D grid, and adds `clause_compact`'s
two-phase compaction.

## `bloom_compact` — fused subset test + stream compaction

[`ops/triton/bloom_compact.py`](../../retrieve/src/retrieve/ops/triton/bloom_compact.py).

Powers `BloomFilter.evaluate_indices` on CUDA. Combines `bloom_match`'s
subset-test inner loop with `clause_compact`'s count → scan → write
compaction, so the dense `[B, N]` bool plus host-side argsort that the
ABC fallback (`compact_mask(bloom_match(.))`) would otherwise produce
never materializes.

```
inputs:    qb     [B, W]   int64    packed query bloom signature
           sigs   [N, W]   int64    packed item bloom signatures
return:    positive_indices [B, N] int64  (full width, written on [:counts] only)
           counts           [B]    int64
```

Same full-width `[B, N]` return contract as `clause_compact` — bound
reads by `counts[b]`; nothing past it is written. Registered as a
`@torch.library.custom_op` for the same reason `clause_compact` is (the
compaction store address is data-dependent), so both compaction kernels
capture under cudagraph trees.

**Launch grid** `(B, tiles_y, tiles_x)`, same 3D shape as the clause
compact/mask kernels (`tile_id = tile_x * tiles_y + tile_y`). Each
program loads `qb[b, :]` once, the `[BLOCK_N, W]` `sigs` tile, computes
the subset test via the shared `common.bloom_subset_pass` (same helper
as `bloom_match`), then hands the tile's count and surviving ids to
`common.compact_stash`; `_host.compact_finish` scans and scatters, exactly
as for `clause_compact`. Wide `W` (=16 at the default `m_bits=1024`) makes
this kernel register-pressure-bound; at large `block_n` with few warps
it spills catastrophically (5–15× slowdown observed at `block_n≥512,
num_warps≤4`). The shipped default keeps `block_n` moderate and warps
high to stay off that cliff.

**Output ordering** within a row is **ascending item order** — the same
contract as `clause_compact`, `torch.equal` to
`ops.reference.bloom_compact` and reproducible across launches and
processes. V2's `fused_masked_knn_topk` and V3's HAS_INDICES path consume
the list in that order, which is what keeps their tie-breaks repeatable.

**Tile config.** `BloomCompactConfig(block_n, num_warps, num_stages)` —
shipped as `DEFAULT_CONFIG`; tests/tuner override via
`_bloom_compact_impl(..., config=)` (not tuned for the two-phase shape;
roadmap G-a, TF-3). Re-tune via `uv run tune-kernels
bloom-compact`. `qb` is built host-side via
`bloom_hash.build_query_signatures`; folding it into the kernel adds
register pressure with no obvious win and is explicitly out of scope.
The same `(clause_idx, value)` keying invariant documented under
`bloom_match` applies — kernel is opaque to bits, so no kernel-side
change.

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
| [`PostfilterKNNInt8`](../../retrieve/src/retrieve/modules/knn.py) | always dense | none — pure torch `torch._int_mm(...)` + `masked_topk` (int8×int8 → int32, IMMA) |
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
same contract as the triton `clause_compact`/`bloom_compact` kernels so
torch and triton paths stay interchangeable. Implementation:
`mask.sum(1)` for counts, then
`mask.float().argsort(descending=True, stable=True)` for the indices —
no `.item()` sync, no narrow slice. The tails differ across
implementations (arbitrary argsort-tail ids here, unwritten memory in
the kernels), so consumers must bound reads by `counts` either way.

### [`popcount_int64`](../../retrieve/src/retrieve/functional.py) (`retrieve.functional`)

Torch-side SWAR popcount. Required because this PyTorch (2.10.0+cu128)
lacks `Tensor.bitwise_count`. Returns int32 to keep the downstream sum
narrow. The kernel side uses the hardware `POPC`; both are exact, so the
OPORP/SimHash parity tests stay bit-exact.

### [`quantize_oporp_1bit`](../../retrieve/src/retrieve/indexing/quantize.py) (`retrieve.indexing`)

Build-time only. `O(D)` parameter cost — the `signs` vector and `perm`
permutation are the entire projection. Apply with one elementwise multiply
and one `index_select`, no matmul needed. The `_pack_signs_to_int64`
helper packs the sign-quantized output into `[..., W]` int64 words using
`<<` and `sum(-1)`; same packing used both at index time and at query time.

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
  launch tax collapses — measured ~2.5× speedup at B=8 and B=64.
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

**The SilverTorch dequant is one expression in five places.** The
bit-exact contract across `triton` / `torch` / `official` rests on the
left-associated `dot.float() * q_scale * global_scale` being written the
same way at every site that computes it: the two Triton kernels
([`codesigned_probe_score.py`](../../retrieve/src/retrieve/ops/triton/codesigned_probe_score.py),
[`codesigned_probe_score_exact.py`](../../retrieve/src/retrieve/ops/triton/codesigned_probe_score_exact.py)),
the reference op ([`ops/reference/codesigned_probe_score.py`](../../retrieve/src/retrieve/ops/reference/codesigned_probe_score.py),
shared by its bloom and exact siblings), `SilverTorch._forward_candidates`
([`modules/silvertorch.py`](../../retrieve/src/retrieve/modules/silvertorch.py))
and the official epilogue `dequantize_scores`
([`ops/official/adapter.py`](../../retrieve/src/retrieve/ops/official/adapter.py)).
Reassociating any one of them (`dot * (q_scale * global_scale)`) changes
the last bit and breaks `torch.equal` in `test_official.py` T1 — it is
enforced by those tests, not by the code. In the Triton kernels both
scalars go through `tl.cast(·, tl.float32)`. Under `torch.compile`,
Inductor passes `global_scale` into the kernel source as a Python float
rather than a typed argument; `.to()` on it raises inside Inductor's TTIR
analysis, which then produced NaN scores instead of an error. The cast keeps the
product in fp32 on both paths. Eager, it is a no-op.

## SilverTorch kernels

Two co-designed IVF probe + INT8 scoring kernels live in
[`ops/triton/`](../../retrieve/src/retrieve/ops/triton/),
selected by `SilverTorch.filter_mode`: `codesigned_probe_score` for
`filter_mode ∈ {"none", "bloom"}`, `codesigned_probe_score_exact` for
`filter_mode="exact"`. Both power
[`SilverTorch.forward`](../../retrieve/src/retrieve/modules/silvertorch.py)
under `backend="triton"`, while
`backend="torch"` calls the same op names in
[`ops/reference/`](../../retrieve/src/retrieve/ops/reference/codesigned_probe_score.py)
(the eager reference) and `backend="official"` runs
Meta's own kernels through an adapter
([below](#official--metas-torchopsst-kernels-as-the-reference-backend)).

Phase 1 (centroid `q @ centroids^T + topk` to pick the top-`n_probe`
clusters) runs **host-side** in `SilverTorch.forward`; both kernels are
"phase-2+3 fused" — for each `(query, probed-item)` cell each kernel
runs its predicate inline and (when it passes) scores via INT8
dequantize + dot. Items failing the predicate get score `-inf`.

**The layout: cluster-sorted CSR, compact probe rows.** Every backend
registers the same index (`indexing.ivf.csr_layout`): `item_codes` in
cluster-sorted order, `cluster_offsets [n_lists + 1]`, `cluster_sizes`,
`sort_perm` (sorted position → original id) and `inv_perm`, with the
filter buffers (`item_clause_attrs`, the bloom index) in the same sorted
order. A query row's probed clusters are laid out back to back, so
the scorer's output is `[B, width]` with
`width = indexing.ivf.probe_width(cluster_sizes, n_probe)`, the sum of the
`n_probe` largest clusters. That is a static bound on any row's item count
(the probed clusters are distinct), cached as the Python int
`SilverTorch._probe_width`, rederived by `set_query_params` and by the
load hook. Slots past a row's items score `-inf`. This replaced a padded
layout of width `n_probe · max_cluster_size`, which read one `-1` pad
per slot: 611,520 slots against 64,757 on goodreads at `n_probe` 24, whose
largest cluster holds 25,480 items (roadmap G-a, TF-9).
`SilverTorch.register_index` / `set_query_params` raise unless
`width ≥ k`, so `topk(k)` needs no pad path.

**How a program finds its items** (`common.probe_tile`). There is no host
table. Each program loads its row's `n_probe` cluster ids as one vector
(`NPP`, the next power of two), gathers their offsets, and builds the
row's tile and slot prefix sums with `tl.cumsum`. Its tile is then
located by a masked vector reduction. Tiles are **cluster-aligned**: a
tile lies inside one cluster (at most one partial tile per probe), so its
lanes read `lo + off + arange(BLOCK_P)`, one contiguous run of code rows.
The grid carries `n_probe` extra tiles. Tiles past the row's cluster
tiles only store the `-inf` tail over `[total, width)`, which is 70 % of
the width on goodreads. A lane past its cluster's end holds the next
cluster's slot, so the score store is masked by in-cluster validity, not
by the predicate.

**Launch grid** `(B, tiles_y, tiles_x)` via `_host.grid_batch_tiles`, the
batch on `grid_x`. The rows' early probes then run concurrently, and
batch-mates that probe the same clusters read them from L2 (a 16-query
arXiv batch probes 321 distinct of 384 `(row, cluster)` pairs). Measured
against a tile-first grid: arXiv none 112 → 91 µs, bloom 126 → 107 µs
([artifacts](../artifacts/kernel-opt/predictions.md)).

**Ids after the top-k** (`common.probe_ids_kernel`, one launch after
`torch.topk`). The scorer writes scores only. A one-program-per-row
kernel maps each winning slot back to its sorted position with the same
per-row table, and writes `sort_perm[pos]`, or `-1` at every `-inf`
slot. That is the `masked_topk` sentinel contract the other backends meet
(`interfaces.py`). Writing ids from the scorer measured 13–20 µs slower
on arXiv, and eight torch ops per epilogue measured ~230 µs of eager
launch overhead (57 launches against the padded design's 38). The
current form is 32–44 launches. Both launches stay textually inside
each `@triton_op` body (`_host.probe_prep` / `probe_topk` build their
arguments).

### `codesigned_probe_score` — IVF + INT8 + Bloom

[`ops/triton/codesigned_probe_score.py`](../../retrieve/src/retrieve/ops/triton/codesigned_probe_score.py).

The `int8 → fp32` code cast never touches HBM. Bloom mode reads the
**transposed index** of the paper's "rotate the matrix" phase 2 (roadmap
G-a, TF-1): `bloom_transposed [m_bits, ceil(N/64)]`
(`bloom_hash.build_transposed_sigs` over the cluster-sorted row-wise
signatures), in which bit `pos % 64` of word `pos // 64` of row `m` is bit
`m` of item `pos`'s signature. The query comes in as its set-bit positions,
`[B, C·k_hash]` from `bloom_hash.build_query_bit_positions`, with `-1` for an
inactive clause's slots. The same `_bit_positions` hash core as
`build_query_signatures` is used, so the positions are the query
signature's bits exactly (pinned in `test_bloom_hash.py`). The kernel ANDs
one word per set bit: `(qb & sig) == qb ⇔` every set bit `m` of `qb` is set
in `sig`. At the shipped settings that is ≤ 10 reads of an 8-byte word
shared by 64 items, instead of a 128-byte row per item. The result is
boolean-identical to the row-wise test (the parity file's private
oracle). The bloom inputs inherit the `(clause_idx, value)` keying
invariant documented under `bloom_match`.

**Tile config.** `CodesignedProbeScoreConfig(block_p, num_warps,
num_stages)` — shipped as `DEFAULT_CONFIG` on the kernel module; pass
`config=` to override. `256 × 4` was re-swept on the compact layout and
stays the best of 9 configurations on both goodreads and arXiv. Re-tune
on a new arch via `uv run tune-kernels codesigned-probe-score`. `HAS_QB`
is a body-level constexpr (the bloom-on and bloom-off paths JIT-specialise
on it). The score buffer is `torch.empty([B, width])`: every slot is
written (a dot, or `-inf`), so there is no pre-fill.

### `codesigned_probe_score_exact` — IVF + INT8 + exact AND-of-OR

[`ops/triton/codesigned_probe_score_exact.py`](../../retrieve/src/retrieve/ops/triton/codesigned_probe_score_exact.py).

Same layout, launch shape and IVF + INT8 scoring path as
`codesigned_probe_score`; swaps the bloom subset test for an exact
AND-of-OR attribute predicate fully unrolled over `(C, A_max)` against the
cluster-sorted `[N, C, A_max]` narrow item attrs (SilverTorch's
`item_clause_attrs` buffer) and `[B, C]` query attrs. Per program: inner
OR over `A_max` attribute slots per clause, outer AND over `C` clauses,
reverse-XOR per clause, with `q_c == -1` overriding to "always passes" —
the same `common.clause_pass` helper as the standalone `clause_mask`
kernel (`ids=pos`, `load_mask=valid`), fused into the score path and
applied only to the probed items. No false positives (cf. bloom mode).
It reads `C·A_max` int64 attrs per item, 160 B at arXiv's `C = 5, A_max = 4`
and more than the code row. This is why exact mode is the slowest of the
three here (validation's head-to-head).

**Tile config.** `CodesignedProbeScoreExactConfig(block_p, num_warps,
num_stages=3)` — shipped as `DEFAULT_CONFIG` on the kernel module;
default `block_p=256, num_warps=4`, re-swept on the compact layout. Re-tune
via its own subcommand, `uv run tune-kernels codesigned-probe-score-exact`,
whose regime axes are `(N, B, C, A_MAX)` clause shapes (repeatable
`--regime`). The score buffer is `torch.empty([B, width])`, written in
full, with the same id epilogue.


### `official` — Meta's `torch.ops.st.*` kernels as the reference backend

[`ops/official/`](../../retrieve/src/retrieve/ops/official/adapter.py) (`__init__.py`: loader,
`OfficialConfig`, the upstream constants and `st = torch.ops.st`; `adapter.py`: the rest).
Not a kernel of ours: an adapter over the ops of
[meta-recsys/silvertorch](https://github.com/meta-recsys/silvertorch)
(pinned at `21aa35e`, the `official` extra), selected by
`SilverTorch(backend="official")`. This section is the *what runs*; the
upstream facts it relies on were measured by
[`official_facts.py`](../artifacts/official-silvertorch/official_facts.py)
([artifacts](../artifacts/official-silvertorch/README.md)). Its parity
gate state is in [validation](../validation.md#library-gates).

**Layout.** Phase 1 is ours and shared: the same `KMeans`,
the same `quantize_int8_global` codes, the same centroid top-`n_probe`.
The official scorer reads the same cluster-sorted CSR every backend
registers ([SilverTorch kernels](#silvertorch-kernels)). Forward passes
`cluster_ids = probe_ids [B, n_probe]`, `cluster_length =
cluster_sizes[probe_ids]` and `max_tensor_size_per_row` equal to the
compact width `_probe_width` the Triton scorer writes. The official output
is therefore our `[B, width]` (rounded up to 32), and the host
`masked_topk` costs the same in every arm. Returned indices are sorted
positions and map back through `sort_perm`. Slots the op never writes
(pads, filtered docs) carry `indices == -1` and become `-inf` / `-1`.

**Ops called** (schemas in the module docstring, matched to the upstream
`TORCH_LIBRARY_FRAGMENT` registrations): `fused_kmean_ann`,
`fused_kmean_ann_with_partial_masks`, `bloom_index_build`,
`parse_expression_query_batch` (CPU),
`bloom_index_search_batch`, `bloom_index_search_batch_return_partial_response`.
Nothing else upstream is used — no `BloomIndexSearchModule`, no
`is_topk`, no `*_multiple` sharded variants.

**Scores** (`OfficialConfig.score_path`). `"int32"`:
`divisor_for_int8=-1` returns the raw int32 dot; the host epilogue
`(dot.float() · q_scale[b]) · global_scale` is the same two
left-associated fp32 multiplies as the Triton kernel and the torch path,
so scores are **bit-identical** (the parity contract). `"fp16"`
(default — the instantiation Meta ships for int8 serving, hence the
timed path): `divisor_for_int8 = default_divisor(D)`, the smallest power
of two with `127²·D / divisor ≤ 65504` (16 / 32 / 64 at D = 64 / 128 /
256); the kernel writes `fp16(dot / divisor)` — exact division, one fp16
rounding (relative ≤ 2⁻¹¹) — and the epilogue multiplies by `divisor ·
q_scale · global_scale`. `per_embedding_scale` is not used: upstream
casts the int32 dot to fp16 *before* dividing by it, which overflows for
any realistic `D`.

**Filter modes.** `none` → `fused_kmean_ann`. `exact` → our Triton
`clause_mask` over the cluster-sorted `item_clause_attrs`, packed by
`pack_mask` into the scorer's `filtering_bit_mask` (int64 `[B,
ceil(N/64)]`, doc `d` at bit `63 − d % 64` of word `d // 64` —
`MASK_BIT_ORDER`, see below) and passed to `fused_kmean_ann`: phase 2
ours and full-`N`, labelled so in every table. `bloom` → **Meta's
bloom**, not ours: at `register_index`, `attrs_to_features` turns the
sorted `[N, C, A_max]` attrs into the jagged `(feature_ids int32 [C],
feature_offsets int64 [N·C+1], feature_values int64)` layout — the
clause index is the feature id, the same `(clause, value)` keying as our
salt — and `bloom_index_build(b_multiplier, k)` builds `bloom_index [W]`
+ `bundle_b_offsets`; per forward, `queries_to_expressions` renders each
`[C]` query row as `"0:v0 AND 1:v1"` (`NOT c:v` for reverse clauses,
`""` = match all), `parse_plans` runs the CPU parser (plans kept on
CPU; memoised per distinct expression tuple by default —
`OfficialConfig.cache_plans=True` — or parsed on every forward with
`cache_plans=False`), and then either
(`bloom_path="partial"`, default — the paper's co-design)
`bloom_index_search_batch_return_partial_response` over the probed
clusters feeds `fused_kmean_ann_with_partial_masks`, or
(`"full"`, the S9 ablation) `bloom_index_search_batch(return_bool_mask=
False)` over all `N` feeds `fused_kmean_ann(filtering_bit_mask=…)`. The
search `k` is our `k_hash` (must be ≤ 10: `MAX_K_V2`, a fixed array
size the upstream kernel never checks), `OfficialConfig.n_stored_hashes`
(default 7; the ops' `hash_k` argument, renamed because `k_hash` is the
library-wide name of the search `k`) is the number of raw hashes stored
per term, `build_k` defaults to the
search `k`. A bloom index registered without attributes is empty and a
later query with attributes raises.

**Bit order — measured, high-first.** The upstream scorer reads mask
words high-first ("lower doc id put at higher bits",
`bloom_index_util.cuh`), and the packed bloom output is stored the same
way; both are constants in the adapter (`MASK_BIT_ORDER`,
`BLOOM_OUTPUT_BIT_ORDER`) and `bloom_filtering_mask` bit-reverses per
word if they ever differ. The scorer's order is measured on the A100
three independent ways
([official_facts](../artifacts/official-silvertorch/official_facts.txt)) — the packed
search output for a doc-0 predicate is `0x8000000000000000`, a
hand-built mask with bit 63 set scores doc 0 (bit 0 scores doc 63), and
the packed output round-trips as a scorer mask — so both constants are
`"high_first"`. `test_official.py` pins that value as
`OFFICIAL_BIT_ORDER` and T3 asserts the adapter constants equal it,
with the other order kept as a negative control (nothing scored).

**Eager only.** Every op syncs the host (`repeat_interleave`
without an output size, `.item()` on cumsums, a per-call host decode and
pageable upload of the plans), and the partial-response output
shape is data-dependent, so there is no `torch.compile` or CUDA-graph
path: `SilverTorch.compile()` raises on this backend and a compiled
forward raises `RuntimeError` when traced. Measured on the A100
(`torch.profiler` + fd-2 sync counting): `fused_kmean_ann` is
**19 launches and 3 host syncs** per unfiltered forward (only 2 of the
19 are the scoring kernels; the rest is payload prep — scans, a
`repeat_interleave`, fills), `fused_kmean_ann_with_partial_masks` 19 / 4,
the partial-response bloom search 13 / 2 with two H2D plan uploads, so a
bloom forward is ≈ 32 launches and ≥ 5 syncs against Triton's one launch.
On top of that the CPU expression parse costs ≈ 59 µs per call at B=16
(`c:v AND c:v`), 10–20 % of an eager bloom forward, which the
`parse_plans` LRU cache hides after the first call for a repeated
batch. **A timing run must therefore set `OfficialConfig(cache_plans=
False)`** — every forward pays the parse, as serving fresh queries does
— **or report both settings, labelled**; results are identical either
way (T6 checks the uncached path bit for bit, T7 records its sync
count). The official arm loses at small `P` / `B=1` for host reasons, so
the kernel-only tier of the head-to-head is what compares kernels.

**Measured.** The head-to-head against Triton (end to end, kernel-only,
phase 2, parity, memory) is in
[validation](../validation.md#official-against-our-triton-reimplementation-citable-contested),
with tables in [artifacts](../artifacts/official-silvertorch/b3/tables.md).
Both scorers now read the same CSR. The b3 split (Meta's
scorer 10.9–17.8× ours on goodreads) was our former padded probe layout,
and it closed with TF-9 / TF-1. The current kernel-only numbers, re-run on
the b3 methodology, are in the same validation section.

### Deleted backends

A hand-written CUDA C++ SilverTorch backend and its one-to-one port into
NVIDIA's CUTLASS Python DSL were deleted once the official backend's
parity gate was green; the git tag `cuda-cute-backends-final` holds them
([decisions](../decisions.md#library)). Both ran the paper's two-kernel
design — a phase-2 mask kernel over a *transposed, cluster-major* bloom
index (or a clause-mask kernel for `"exact"`) producing 1-bit-per-item
masks, then a masked `__dp4a` scorer — and were bit-identical to the
Triton kernels on scores.

Their measurements stand (raw outputs:
[artifacts/cute-dsl-scorer/](../artifacts/cute-dsl-scorer/README.md)): the
transposed index reads ~40× fewer filter bytes than the row-wise Triton
form and is 2.1–2.5× faster kernel-only (1.3× wall) once the scorer has
memory-level parallelism; a DSL port is kernel-for-kernel within noise of
the C++ backend, its cost being host launch overhead; and every backend
collapses to one latency under CUDA-graph replay. The transposed index
is back in Triton, over the cluster-sorted order rather than per-cluster
word spans: [`codesigned_probe_score`](#codesigned_probe_score--ivf--int8--bloom).

## Adding an op

A new Triton kernel touches every one of these; missing one is the usual
way a kernel ships half-integrated.

1. `ops/triton/<k>.py` — the `@triton.jit` body, the `<Name>Config`
   dataclass, `DEFAULT_CONFIG`, the shared `_<k>_prep` / `_<k>_finish`
   pair, the `_<k>_impl`, and the public op (`@torch.library.triton_op`
   unless the launch's store address is data-dependent, in which case
   `@torch.library.custom_op` — see "Graph-break behavior" above).
2. `ops/triton/_load.py` — import the new module so `import
   retrieve.ops.triton` registers the op.
3. `ops/reference/<k>.py` — the pure-torch twin, same op name and
   signature. This is the parity oracle by default
   ([testing.md](testing.md#oracle-policy)) and the `"torch"` backend.
4. `ops/tune.py` — a `KernelTuneSpec` in `KERNELS` plus a curated entry
   in `DEFAULT_CONFIG`, so `tune-kernels` covers it and
   `test_tune_smoke.py` catches schema drift between `tune.py` and the
   kernel's `_impl`.
5. `retrieve/tests/parity/test_<k>.py` — kernel vs. reference, following
   [testing.md](testing.md#new-parity-test-for-a-triton-kernel).
6. A compile/graph-break gate: a row in `test_silvertorch_compile.py` or
   `test_linr_compile.py` if a module's forward calls the op under
   `torch.compile`, or a dedicated case in `tests/compile/` if it
   doesn't already fall under one of those.
7. This file — a kernel section with its inputs/outputs, launch grid,
   tile config and any numerics caveat.
8. [architecture.md](architecture.md)'s op table (`## Kernels`) — one row
   naming the op and its consumer module.
9. A `DISPATCH` row in `interfaces.py` if a new module uses the op —
   [architecture.md](architecture.md#backend-dispatch) is derived from
   it, not maintained separately.
