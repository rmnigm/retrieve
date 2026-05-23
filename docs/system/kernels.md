<!-- claude code generated file -->

# `retrieve` kernels

> Previously: `retrieve/docs/kernels.md` (originally `retrieve/docs/KERNELS.md`).

The Triton kernels split into three trees by domain:

- [`linr/`](../../retrieve/src/retrieve/kernels/linr/) — kernels
  used by `PrefilterKNN` / `OneBitKNN` (`fused_masked_knn_topk`,
  `oporp_1bit_match_topk`). `PostfilterKNN`'s dense fp16 matmul +
  top-K and `PostfilterKNNInt8`'s int8 `_int_mm` + int32 top-K are
  both pure torch — there's no real fusion to win over cuBLAS LtGemm +
  CUB.
- [`filters/`](../../retrieve/src/retrieve/kernels/filters/) —
  standalone filter primitives consumed by the `FilterModule` family:
  `clause_compact` (powers `ExactAttributeFilter.evaluate_indices`),
  `clause_mask` (powers `ExactAttributeFilter.evaluate_mask`), and
  `bloom_compact` (powers
  `BloomFilter.evaluate_indices`). The mask/compact split mirrors the
  filter API split documented in [filtering.md](filtering.md).
- [`silvertorch/`](../../retrieve/src/retrieve/kernels/silvertorch/) —
  the two co-designed IVF probe+score kernels (`codesigned_probe_score`
  for the IVF + INT8 + Bloom co-design, `codesigned_probe_score_exact`
  for the IVF + INT8 + exact AND-of-OR variant — `SilverTorch.filter`
  picks one) plus `bloom_match` (lives in this tree for historical
  reasons but is now a cross-tree filter primitive consumed by
  `BloomFilter`).

The LinR kernels are the focus of this doc; the filter primitives
(`clause_compact`, `clause_mask`, `bloom_match`, `bloom_compact`) are
covered next, and the SilverTorch-only kernels at the end for context.

All kernels follow the same conventions:

- One launch per `forward()` call. No persistent threads, no streams.
- Inputs are CUDA tensors with explicit strides; the host wrapper passes
  `.stride(i)` for every axis instead of assuming contiguity.
- Top-K selection is **not** in-kernel. Each kernel writes a `[B, ·]` score
  buffer and the host calls `torch.topk` on it. CUB's top-K (under torch)
  is faster than anything we can implement in pure Triton without a
  warp-level radix-select primitive.
- Tile config (`block_n`, `num_warps`, `num_stages`; `block_p` for the
  silvertorch kernels) is offline-tuned per kernel and shipped as a
  single `DEFAULT_CONFIG` constant on the kernel module. No runtime
  `@triton.autotune`. Callers who want a non-default tile pass
  `config=<Kernel>Config(...)`; for `@custom_op`-wrapped kernels the
  override goes to the private `_<name>_impl(..., config=)` companion
  (the public op has a fixed schema and always uses `DEFAULT_CONFIG`).
  The `tune-kernels` CLI (shipped with the library at
  `retrieve.tune:main`) sweeps the candidate grid on a given arch and
  prints the line to paste into the kernel file. The same convention applies uniformly across linr, filter, and
  silvertorch kernels — see [Autotune separation](#autotune-separation)
  below for the rationale.
- Graph-break behavior. Every host wrapper in this tree is decorated
  with either `@torch.library.custom_op` + `register_fake` (the five
  filter/compact kernels — `clause_mask`, `clause_compact`,
  `bloom_compact`, `fused_masked_knn_topk`) or
  `@torch.library.triton_op` + inline `wrap_triton(_kernel)[grid](...)`
  (the silvertorch + onebitknn kernels — `bloom_match`,
  `codesigned_probe_score`, `codesigned_probe_score_bloom`,
  `codesigned_probe_score_exact`, `oporp_1bit_match_topk_full`,
  `oporp_1bit_match_topk_indirect`). Both decorators stop dynamo from
  graph-breaking at the wrapper boundary, so each layer's full forward
  captures into one cudagraph_trees graph; `triton_op` additionally
  lets inductor see the underlying `@triton.jit` kernel (preserves the
  reference under `torch.export`; opens epilogue-fusion headroom for
  Stage 3).

## Autotune separation

The kernels were originally tuned via `@triton.autotune` (linr) or
module-level `_BLOCK_N` / `_NUM_WARPS` constants (filters). Both
patterns had drawbacks:

- `@triton.autotune` re-tunes at every cache-key shape change, leaking
  compile pressure into the cudagraph-trees capture for
  `torch.compile(dynamic=True, mode="reduce-overhead")` and re-running
  the autotune sweep across `tl.atomic_add` kernels corrupts output
  buffers across trials (the compact kernels can't safely autotune
  in-kernel).
- Hard-coded `_BLOCK_N = 256` etc. were guesses, not measurements;
  picked once and never re-checked against real-eval shapes.

The shipped pattern, applied uniformly to every kernel in this tree:

1. A `@dataclass(frozen=True) class <Name>Config` next to the
   `@triton.jit` body holds `block_n`, `num_warps`, `num_stages`.
2. A `DEFAULT_CONFIG` module constant holds the single curated
   default for the current arch (sm_80 / A100 in this repo).
3. The host wrapper takes `config: <Name>Config | None = None`;
   `cfg = config if config is not None else DEFAULT_CONFIG` resolves it.
4. For both decoration styles the same private `_<name>_impl(..., *,
   config: ...Config | None = None)` lives next to the public wrapper.
   `@custom_op`-wrapped kernels (`clause_mask`, `clause_compact`,
   `bloom_compact`, `fused_masked_knn_topk`) delegate to `_impl` with
   `config=None`. `@triton_op`-wrapped kernels (`bloom_match`, the two
   `codesigned_probe_score*` files, `oporp_1bit_match_topk`) instead
   inline a near-duplicate of `_impl`'s body — the duplication is
   intentional so that `wrap_triton(_kernel)[grid](...)` appears
   textually in the decorated function's source, which is what
   `torch.export`'s kernel registry walks to preserve the kernel
   reference. The schema doesn't carry the dataclass either way; tests
   and the tuner reach `_impl` directly to pass an override.
5. Tuning is offline: `retrieve/src/retrieve/tune.py` (`uv run
   tune-kernels <kernel-subcommand>`) sweeps a hard-coded `(block_n,
   num_warps)` grid against a built-in shape regime list mirroring
   real-eval workloads (catalog sizes from
   `evaluation/data/<dataset>/item_attrs_narrow.pt`, batch sizes from
   `evaluation/retrieval/config.py`). Filter-kernel subcommands also
   accept repeatable `--regime N,B,C,A_MAX` (or `N,B,W`) flags so
   end-users can tune for their own catalog shapes. Picks one default
   per arch via plurality vote across regime winners; emits a pasteable
   `DEFAULT_CONFIG = ...` line. Re-run once per new arch; commit the
   line.

For the compact kernels, the offline tuner avoids the `atomic_add`
hazard naturally: it calls the host wrapper, which allocates fresh
`out_indices` (`-1`-filled) and `counts` (zeros) on every call — `do_bench`
reps each pay one allocation, so no cross-rep accumulation. The
warning that lived on the kernel files about
`@triton.autotune`-time corruption still applies to in-kernel
autotune; the offline path is unaffected.

## Score conventions

- Real-valued similarity (V1, V2): plain dot product, fp32. Higher is better.
  `-inf` marks masked-out / padded positions.
- 1-bit Sign-OPORP (V3): `D - 2 * popcount(query_bits ^ item_bits)`, fp32.
  `D = 64 * W`. This is the standard Hamming-to-dot-product relation for
  sign-quantized vectors. Higher is better. Same `-inf` sentinel.

## OPORP layout

Used only by V3. Built once at index time by [`quantize_oporp_1bit`](../../retrieve/src/retrieve/layers/utils/quantize.py):

- `signs[D]` int8 ∈ {-1, +1} — Rademacher (random sign vector).
- `perm[D]` int64 — permutation of `[0, D)`.
- `item_bits[N, W]` int64, `W = D // 64`. Bit `b` of word `w` is set iff
  `(items * signs)[perm][..., 64*w + b] > 0`.

Queries land in the same bit space via [`project_oporp_1bit_query`](../../retrieve/src/retrieve/layers/utils/quantize.py):
`query_bits = pack_signs(((query * signs)[perm]) > 0)`.

The projection is **deterministic**: the seed determines `signs` and `perm`,
so loading the same checkpoint produces the same bits. V3 store (`item_bits`,
`signs`, `perm`) as buffers; both backends call the same projection helper
on every forward, so the torch reference and the Triton kernel see byte-for-
byte identical bits.

## PostfilterKNN dense path — pure torch, no kernel

`PostfilterKNN`'s forward is `query @ item_embs.T` + optional
`masked_fill(-inf)` + `torch.topk` — implemented directly in
[`PostfilterKNN`](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py).
An earlier `fused_matmul_topk` Triton kernel sat in this slot, but it only
fused the matmul: it materialized the full `[B, N]` score buffer to global
memory and then called the same host-side `torch.topk`, so its memory
traffic and selection cost matched cuBLAS + CUB exactly. With no fusion
benefit, the kernel was removed; `PostfilterKNN` accepts the
`backend=` flag for API symmetry but both values dispatch to this same
pure-torch path.

## PostfilterKNNInt8 dense path — pure torch, no kernel

`PostfilterKNNInt8`'s forward is `torch._int_mm(query_codes,
item_codes_T)` (int8×int8 → int32, cuBLAS LtGemm, IMMA tensor cores on
sm_80+) + optional `masked_fill(int32_min)` + `torch.topk` on the int32
result — implemented directly in
[`PostfilterKNNInt8`](../../retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py).
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

[`kernels/linr/fused_masked_knn_topk.py`](../../retrieve/src/retrieve/kernels/linr/fused_masked_knn_topk.py).

Scores only the items in a precompacted `positive_indices` buffer (gather
+ dot + write). Returns `(ids[B, K], scores[B, K])` with `-1` / `-inf`
padding when fewer than `K` candidates pass.

```
inputs:   query             [B, D]    fp32
          item_embs         [N, D]    fp32
          positive_indices  [B, P]    int64
          counts            [B]       int64
output:   scores            [B, P]    fp32  (kernel-internal)
return:   ids               [B, K]    int64
          scores            [B, K]    fp32
```

**Launch grid** `(B, cdiv(P, BLOCK_N))`. Each program owns one
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

**Bucketing.** `P` is rounded up via `_bucket_p` to one of `{256, 2048,
16384, 131072, 1048576}` and passed as `tl.constexpr` to the kernel,
so the JIT cache compiles once per bucket × D regardless of how
`counts.max()` shifts across calls (the role this used to play as the
autotune cache key). The host wrapper allocates the score buffer at
`[B, P_BUCKET]` so the kernel's `n_offsets < P` store mask sees a
stable width; lanes in `[P_real, P_BUCKET)` get `-inf` automatically
because `count[bid] <= P_real`, so the `in_count` mask gates them. The
caller-supplied `positive_indices` stays at width `P_real`; the
post-topk `gather` clamps `topk_local` to `P_real - 1` before
indexing (the `where(isfinite, …, -1)` mask then overwrites those
slots with the `-1` sentinel).

## `oporp_1bit_match_topk` — V3 (all paths)

[`kernels/linr/oporp_1bit_match_topk.py`](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py).

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

**Popcount** is the [SWAR bit-twiddle](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py)
(`_popcount_int64`): five mask-shift-add steps, no libdevice dependency.
The matching torch reference [`popcount_int64`](../../retrieve/src/retrieve/layers/utils/quantize.py)
uses the exact same algorithm so torch and Triton produce **bit-exact**
identical scores.

**Launch grid** `(B, cdiv(n, BLOCK_N))`, modeled on `bloom_match` (the
already-winning kernel of identical shape: int64-word reduction over `W`).

**HAS_INDICES path**: replaces the contiguous `n_off` load with an
indirect `pos_indices[b, n_off]` lookup; otherwise identical. Used for
candidate-set rerank and for the masked V3 path (after `compact_mask`).

**Tile config.** `Oporp1BitMatchTopkConfig(block_n, num_warps,
num_stages)` — shipped as `DEFAULT_CONFIG` on the kernel module; pass
`config=` to override. Re-tune on a new arch via `uv run tune-kernels
oporp-1bit-match-topk`.

**Bucketing.** For the `HAS_INDICES=True` path, the score-buffer
width is `n_kernel = max(_bucket_n(positive_indices.shape[1]),
_bucket_n(k))` (buckets `{4096, 65536, 1048576, 16777216}`), passed
as `tl.constexpr N`. Bucketing `n_loop` gives the same JIT-cache
invariant as `fused_masked_knn_topk`; taking the max with
`_bucket_n(k)` guarantees the buffer always has at least K lanes so
`torch.topk(all_scores, k)` works directly without a host-side pad
tail. `positive_indices` stays at its caller width; the kernel's
indirect `pos_indices` load is gated by `in_count = (n_off <
count[bid])` (not `n_valid = (n_off < N)`) so it never reads OOB
when `N > n_loop`. The post-topk
`safe_local.clamp_max(n_loop - 1)` + `where(isfinite(scores), ids,
-1)` tail handles the per-row "ran short" case (rows where
`counts[b] < k` get `-1` sentinels in the bottom slots). For
`HAS_INDICES=False`, `N = item_bits.shape[0]` is fixed per
registered index — `OneBitKNN.register_index` asserts `k <= N`, no
bucketing needed.

## `clause_compact` — fused clause eval + stream compaction

[`kernels/filters/clause_compact.py`](../../retrieve/src/retrieve/kernels/filters/clause_compact.py).

Powers `ExactAttributeFilter.evaluate_indices`. Avoids materializing the
dense `[B, N]` bool that `evaluate_mask` would otherwise produce, then doing
a host-side argsort to compact it. One launch produces the
`(positive_indices, counts)` pair that `PrefilterKNN` and `OneBitKNN`'s
sparse paths consume.

```
inputs:   item_clause_attrs   [N, C, A_max]  int64
          clause_is_reverse   [C]            bool
          query_clause_attrs  [B, C]         int64
output:   out_indices         [B, N]         int64  (worst-case scratch)
          counts              [B]            int64
return:   positive_indices    [B, P]         int64  (P = max(counts.max(), 1))
          counts              [B]            int64
```

**Launch grid** `(B, tiles_y, tiles_x)` — batch on `grid_x` so adjacent
dispatched programs share the same item tile (good L2 reuse on the
`[N, C, A_MAX]` item-attrs read); the `cdiv(N, BLOCK_N)` tile count is
split across `grid_y × grid_z` to dodge the 65,535 cap on a single
axis (which would otherwise overflow at N>~16M with `block_n=256`). The
kernel reconstructs `tile_id = tile_x * tiles_y + tile_y`. Each program
owns one `(query, n-tile)` cell and produces:

```
per program (b, tile):
    pass_mask = AND over clauses of (any item attr == query attr) ^ reverse
    intra     = tl.cumsum(pass_mask) - 1               # intra-tile offset
    base      = tl.atomic_add(counts[b], tile_sum)     # row base offset
    tl.store(positive_indices[b, base + intra], item_id, mask=pass_mask)
```

**Clause loop** is fully unrolled (`C` and `A_MAX` are `tl.constexpr`):
inner OR over the `A_MAX` attribute slots per clause, outer AND over the
`C` clauses, with the reverse flag XORed in per-clause and `q_c == -1`
overriding to "always passes."

**Output ordering** within a row is **unspecified** — atomics across tiles
race with each other. Downstream consumers (`fused_masked_knn_topk`,
`oporp_1bit_match_topk` HAS_INDICES path) only care about the *set* of
passing ids, so this is fine. Callers that need a deterministic order must
sort.

**Tile config.** `ClauseCompactConfig(block_n, num_warps, num_stages)`
— shipped as `DEFAULT_CONFIG` on the kernel module; tests/tuner override
via `_clause_compact_impl(..., config=)`. Re-tune on a new arch via
`uv run tune-kernels clause-compact`. The `@triton.autotune`
hazard around `tl.atomic_add` accumulating across trials does **not**
apply to the offline tuner — see [Autotune separation](#autotune-separation).

## `clause_mask` — fused clause eval emitting `[B, N]` bool

[`kernels/filters/clause_mask.py`](../../retrieve/src/retrieve/kernels/filters/clause_mask.py).

Powers `ExactAttributeFilter.evaluate_mask` on CUDA. Same inner loop as
`clause_compact` minus the cumsum + `atomic_add` epilogue — emits the
`[B, N]` bool directly without the host-side argsort the dense path used
to need. Replaces the pure-torch broadcast that materialized
`[B, N, C, A_max]` bool — `C·A_max`× the output mask — before reducing.

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

**Inner loop** is the same as `clause_compact`'s: inner OR
over `A_MAX` slots, outer AND over `C` clauses, reverse XOR, inactive
override. The epilogue is a single `tl.store` of the `pass_mask` tile —
no cumsum, no atomics.

**Tile config.** `ClauseMaskConfig(block_n, num_warps, num_stages)` —
shipped as `DEFAULT_CONFIG` on the kernel module; tests/tuner override
via `_clause_mask_impl(..., config=)`. Re-tune on a new arch via
`uv run tune-kernels clause-mask`.

## `bloom_match` — Bloom subset test

[`kernels/silvertorch/bloom_match.py`](../../retrieve/src/retrieve/kernels/silvertorch/bloom_match.py).
Lives in the SilverTorch kernel tree for historical reasons (it was
written when only SilverTorch's bench tests consumed it) but is now a
standalone filter primitive: powers
[`BloomFilter.evaluate_mask`](../../retrieve/src/retrieve/layers/filters/bloom.py)
on CUDA. SilverTorch's in-cluster bloom is fused separately into
`codesigned_probe_score`; the two paths are independent.

Implements the conjunctive subset test `(qb & sigs) == qb` reduced over
`W` int64 words. The item-side sigs come from
`_build_signatures` ([layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py))
at `register_index` time; the per-call query bits go through the
loop-free `_build_query_signatures` path (same module). The kernel is
purely a bitwise reduction.

**Query-build path.** `_build_signatures` is a python `range()`-driven
chunk loop over the index — fine at register time (bandwidth-bound,
~128 ms for N=3M) but a poor fit for the per-forward query build, where
n=B is always small and the loop's trip count makes dynamo specialize on
shape. `_build_query_signatures` mirrors the body without the loop;
keeping it purely tensor-flow lets the outer
`torch.compile(dynamic=True, mode="reduce-overhead")` wrapped around
each algo's forward in `evaluation/retrieval/algos/` install a single
symbolic-shape graph that's reused for all B. The eager body fires
~15 separate CUDA kernels per call (~0.4 ms wall-clock, **flat in
B**) — pure launch overhead, since the actual work is microseconds;
under the algo-level cudagraph_trees capture the same work collapses
into one replay. Standalone use of `BloomFilter` (not via an Algo
wrapper) runs the function eagerly — wrap externally with
`torch.compile` if you want the cudagraph win there too.

**Hash invariant.** `_build_signatures` keys each hash on `(clause_idx,
value)` (paper §4.1: "for each feature" — a *feature* is a `(key, value)`
pair). It does this by XOR-ing a per-clause salt — `_mix64(clause_id, …)` —
into the post-`_mix64` hash, before the position mask. This is what
prevents value `V` in clause C0 from colliding with the same `V` in clause
C3 when clauses share a value vocabulary; without it, single-clause
queries on overlapping vocabularies leak ~25–30% of non-matching items as
false positives. The kernel itself is untouched: it consumes `[N, W]` /
`[B, W]` int64 buffers as opaque bits. The salt costs one extra elementwise
XOR per chunk inside `_build_signatures` (well under measurement noise vs
the existing scatter + word-pack reduction); kernel HBM traffic and launch
shape are unchanged. `bloom_sigs` buffers persisted from any pre-fix run
are stale and must be rebuilt.

```
inputs:    qb     [B, W]     int64    packed query bloom signature
           sigs   [N, W]     int64    packed item bloom signatures
output:    mask   [B, N]     bool     (qb & sigs[n]) == qb, AND over W
```

**Launch grid** `(B, cdiv(N, BLOCK_N))`, identical 2-D shape to
`oporp_1bit_match_topk` (this kernel is the structural ancestor of
that one — the popcount step is the only material difference). Each
program loads `qb[b, :]` once, then a `[BLOCK_N, W]` tile of `sigs`,
and emits `[BLOCK_N]` bool to the output buffer.

**Inner op**: `(qb[None, :] & sigs) == qb[None, :]` per-word, then
`tl.min` over the `W` axis to AND-reduce. The min-of-int trick avoids a
boolean reduction Triton doesn't have; the cast back to bool is on store.

**Wins** 10–20× over the pure-torch broadcast `(qb.unsqueeze(1) & sigs)
== qb.unsqueeze(1)).all(-1)` because the broadcast materializes
`[B, N, W]` int64 (24 GiB at `B=64, N=2M, W=16` worst case) before the
reduction; the kernel keeps the tile in registers.

**No autotune** today — fixed `BLOCK_N = 128 if N >= 128 else
next_power_of_2(N)`. The fused `bloom_compact` (next section) shares
this kernel's inner subset-test loop and adds `clause_compact`'s
cumsum + `atomic_add` tail, but uses the 3D launch grid the compact
kernels need at large N.

## `bloom_compact` — fused subset test + stream compaction

[`kernels/filters/bloom_compact.py`](../../retrieve/src/retrieve/kernels/filters/bloom_compact.py).

Powers `BloomFilter.evaluate_indices` on CUDA. Combines `bloom_match`'s
subset-test inner loop with `clause_compact`'s cumsum + `atomic_add`
epilogue, so the dense `[B, N]` bool plus host-side argsort that the
ABC fallback (`compact_mask(bloom_match(.))`) would otherwise produce
never materializes.

```
inputs:    qb     [B, W]   int64    packed query bloom signature
           sigs   [N, W]   int64    packed item bloom signatures
output:    out_indices [B, N]   int64    (worst-case scratch)
           counts      [B]      int64
return:    positive_indices [B, P] int64  (P = max(counts.max(), 1))
           counts           [B]    int64
```

**Launch grid** `(B, tiles_y, tiles_x)`, same 3D shape as the clause
compact/mask kernels (`tile_id = tile_x * tiles_y + tile_y`). Each
program loads `qb[b, :]` once, the `[BLOCK_N, W]` `sigs` tile, computes
`(qb & sigs) == qb` AND-reduced over `W` (lifted from `bloom_match`),
then runs the same `cumsum → atomic_add → store` epilogue as
`clause_compact`. Wide `W` (=16 at the default `m_bits=1024`) makes
this kernel register-pressure-bound; at large `block_n` with few warps
it spills catastrophically (5–15× slowdown observed at `block_n≥512,
num_warps≤4`). The shipped default keeps `block_n` moderate and warps
high to stay off that cliff.

**Output ordering** within a row is **unspecified** — same convention as
`clause_compact`. V2's `fused_masked_knn_topk` and V3's HAS_INDICES path
consume the *set*, not the order.

**Tile config.** `BloomCompactConfig(block_n, num_warps, num_stages)` —
shipped as `DEFAULT_CONFIG`; tests/tuner override via
`_bloom_compact_impl(..., config=)`. Re-tune via `uv run tune-kernels
bloom-compact`. The atomic-add hazard around in-kernel autotune
is real but does not affect the offline tuner — see
[Autotune separation](#autotune-separation). `qb` is built host-side via
`_build_signatures`; folding it into the kernel adds register pressure
with no obvious win and is explicitly out of scope. The same
`(clause_idx, value)` keying invariant documented under `bloom_match`
applies — kernel is opaque to bits, so no kernel-side change.

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
| [`PostfilterKNN`](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py) | always dense | none — pure torch `(q @ x.T).masked_fill(...).topk` |
| [`PostfilterKNNInt8`](../../retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py) | always dense | none — pure torch `torch._int_mm(...).masked_fill(...).topk` (int8×int8 → int32, IMMA) |
| [`PrefilterKNN`](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py)         | masked       | `compact_mask` → `fused_masked_knn_topk`          |
|                                                                | unmasked     | none — pure torch dense path (nothing to pre-filter) |
| [`OneBitKNN`](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py)             | full         | `oporp_1bit_match_topk` (HAS_INDICES=False)       |
|                                                                | masked       | `compact_mask` → `oporp_1bit_match_topk` (HAS_INDICES=True) |
|                                                                | candidates   | `oporp_1bit_match_topk` (HAS_INDICES=True)        |

`OneBitKNN`'s masked path always compacts and uses HAS_INDICES — popcount is
cheap enough that the gather penalty never crosses the dense-fallback
break-even point. `PostfilterKNN`'s dense fp32 path is also kept as the
default (no compaction) because cuBLAS + `torch.topk` already handle the
inline-mask case at the same cost a tile-fused kernel would.

## Helpers

### [`compact_mask`](../../retrieve/src/retrieve/layers/utils/compact.py)

Bool `[B, N]` → `(positive_indices[B, P], counts[B])` where
`P = max(counts)`. Implementation: `mask.sum(1)` for counts, then
`mask.float().argsort(descending=True, stable=True)[:, :P]` for the
indices. Sync point — does `int(counts.max().item())` to size the slice.

Rows shorter than `P` carry arbitrary item ids past `counts[b]`; downstream
kernels must use `counts` to bound valid reads.

### [`popcount_int64`](../../retrieve/src/retrieve/layers/utils/quantize.py)

Torch-side SWAR popcount. Required because this PyTorch (2.10.0+cu128)
lacks `Tensor.bitwise_count`. Returns int32 to keep the downstream sum
narrow. Matches the kernel's `_popcount_int64` step-for-step — useful
property for any future torch-vs-Triton bit-exact assertion.

### [`quantize_oporp_1bit`](../../retrieve/src/retrieve/layers/utils/quantize.py)

Build-time only. `O(D)` parameter cost — the `signs` vector and `perm`
permutation are the entire projection. Apply with one elementwise multiply
and one `index_select`, no matmul needed. The `_pack_signs_to_int64`
helper packs the sign-quantized output into `[..., W]` int64 words using
`<<` and `sum(-1)`; same packing used both at index time and at query time.

### Compile on the V3 torch reference

The pure-torch hot bodies on V3's reference path —
`project_oporp_1bit_query` and `OneBitKNN._score_full` — are pure
tensor-flow free of `.item()` and Python control flow, so the outer
`torch.compile(dynamic=True, mode="reduce-overhead")` wrapped around
each algo's forward in `evaluation/retrieval/algos/` traces them into
its cudagraph capture cleanly:

- **`project_oporp_1bit_query`** ([`quantize.py`](../../retrieve/src/retrieve/layers/utils/quantize.py)).
  Per-query OPORP projection: `multiply → index_select → sign-pack`
  (~5 small kernels in eager). Under the outer cudagraph_trees the
  launch tax collapses — measured ~2.5× speedup at B=8 and B=64.
- **`OneBitKNN._score_full`** ([`one_bit_knn.py`](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py)).
  `xor → popcount → reduce` over the full corpus. The win here is
  *fusion*, not launch elision: eager materializes the `[B, N, W]` xor
  once and re-streams it through six SWAR popcount ops; Inductor fuses
  those into a single elementwise triton kernel. Measured ~3× at B=8,
  ~43× at B=64, N=50k — by far the largest compile win in the repo.
  ``d_total`` is derived from `item_bits.shape[1]` inside the body so it
  stays symbolic under `dynamic=True` (one graph across all `(B, N, W)`).

The matmul-bearing references (`PostfilterKNN`, `FullScanKNN`) were
tried with their own dedicated compile
wrappers and reverted — cuBLAS + CUB already win the heavy op, and the
cudagraph capture + mandatory output clone (to escape the
`reduce-overhead` buffer pool) cost more than they save.

## Numerics

V1's pure-torch dense path uses cuBLAS for the matmul, so the torch and
Triton-backend classes go through identical kernels and produce
bit-identical scores. V2's sparse path scores per-cell with
`tl.sum(emb_rows * q[None, :], axis=1)` — elementwise multiply +
reduction, not `tl.dot` — so it doesn't share the tensor-core
tile-reduction order quirks. Parity tests still use
[`assert_topk_matches`](../../retrieve/tests/parity/conftest.py)
(set + sorted-score tolerance) for V2 because gather order across
duplicate scores can vary.

**OPORP popcount is bit-exact.** V3's torch reference and the Triton
kernel both use the same SWAR popcount on the same packed bits, so the
parity test in
[`test_oporp_1bit_match_topk.py`](../../retrieve/tests/parity/test_oporp_1bit_match_topk.py)
asserts strict equality on returned ids and scores. If they ever
diverge, a popcount or packing bug has been introduced — the kernel's
correctness depends on bit identity here.

## SilverTorch kernels

Two co-designed IVF probe + INT8 scoring kernels live in
[`silvertorch/`](../../retrieve/src/retrieve/kernels/silvertorch/),
selected by `SilverTorch.filter`: `codesigned_probe_score` for
`filter ∈ {"none", "bloom"}`, `codesigned_probe_score_exact` for
`filter="exact"`. Both power
[`SilverTorch.forward`](../../retrieve/src/retrieve/layers/silvertorch/main.py).
(`bloom_match` also lives in this tree but is documented above as a
standalone filter primitive — it has a non-SilverTorch consumer now.)

Phase 1 (centroid `q @ centroids^T + topk` to pick the top-`n_probe`
clusters) runs **host-side** in `SilverTorch.forward`; both kernels are
"phase-2+3 fused" — for each `(query, probed-item)` cell each kernel
runs its predicate inline and (when it passes) scores via INT8
dequantize + dot. Items with `id == -1` (cluster padding) and items
failing the predicate get score `-inf`.

### `codesigned_probe_score` — IVF + INT8 + Bloom

[`silvertorch/codesigned_probe_score.py`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py).

The bloom intermediate (`[B, P, W]` sigs / bool match) and the
`int8 → fp32` code cast (`[B, P, D]`) never touch HBM — they live in
registers/SRAM. Bloom subset uses an OR-reduce trick:
`(qb & sig) == qb ⇔ qb & ~sig == 0` per word, OR-reduce over `W`,
saves the int32 cast + min reduction the equality form required.

The bloom inputs (`query_bits`, `bloom_sigs`) inherit the
`(clause_idx, value)` keying invariant from
[`_build_signatures`](../../retrieve/src/retrieve/layers/filters/bloom.py)
documented under `bloom_match` — same shared host-side path. Kernel is
unchanged.

The launch grid is `(cdiv(P, BLOCK_P), B)` — tile axis on **grid_x**
(≤ 2³¹) since `n_probe × max_cluster_size` can exceed the 65,535 limit
on grid_y/grid_z at large catalogs. **Tile config.**
`CodesignedProbeScoreConfig(block_p, num_warps, num_stages)` — shipped
as `DEFAULT_CONFIG` on the kernel module; pass `config=` to override.
Re-tune on a new arch via `uv run tune-kernels
codesigned-probe-score`. `P = n_probe * max_cluster_size` is fixed per
registered SilverTorch index, so no bucketing is needed; `HAS_QB`
remains a body-level constexpr (the bloom-on and bloom-off paths still
JIT-specialise on it). Score buffer is `torch.empty([B, P])` — every
in-bounds lane is overwritten (real dot or `-inf`), so no pre-fill
kernel is needed. The host then `torch.topk(scores, K)` directly and
gathers global ids. `SilverTorch.register_index` asserts
`K <= n_probe * max_cluster_size` so `P >= K` is structurally
guaranteed; the wrapper has no host-side pad tail (the prior
"`P < K` ⇒ pad to width K" path is gone — it was dead in production
and broke under `triton_op` SymInt tracing).

### `codesigned_probe_score_exact` — IVF + INT8 + exact AND-of-OR

[`silvertorch/codesigned_probe_score_exact.py`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py).

Same launch shape and IVF + INT8 scoring path as `codesigned_probe_score`;
swaps the bloom subset test for an exact AND-of-OR attribute predicate
fully unrolled over `(C, A_max)` against the `[N, C, A_max]` narrow item
attrs (`item_clause_attrs_narrow`) and `[B, C]` query attrs. Per program:
inner OR over `A_max` attribute slots per clause, outer AND over `C`
clauses, with `q_c == -1` overriding to "always passes" — the same inner
loop as the standalone `clause_mask` kernel, but fused into the score
path and applied only to the IVF-probed item subset. No false
positives (cf. bloom mode); bandwidth-cheaper per item at small
`C × A_max` because there's no `W`-word signature read. Trades the
bloom hash flexibility for exact-value match — schema-bound,
equality-only.

**Launch grid.** `(cdiv(P, BLOCK_P), B)` — identical to
`codesigned_probe_score`. `P = n_probe * max_cluster_size` is again
fixed per registered SilverTorch index, so no bucketing.

**Tile config.** `CodesignedProbeScoreExactConfig(block_p, num_warps,
num_stages=3)` — shipped as `DEFAULT_CONFIG` on the kernel module;
default `block_p=256, num_warps=4` mirrors the `codesigned_probe_score`
A100 tuning. Pass `config=` to override. Re-tune (it mirrors the
regular variant) via `uv run tune-kernels codesigned-probe-score`. Score
buffer is `torch.empty([B, P])` — same convention as
`codesigned_probe_score`, no pre-fill kernel.
