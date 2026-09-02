# `retrieve` kernels

Almost every kernel here is Triton. The one exception is a single CUDA
C++ translation unit under
[`silvertorch/cuda/`](../../retrieve/src/retrieve/kernels/silvertorch/cuda/),
which backs `SilverTorch(backend="cuda")` and is documented in
[its own section](#codesigned_probe_score_cuda--the-cuda-c-backend).

The Triton kernels split into three trees by domain:

- [`linr/`](../../retrieve/src/retrieve/kernels/linr/) — kernels
  used by `PrefilterKNN` / `OneBitKNN` / `SimHashKNN`
  (`fused_masked_knn_topk`,
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
  for the IVF + INT8 + exact AND-of-OR variant —
  `SilverTorch.filter_mode` picks one) plus `bloom_match` (lives in this
  tree for historical reasons but is now a cross-tree filter primitive
  consumed by `BloomFilter`). The `cuda/` subdirectory holds the CUDA
  C++ alternative to `codesigned_probe_score`, selected by
  `SilverTorch(backend="cuda")`.

Cross-tree `@triton.jit` building blocks live in
[`kernels/common.py`](../../retrieve/src/retrieve/kernels/common.py) —
see [Shared kernel helpers](#shared-kernel-helpers-kernelscommonpy).

The LinR kernels are the focus of this doc; the filter primitives
(`clause_compact`, `clause_mask`, `bloom_match`, `bloom_compact`) are
covered next, and the SilverTorch-only kernels at the end for context.

All **Triton** kernels follow the same conventions. The CUDA backend
deliberately departs from several of them (two launches, no
`num_stages`, `@torch.library.custom_op` instead of `@triton_op`); its
section spells out each divergence.

- One launch per `forward()` call. No persistent threads, no streams.
- Inputs are CUDA tensors with explicit strides; the host wrapper passes
  `.stride(i)` for every axis instead of assuming contiguity.
- Top-K selection is **not** in-kernel. Each kernel writes a `[B, ·]` score
  buffer and the host calls `torch.topk` on it. CUB's top-K (under torch)
  is faster than anything we can implement in pure Triton without a
  warp-level radix-select primitive.
- Tile config (`block_n`, `num_warps`, `num_stages`; `block_p` for the
  silvertorch kernels — the CUDA backend's config is `(block_p,
  num_warps)` only, since `num_stages` is a Triton pipelining concept)
  is offline-tuned per kernel and shipped as a
  single `DEFAULT_CONFIG` constant on the kernel module. No runtime
  `@triton.autotune`. Callers who want a non-default tile pass
  `config=<Kernel>Config(...)` to the private
  `_<name>_impl(..., config=)` companion — the public op has a fixed
  schema and always uses `DEFAULT_CONFIG`. The `tune-kernels` CLI
  (shipped with the library at `retrieve.tune:main`) sweeps the
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
- Graph-break behavior. Every host wrapper in this tree is decorated
  with `@torch.library.triton_op` + a textually-inline
  `wrap_triton(_kernel)[grid](**launch.kwargs)` launch — all ten
  Triton-registered ops across the seven Triton kernel files:
  `clause_mask`, `clause_compact`, `bloom_compact`,
  `fused_masked_knn_topk`, `bloom_match`, `codesigned_probe_score`,
  `codesigned_probe_score_bloom`, `codesigned_probe_score_exact`,
  `oporp_1bit_match_topk_full`, `oporp_1bit_match_topk_indirect`. The
  CUDA backend adds three more ops on `@torch.library.custom_op`
  (`codesigned_probe_score_cuda`, `codesigned_probe_score_bloom_cuda`,
  `codesigned_probe_score_exact_cuda`) — 13 ops across 8 kernel files in
  total. `@triton_op` is for Triton
  kernels specifically; a C++ extension has no `@triton.jit` body for
  inductor to see, so an opaque custom op is the correct registration
  there. The decorator stops dynamo from
  graph-breaking at the wrapper boundary, so each layer's full forward
  captures into one cudagraph_trees graph, and lets inductor see the
  underlying `@triton.jit` kernel (preserves the reference under
  `torch.export` — gated by
  [`tests/compile/test_export_kernel_ref.py`](../../retrieve/tests/compile/test_export_kernel_ref.py);
  opens epilogue-fusion headroom for Stage 3). The `wrap_triton` call
  must stay textually inside each decorated body — dedup targets the
  code around it, never the launch line itself.

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
4. A private `_<name>_impl(..., *, config: ...Config | None = None)`
   lives next to the public op(s) in every file. Both share the same
   `_<name>_prep` / `_<name>_finish` helpers, so their bodies differ
   only in the launch line (`_kernel[grid](**kwargs)` eager vs
   `wrap_triton(_kernel)[grid](**kwargs)` in the `@triton_op` body —
   the launch must appear textually in the decorated function's source,
   which is what `torch.export`'s kernel registry walks to preserve the
   kernel reference) and, where policies intentionally diverge, in
   explicit prep/finish flags (e.g. `fused_masked_knn_topk`'s
   `bucket=` / `pad_to_k=`). The op schema doesn't carry the config
   dataclass; tests and the tuner reach `_impl` directly to pass an
   override.
5. Tuning is offline: `retrieve/src/retrieve/tune.py` (`uv run
   tune-kernels <kernel-subcommand>`) is a declarative registry — one
   `KernelTuneSpec` per kernel in the `KERNELS` tuple, from which the
   eight click subcommands are generated
   (`fused-masked-knn-topk`, `oporp-1bit-match-topk`,
   `codesigned-probe-score`, `codesigned-probe-score-cuda`,
   `codesigned-probe-score-exact`,
   `clause-mask`, `clause-compact`, `bloom-compact`), all driven by one
   generic `_sweep` + `_print` pair. Each spec sweeps its `(block,
   num_warps)` grid against built-in shape regimes mirroring real-eval
   workloads (goodreads N≈800k, arxiv N≈3M, synth-15M; batch sizes from
   the eval configs). The regime-swept kernels
   (`codesigned-probe-score-exact` and the three filter kernels) accept
   repeatable `--regime N,B,C,A_MAX` (or `N,B,W`) flags; the
   dimension-swept kernels expose `--d`/`--b`/`--w` flags and derive
   regimes by crossing them with the module's bucket/P axes. Picks one
   default per arch via plurality vote across regime winners (ties →
   lower `num_warps`); emits a pasteable `DEFAULT_CONFIG = ...` line
   (optional `--json-out` dumps the full per-regime sweep, keyed
   `per_regime`). Re-run once per new arch; commit the line.
   (`bloom_match` has no subcommand — see its section.) A CUDA-gated
   smoke test
   ([`test_tune_smoke.py`](../../retrieve/tests/correctness/test_tune_smoke.py))
   runs one tiny regime per spec so schema drift between tune.py and
   the kernel `_impl`s breaks CI instead of a tuning session.

For the compact kernels, the offline tuner avoids the `atomic_add`
hazard naturally: it calls the host wrapper, which allocates fresh
`out_indices` (`-1`-filled) and `counts` (zeros) on every call — `do_bench`
reps each pay one allocation, so no cross-rep accumulation. The
warning that lived on the kernel files about
`@triton.autotune`-time corruption still applies to in-kernel
autotune; the offline path is unaffected.

## Shared kernel helpers (`kernels/common.py`)

[`kernels/common.py`](../../retrieve/src/retrieve/kernels/common.py)
holds the `@triton.jit` building blocks the kernel bodies call (Triton
inlines them). Every helper is a pure function of already-loaded tiles
or takes fully-resolved addressing from the caller — no helper decides
its own tile shape, launch grid, or masking policy:

- `popcount_int64(x) → int32` — SWAR popcount over int64 lanes; used by
  `oporp_1bit_match_topk`. Its torch twin is
  [`layers/utils/quantize.py::popcount_int64`](../../retrieve/src/retrieve/layers/utils/quantize.py) —
  the bit-exact pairing is load-bearing (see [Numerics](#numerics)).
- `bloom_subset_pass(qb, sigs) → [BLOCK] int1` — the bloom subset test
  in the `qb & ~sig == 0` OR-reduce form (boolean-identical to, and
  cheaper than, the equality + min-reduce form; all three bloom
  consumers — `bloom_match`, `bloom_compact`,
  `codesigned_probe_score` — now standardize on it). Callers still AND
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
- `compact_store(pass_mask, ids, counts_ptr, out_ptr, bid, ...)` — the
  stream-compaction epilogue (`cumsum` intra-tile offsets +
  `atomic_add` row base + masked store), shared by `clause_compact` and
  `bloom_compact`.
- `or_combine(a, b)` — combine_fn for `tl.reduce` OR-reductions.

Perf caveat: helpers with many pointer params (`clause_pass`) can
perturb register allocation. The documented fallback if a kernel
regresses > 5% on the tune-kernels gate is to revert *that kernel body*
to the inlined predicate with a
`# keep in sync with kernels/common.py::clause_pass` breadcrumb.

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
`candidate_pool`), and `p >= k > 0` is guaranteed by `PrefilterKNN` —
the policy rationale is written down once in the module docstring.
When bucketed, the score buffer is allocated at `[B, P_BUCKET]`; lanes
in `[P_real, P_BUCKET)` get `-inf` automatically because
`count[bid] <= P_real`, the caller-supplied `positive_indices` stays at
width `P_real`, and the post-topk `gather` clamps `topk_local` to
`P_real - 1` before indexing (the `where(isfinite, …, -1)` mask then
overwrites those slots with the `-1` sentinel; on the unbucketed public
op the clamp is an identity).

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

**Popcount** is the SWAR bit-twiddle
[`common.popcount_int64`](../../retrieve/src/retrieve/kernels/common.py):
five mask-shift-add steps, no libdevice dependency.
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
return:   positive_indices    [B, N]         int64  (full width, -1 tails)
          counts              [B]            int64
```

The returned index buffer is **full-width** `[B, N]` — only the first
`counts[b]` entries per row are meaningful; the tail keeps the `-1`
prefill (no host-side `counts.max().item()` sync, no narrow slice).
Downstream kernels bound reads by `counts`.

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

**Clause loop** is the shared
[`common.clause_pass`](../../retrieve/src/retrieve/kernels/common.py)
helper, fully unrolled (`C` and `A_MAX` are `tl.constexpr`): inner OR
over the `A_MAX` attribute slots per clause, outer AND over the `C`
clauses, with the reverse flag XORed in per-clause and `q_c == -1`
overriding to "always passes." The compaction epilogue is
`common.compact_store`.

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

**Inner loop** is the same shared `common.clause_pass` helper as
`clause_compact`'s: inner OR over `A_MAX` slots, outer AND over `C`
clauses, reverse XOR, inactive override. The epilogue is a single
`tl.store` of the `pass_mask` tile — no cumsum, no atomics.

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
[`build_signatures`](../../retrieve/src/retrieve/layers/filters/bloom_hash.py)
(`layers/filters/bloom_hash.py`, the single home of the bloom hash
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

**Hash invariant.** `_signature_batch` (the shared core in
`bloom_hash.py`) keys each hash on `(clause_idx, value)` (paper §4.1:
"for each feature" — a *feature* is a `(key, value)` pair). It does
this by XOR-ing a per-clause salt — `_mix64(clause_id, …)` — into the
post-`_mix64` hash, before the position mask. This is what prevents
value `V` in clause C0 from colliding with the same `V` in clause C3
when clauses share a value vocabulary; without it, single-clause
queries on overlapping vocabularies leak ~25–30% of non-matching items
as false positives. The kernel itself is untouched: it consumes
`[N, W]` / `[B, W]` int64 buffers as opaque bits. The salt costs one
extra elementwise XOR per chunk (well under measurement noise vs the
existing scatter + word-pack reduction); kernel HBM traffic and launch
shape are unchanged. The hash math is pinned: persisted `bloom_sigs`
buffers must stay bit-valid across refactors, so any change to
`bloom_hash.py` invalidates every stored index (buffers persisted
before the `(clause_idx, value)` keying are stale and must be rebuilt).

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

**Inner op**: the shared
[`common.bloom_subset_pass`](../../retrieve/src/retrieve/kernels/common.py)
helper — `qb & ~sig` per word, OR-reduced over `W`, `== 0` at the end
(`(qb & sig) == qb ⇔ qb & ~sig == 0`). This is the cheaper algebraic
form (saves the int32 cast + min reduction the old equality +
min-reduce form required) and is boolean-identical; all three bloom
consumers standardize on it.

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
subset-test helper and adds `clause_compact`'s cumsum + `atomic_add`
tail, but uses the 3D launch grid the compact kernels need at large N.

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
return:    positive_indices [B, N] int64  (full width, -1 tails)
           counts           [B]    int64
```

Same full-width `[B, N]` return contract as `clause_compact` — bound
reads by `counts[b]`, tails keep the `-1` prefill.

**Launch grid** `(B, tiles_y, tiles_x)`, same 3D shape as the clause
compact/mask kernels (`tile_id = tile_x * tiles_y + tile_y`). Each
program loads `qb[b, :]` once, the `[BLOCK_N, W]` `sigs` tile, computes
the subset test via the shared `common.bloom_subset_pass` (same helper
as `bloom_match`), then runs the shared `common.compact_store`
epilogue (same as `clause_compact`). Wide `W` (=16 at the default `m_bits=1024`) makes
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
| [`PostfilterKNN`](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py) | always dense | none — pure torch `(q @ x.T)` + `masked_topk` |
| [`PostfilterKNNInt8`](../../retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py) | always dense | none — pure torch `torch._int_mm(...)` + `masked_topk` (int8×int8 → int32, IMMA) |
| [`PrefilterKNN`](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py)         | candidates   | `fused_masked_knn_topk` over caller-precompacted `(candidate_ids, counts)` |
|                                                                | unmasked     | none — pure torch dense path (nothing to pre-filter) |
| [`OneBitKNN`](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py) / [`SimHashKNN`](../../retrieve/src/retrieve/layers/linr/simhash_knn.py) | full         | `oporp_1bit_match_topk_full` (HAS_INDICES=False)       |
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

### [`compact_mask`](../../retrieve/src/retrieve/layers/utils/compact.py)

Bool `[B, N]` → `(positive_indices[B, N], counts[B])` — **full width**,
same contract as the triton `clause_compact`/`bloom_compact` kernels so
torch and triton paths stay interchangeable. Implementation:
`mask.sum(1)` for counts, then
`mask.float().argsort(descending=True, stable=True)` for the indices —
no `.item()` sync, no narrow slice. The tails differ across
implementations (arbitrary argsort-tail ids here vs `-1`-prefilled for
the kernels), so consumers must bound reads by `counts` either way.

### [`popcount_int64`](../../retrieve/src/retrieve/layers/utils/quantize.py)

Torch-side SWAR popcount. Required because this PyTorch (2.10.0+cu128)
lacks `Tensor.bitwise_count`. Returns int32 to keep the downstream sum
narrow. Matches the kernel-side `common.popcount_int64`
step-for-step — the bit-exact pairing the OPORP/SimHash parity tests
depend on (the SWAR constants exist in exactly these two places).

### [`quantize_oporp_1bit`](../../retrieve/src/retrieve/layers/utils/quantize.py)

Build-time only. `O(D)` parameter cost — the `signs` vector and `perm`
permutation are the entire projection. Apply with one elementwise multiply
and one `index_select`, no matmul needed. The `_pack_signs_to_int64`
helper packs the sign-quantized output into `[..., W]` int64 words using
`<<` and `sum(-1)`; same packing used both at index time and at query time.

### Compile on the V3 torch reference

The pure-torch hot bodies on V3's reference path —
`project_oporp_1bit_query` and the bit-KNN base's
`_score_full_bits_eager`
([`_bit_knn.py`](../../retrieve/src/retrieve/layers/linr/_bit_knn.py)) —
are pure tensor-flow free of `.item()` and Python control flow, so the
outer `torch.compile(dynamic=True, mode="reduce-overhead")` wrapped
around each algo's forward in `evaluation/retrieval/algos/` traces them
into its cudagraph capture cleanly:

- **`project_oporp_1bit_query`** ([`quantize.py`](../../retrieve/src/retrieve/layers/utils/quantize.py)).
  Per-query OPORP projection: `multiply → index_select → sign-pack`
  (~5 small kernels in eager). Under the outer cudagraph_trees the
  launch tax collapses — measured ~2.5× speedup at B=8 and B=64.
- **`_score_full_bits_eager`** ([`_bit_knn.py`](../../retrieve/src/retrieve/layers/linr/_bit_knn.py)).
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
selected by `SilverTorch.filter_mode`: `codesigned_probe_score` for
`filter_mode ∈ {"none", "bloom"}`, `codesigned_probe_score_exact` for
`filter_mode="exact"`. Both power
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
registers/SRAM. Bloom subset is the shared
`common.bloom_subset_pass` OR-reduce form:
`(qb & sig) == qb ⇔ qb & ~sig == 0` per word, OR-reduce over `W`,
saves the int32 cast + min reduction the equality form required.

The bloom inputs (`query_bits`, `bloom_sigs`) inherit the
`(clause_idx, value)` keying invariant from
[`bloom_hash.build_signatures`](../../retrieve/src/retrieve/layers/filters/bloom_hash.py)
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
attrs (SilverTorch's `item_clause_attrs` buffer) and `[B, C]` query
attrs. Per program: inner OR over `A_max` attribute slots per clause,
outer AND over `C` clauses, reverse-XOR per clause, with `q_c == -1`
overriding to "always passes" — the same `common.clause_pass` helper
as the standalone `clause_mask` kernel, but fused into the score
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
A100 tuning. Pass `config=` to override. Re-tune via its own
subcommand, `uv run tune-kernels codesigned-probe-score-exact` —
its regime axes are `(N, B, C, A_MAX)` clause shapes (repeatable
`--regime`), not the bloom-word `W` axis of the regular variant; the
optimum also tracks `P = n_probe × max_cluster_size`, so re-tune with
`--regime` per deployment rather than trusting the defaults. The
predicate body is the shared `common.clause_pass`
(`ids=safe_ids`, `load_mask=valid` — indirect addressing over the
probed items). Score buffer is `torch.empty([B, P])` — same convention
as `codesigned_probe_score`, no pre-fill kernel.


### `codesigned_probe_score_cuda` — the CUDA C++ backend

[`silvertorch/codesigned_probe_score_cuda.py`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_cuda.py)
(host side) +
[`silvertorch/cuda/codesigned_probe_score.cu`](../../retrieve/src/retrieve/kernels/silvertorch/cuda/codesigned_probe_score.cu)
(device side), selected by `SilverTorch(backend="cuda")`. All three
`filter_mode`s run here.

This is the paper-faithful implementation of Algorithm 1
(`partial_bloom`) phases 2+3. The Triton kernel and this one compute the
same function by different means, and the difference is the point of the
backend existing: the Triton kernel fuses a **row-wise** bloom signature
read into the scoring pass; the paper instead evaluates bloom against a
**transposed** bit matrix and hands the scorer a 1-bit-per-item mask.
That needs two kernels, so it cannot be expressed as one fused Triton
launch.

The split generalizes past bloom. *Any* filter here is a
1-bit-per-item mask laid out cluster-major over the probed spans (the
paper's `M_c`), and phase 3 is filter-agnostic — it tests one bit and
never learns what produced it. So `filter_mode="exact"` is a second
phase-2 kernel rather than a third scoring kernel: `cps_bloom_mask_kernel`
and `cps_clause_mask_kernel` write the same layout, and
`cps_score_kernel<…, HAS_MASK=true>` consumes either unchanged.

#### Phase 2 — the transposed bloom index

The bloom index is stored rotated (paper Fig. 3(b)). `sigs_t[m]` is a
bit-vector over *items*, 64 items per int64 word, with items laid out
cluster-major in padded IVF order so that every cluster occupies a
contiguous, word-aligned span of `wpc = ceil(max_cluster_size / 64)`
words. `build_transposed_sigs(bloom_sigs, padded_cluster_items)` builds
it host-side once at `register_index`; bit `s % 64` of word
`c * wpc + s // 64` in row `m` is bit `m` of
`bloom_sigs[padded_cluster_items[c, s]]`, with `-1` padding slots
contributing `0`.

`cps_bloom_mask_kernel` then iterates **only the set bits** of the query
signature (`__ffsll` / `bits &= bits - 1`) and ANDs the matching rows:
one `and.b64` decides 64 items. Filter traffic is therefore proportional
to `popcount(QB)` — typically `k_hash × active clauses`, tens of bits —
rather than to `m_bits`. At `W=16` (`m_bits=1024`) the row-wise form
reads 128 B per probed item where this reads `~popcount(QB)/64 × 8` B.
That gap is the paper's headline bloom win, and it is why the
transposition is worth a second launch.

The accumulator starts at `~0LL`, the AND identity: a query with no set
bits passes everything, which is exactly the row-wise `(qb & ~sig) == 0`
semantics. One thread produces one output mask word; consecutive threads
walk consecutive words of the same cluster span, so the row reads
coalesce. Output is `[B, n_probe · wpc]` int64.

One asymmetry with the clause kernel below, deliberate but easy to trip
over: this kernel does **not** force the pad tail of a span's last word to
0. `build_transposed_sigs` zero-fills those columns, so any set query bit
clears them — but an empty `QB` short-circuits to the `~0` identity and
leaves the tail at 1. Harmless, because phase 3 only ever addresses bit
`slot` for `slot < max_cluster_size`. Nothing should start depending on a
zero tail from *this* kernel; the clause kernel's zero tail is a real
guarantee and is tested.

**Index memory.** `bloom_sigs_t` is `[m_bits, n_lists · wpc]` int64:

```
bytes(bloom_sigs_t) = W · 8 · n_lists · wpc · 64
                    = bytes(bloom_sigs) × (n_lists · wpc · 64 / N)
```

The factor is the IVF *padding slack* — `n_lists · max_cluster_size / N`,
rounded up to whole 64-slot words — because every cluster is padded out to
the largest one. On a balanced index it is 1.0–1.1× and, since the cuda
backend registers `bloom_sigs_t` *instead of* `bloom_sigs` (not alongside),
total filter memory is roughly unchanged versus triton. It grows without
bound as clusters get uneven, so `_register_filter_buffers` emits a
`RuntimeWarning` above 2× pointing at re-clustering. Record the ratio per
dataset — the runbook's §3.4 / §7 memory-delta gate expects it.

#### Phase 2 (exact) — the clause mask

`cps_clause_mask_kernel` fills the *same* `[B, n_probe · wpc]` layout
from the exact AND-of-OR clause predicate, so exact mode reuses phase 3
untouched.

One **thread** owns one slot and one **warp** owns one 32-bit half of an
output word: lane `l` evaluates slot `l` of the half, one `__ballot_sync`
packs the 32 verdicts, and lane 0 stores the half through a 32-bit view
of the int64 word (little-endian: low half at `2·word`, high half at
`2·word + 1`). Every lane must reach the ballot, so the early return for
`word_idx >= mask_words` is warp-uniform and the per-lane work sits in a
branch *before* the ballot, never around it.

Ids come from **`flat_items`** — the same `[B, P]` tensor phase 3 reads —
not from `probe_ids` plus the padded index. Two consequences, both
deliberate:

- The 32 lanes of a half read 32 consecutive `int64` ids: one coalesced
  256-byte run, the same access phase 3 makes.
- Exact mode registers **no new buffer**. Unlike bloom, where cuda swaps
  `bloom_sigs` for `bloom_sigs_t`, a cuda+exact module's `state_dict` is
  identical to a triton+exact one's — `item_clause_attrs[N, C, A_max]`
  int64 and `clause_is_reverse[C]` bool, read in place. The op takes
  `max_size` (the padded cluster width, `P // n_probe`) as a Python int
  so it can still lay the mask out cluster-major.

The predicate is bit-identical to `common.clause_pass`: seed `keep` from
`id >= 0`, OR over the `A_max` values of each clause, XOR the clause's
reverse flag, then let `q_c == -1` (inactive) override — in that order,
since the sentinel outranks reverse. Padding slots (`id < 0`) and the pad
tail of a span's last word get bit `0`.

`C` and `A_max` are **template parameters** on a small dispatch table
(`C ≤ 4`, `A_max ∈ {1, 2, 4}`, `C · A_max ≤ 8` — ten instantiations);
any other shape takes the `<0, 0>` instantiation, which is the original
runtime-bound loop. The fast path loads the query row and reverse flags
first (independent of the id), then the id, then **all** `C · A_max`
attribute words of the row into registers with `#pragma unroll`, and only
then evaluates the predicate. This matters because the per-slot chain is
`id → attrs → ballot`: with runtime loop bounds each attribute word waited
on the previous compare, and the first version of the kernel (one warp
per word, two sequential halves) measured 101 µs at `B=16, P=58 k` on
A100 for ~40 bytes of traffic per slot. The register version measures
56 µs, and the same kernel over *sequential* ids runs in ~20 µs — the
remaining ~35 µs is the random 32-byte gather over an `item_clause_attrs`
table larger than L2, which the Triton exact kernel also pays (it is the
~37 µs gap between its no-filter and exact variants). `clause_is_reverse`
is read as `torch.bool` storage through an `unsigned char*` because
`__ldg` has no `bool` overload; `q_c` and the reverse flag are
warp-uniform loads that broadcast.

#### Phase 3 — masked `__dp4a` scoring

`cps_score_kernel<SEG, HAS_MASK, UNROLL>` scores one
`(query, probed slot)` pair per segment: gather the item id (`-1` =
cluster padding), test one mask bit, and only for passing items stream
the int8 code row straight from the embedding table into a `__dp4a`
dot — 4 MACs per instruction, the instruction the paper names. There is
no intermediate gather tensor and no shared memory anywhere.

Each lane owns one **16-byte chunk** of a row (one `int4` load, four
packed code words) and the four matching query words sit in registers
for the whole block, so `SEG = D / 16` lanes cover a row and
`SPW = 32 / SEG` items advance per warp-instruction:

| `D` | `SEG` | items per warp-instruction | note |
|---|---|---|---|
| 64 | 4 | 8 | |
| 128 | 8 | 4 | one row = exactly one 128 B cache line |
| 256 | 16 | 2 | |
| other, `D % 4 == 0` | 32 | 1 | `cps_score_kernel_generic` fallback (4-byte words, runtime loop) |

The launcher also routes a table `D` to the generic kernel when the code
or query base pointer is not 16-byte aligned (a storage-offset view; the
caching allocator's own tensors always are).

The first version of this kernel mapped a full 32-lane warp to one
`D=128` row with 4-byte loads. On A100 that left one 128 B row in flight
per warp and ran at ~1.0 TB/s (188 µs at `B=16, P=58 k`, no filter)
against Triton's ~1.5 TB/s; with the `int4` layout a warp has
`SPW · UNROLL` rows outstanding and the same regime measures 88 µs vs
Triton's 85 µs. Code rows are read with `__ldcs` (evict-first): each row
is touched once per query, so caching it only displaces the id / mask
stream.

Warps own contiguous item tiles (the paper's "one warp per contiguous
tile of items"); segments interleave inside the tile so the `SPW` ids and
score stores of one instruction stay adjacent. Ids are software-pipelined
one iteration ahead — the `id -> row` dependency is the loop's only serial
latency chain, so the next iteration's ids are requested before this
iteration's row gathers. The keep verdict is **segment-uniform**, so
filtered items genuinely skip both the row load and the whole dp4a
chain — that is where the filter's savings are realized. Non-passing
slots store `-INFINITY`; every slot in `[0, P)` is written, so the score
buffer is `torch.empty`, matching the Triton convention. Top-K stays
host-side (phase 4).

`seg_reduce_add<SEG>` uses `__reduce_add_sync` on `sm_80+` and a
`__shfl_xor_sync` butterfly below. Both are exact integer sums, so the
arch split never changes results.

The mask test is **division-free**: a segment tracks the `(cluster, slot)`
of its base item incrementally — one 64-bit divide before the loop, then
`slot += SPW·UNROLL` with a carry loop into `cluster` — instead of
recomputing `p / max_size` per item. The carry loop is bounded by
`SPW·UNROLL ≤ 32` subtractions and stays correct when `max_size` is
smaller than one step, i.e. when a single iteration crosses several
cluster spans.

#### Phase 3 — the `UNROLL` knob

`UNROLL ∈ {1, 2, 4}` (config field) is how many items one segment keeps
in flight per iteration. The loop body runs in three phases — ids and
keep verdicts, then **all** `UNROLL` row gathers, then the dots — so
several independent row loads are issued before the first `__dp4a`
stalls on one. Rejected items are still skipped: the row loads are
predicated on the segment-uniform keep flag rather than sitting behind
serialized `if` blocks.

Measured on A100 (2026-09-02, `D=128`): with the `int4` layout `UNROLL=1`
already keeps 4 rows per warp in flight and wins or ties at every regime
except bloom mode at `P ≈ 58 k`, where `256/4/4` gains ~10 % over the
shipped `128/8/1` at the cost of ~2 µs at small `P`. Registers: 30 at
`UNROLL=1`, 40 at `UNROLL=4` (`cuobjdump -res-usage`), so occupancy is not
a concern at any value. The generic runtime-`D` fallback has **no**
`UNROLL` parameter and the launcher ignores the config's value there.

The arithmetic is identical at every `UNROLL`: same dp4a order over the
lane's four words, same segment reduction, same two left-associated fp32
multiplies. So bit-exactness vs Triton is a property of the kernel, not
of the config, and the parity suite asserts it with `torch.equal` across
configs.

#### Numerics: bit-identical to Triton, and how that is kept

The int32 dot is exact (`|dot| ≤ D_max · 128² = 2²² ≪ 2³¹`, and integer
addition is associative in the no-overflow regime, so any accumulation
order matches `tl.dot` bit-for-bit). The epilogue is two left-associated
fp32 multiplies matching Triton's
`dots.to(f32) * q_scale * global_scale`. Both masks are
boolean-identical to their Triton predicates — the transposed AND to the
row-wise subset test, the ballot-packed clause bits to `clause_pass`.
Therefore the `[B, P]` **score tensors are bit-identical** across the two
backends, and
[`tests/parity/test_codesigned_probe_score_cuda.py`](../../retrieve/tests/parity/test_codesigned_probe_score_cuda.py)
asserts that with `torch.equal` rather than an approximate match.

The returned **ids are gated one notch weaker: identical up to permutation
within tied scores** (`assert_ids_equal_up_to_ties` in
`tests/parity/conftest.py`). Both backends run the same host
`torch.topk` + `gather` on the same tensor, but `torch.topk` documents its
tie order as "not guaranteed stable across invocations", and ties are not
hypothetical here — int32 dots of ~±1e5 magnitude over P≈768 candidates
land on roughly one tied pair per row. A tie permutation is a top-K
property, not a kernel bug; anything outside a tie run still fails.

Two invariants protect it:

- **The build must not pass `--use_fast_math`.** Compiled flags live in
  `_load_ext` (`extra_cuda_cflags=["-O3", "-lineinfo"]`). Fast-math would
  license reassociation of the fp32 epilogue and silently break parity.
- No FMA contraction is possible on a pure product chain, so the
  left-associated multiply order is stable as written.

#### Constraints the kernels enforce

`TORCH_CHECK` failures, not silent degradation:

| constraint | why |
|---|---|
| `D % 4 == 0` | `__dp4a` consumes 4 int8 lanes per word |
| `unroll ∈ {1, 2, 4}` | it is a kernel *template* parameter, so only the instantiated values dispatch; the launcher rejects anything else instead of silently rounding |
| `B ≤ 65535` | batch rides `grid.y`. The Triton kernels dodge this by putting the *tile* axis on `grid.x`; this backend reintroduces the limit **on batch size** |
| `block_p % num_warps == 0` | each warp owns a contiguous `block_p / num_warps` tile |
| `P % max_size == 0` | slot → `(cluster, offset)` arithmetic in the mask test — and on the exact path `max_size` is an *argument*, not derived from a `probe_ids` width, so this is the only thing tying the two apart |
| `query_clause_attrs.size(1) == item_clause_attrs.size(1)` and `clause_is_reverse.size(0) == C` | `C` is a runtime kernel argument, not a `constexpr`; a mismatch would read past the attr row |
| `clause_is_reverse` is `torch.bool` | the kernel reads its one-byte storage directly (`__ldg` has no `bool` overload); the Triton sibling instead converts to int8, so the two wrappers differ here |

Two things that are *not* in that table, because they are not checks:

- **`D ∈ {64, 128, 256}` is dispatch, not a constraint.** Any other
  `D % 4 == 0` silently takes `cps_score_kernel_generic`, which is correct
  but slower (runtime word loop, query words re-read per item). Only
  `D % 4 == 0` itself is a `TORCH_CHECK`.
- **Ids are trusted.** `probe_ids` entries and `flat_items` ids index the
  signature / code / attribute tables with no bounds test on the device.
  They come from the layer's own buffers (`padded_cluster_items` gathered
  by a `topk` over `n_lists` centroids), so they are in range by
  construction, and a per-item check would put a branch in the hot loop.
  A caller who hand-builds them out of range gets an out-of-bounds read,
  not an exception.

#### Build, ops, and tuning

The extension JIT-compiles on first *use* via
`torch.utils.cpp_extension.load`, cached in `TORCH_EXTENSIONS_DIR`.
Importing the module never triggers a build, so CPU-only machines can
import `retrieve` and collect tests freely; `is_available()` is the
sanctioned capability probe and `ensure_built()` forces the compile.
Building needs a system CUDA toolkit whose major version matches the
torch wheel (cu128 → 12.x `nvcc`), `ninja` (in retrieve's dev dependency
group), and `CUDA_HOME` if `nvcc` is off PATH.

The build outcome — the module *or* the exception — is memoized once in a
module global (not `functools.cache`, whose key would include `verbose`),
and the two ways it can be absent are kept apart. `ToolchainMissing` (no
CUDA device, or no `nvcc` on PATH / under `$CUDA_HOME`) is "you cannot run
this here"; anything else is a build that was attempted and broke, and its
`ImportError` carries the nvcc/ninja output. `tests/conftest`'s
`require_cps_cuda()` skips on the first and **fails** on the second —
silently skipping a compile error is precisely how a broken first GPU run
would read as green. The JIT path in `cpp_extension.load` never runs
torch's own `_check_cuda_version`, so the wrapper probes `nvcc --version`
itself and refuses a major-version mismatch with both versions named.

The ops are **not** tagged `torch._C.Tag.cudagraph_unsafe`, and should not
be. The one thing that could make a first call unsafe to capture is the
~1-minute JIT build, and that is always absorbed by an eager
`ensure_built()` / `require_cps_cuda()` / warmup before any capture
window. Everything else the tag would protect against — non-default
streams, raw allocation, host syncs — this backend already avoids by
construction. The handoff's F3 lists the tag as a last-resort fallback if
capture fails anyway, with its perf cost recorded.

The `.cu` also `#error`s out below **sm_61**: `__dp4a` arrived with
Pascal in CUDA 8 (declared in `sm_61_intrinsics.h`; see NVIDIA's
["Mixed-Precision Programming with CUDA 8"](https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/)),
and the whole scoring design *is* that instruction, so an older
`TORCH_CUDA_ARCH_LIST` must fail at compile time rather than fall back to
something slower and unvalidated. The dispatch table costs
`3 (D) × 2 (mask) × 3 (unroll) = 18` instantiations of the scoring
kernel, so the one-time JIT build is correspondingly longer.

All three ops (`codesigned_probe_score_cuda`,
`codesigned_probe_score_bloom_cuda`,
`codesigned_probe_score_exact_cuda`) are
`@torch.library.custom_op(..., device_types="cuda")` with
`register_fake` — opaque to dynamo and export, cudagraph-safe by
construction (kernels launch on the current torch stream, all allocation
goes through `torch.empty`, and `global_scale` / `k` / `max_size` stay
Python scalars so nothing forces a host sync during capture; `max_size`
comes off `padded_cluster_items.shape[1]`, a static buffer shape). They
are three ops rather than one op with optional arguments, so the layer
just routes to the right op.

Config is `CodesignedProbeScoreCudaConfig(block_p, num_warps, unroll)`,
shared by both filtered paths; re-tune with `uv run tune-kernels
codesigned-probe-score-cuda` and `… codesigned-probe-score-exact-cuda`
(the grid is `{128, 256, 512, 1024} × {4, 8} × {1, 2, 4}`, 24 points).
Neither phase-2 mask kernel is swept (fixed 256-thread blocks) — their
traffic is noise next to phase 3. The two subcommands paste into the same
`DEFAULT_CONFIG` line, so reconcile them before pasting; unlike the bloom
pair below, `codesigned-probe-score-exact-cuda` *is* directly comparable
with the Triton `codesigned-probe-score-exact` (same regime axes, same
attribute distribution, same predicate).

> **Comparing tuner output across backends.** The `codesigned-probe-score`
> and `codesigned-probe-score-cuda` specs share regime axes
> `(P, HAS_QB, D, B, W)` so their `--json-out` files join on identical
> keys — but the `HAS_QB=1` rows are **not** comparable. The CUDA spec
> feeds a *sparse* query signature (~25 set bits, the workload the
> set-bit iteration is built for) against a transposed index, while the
> Triton spec feeds dense random row-wise signatures. Dense bits would be
> a pathological worst case for phase 2, so equal-input comparison needs
> the head-to-head methodology in
> [cuda-silvertorch-handoff.md](../plans/cuda-silvertorch-handoff.md).
> The CUDA spec also requires `P` to be a multiple of its synthetic
> `max_size`, so `--regime` values are not freely interchangeable between
> the two subcommands. And `_cps_cuda_probe_family` fills every probed slot
> with a real id — it carries **no `-1` cluster padding**, unlike a real
> index — so the scorer's `id < 0` early-out never fires during a sweep and
> the tuned config is chosen against the fully-populated worst case.

GPU validation and benchmark runbook:
[cuda-silvertorch-handoff.md](../plans/cuda-silvertorch-handoff.md).
