<!-- claude code generated file -->

# `retrieve` kernels

> Previously: `retrieve/docs/kernels.md` (originally `retrieve/docs/KERNELS.md`).

The Triton kernels split into three trees by domain: [`linr/`](../../retrieve/src/retrieve/kernels/triton/linr/)
holds the kernels used by LinR V1/V2/V3, [`filters/`](../../retrieve/src/retrieve/kernels/triton/filters/)
holds the standalone clause-evaluation kernel that powers
`ClauseIndex.evaluate_indices`, and [`silvertorch/`](../../retrieve/src/retrieve/kernels/triton/silvertorch/)
holds those used by SilverTorch. The LinR kernels are the focus of this doc;
`clause_compact` and the SilverTorch kernels are covered briefly at the
end for context.

All kernels follow the same conventions:

- One launch per `forward()` call. No persistent threads, no streams.
- Inputs are CUDA tensors with explicit strides; the host wrapper passes
  `.stride(i)` for every axis instead of assuming contiguity.
- Top-K selection is **not** in-kernel. Each kernel writes a `[B, ·]` score
  buffer and the host calls `torch.topk` on it. CUB's top-K (under torch)
  is faster than anything we can implement in pure Triton without a
  warp-level radix-select primitive.
- Autotune key is shape-only (`B`, `N`, `D`, `P`, `W`, `HAS_INDICES`,
  etc.). Block sizes and `num_warps` are searched; `num_stages=3` is fixed.

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

## `fused_matmul_topk` — V1 dense path

[`kernels/triton/linr/fused_matmul_topk.py`](../../retrieve/src/retrieve/kernels/triton/linr/fused_matmul_topk.py).

Computes `query @ item_embs.T`, optionally masks, and writes a `[B, N]`
score buffer. The host calls `torch.topk` on the buffer.

```
inputs:    query     [B, D]  fp32
           item_embs [N, D]  fp32
           mask      [B, N]  bool   (optional, gated by HAS_MASK)
output:    scores    [B, N]  fp32
```

**Launch grid** `(cdiv(B, BLOCK_M), cdiv(N, BLOCK_N))`. Each program owns
one `[BLOCK_M, BLOCK_N]` output tile. With `BLOCK_M ≥ 16` even `B=1`
queries land on tensor cores — the kernel masks the padded rows on store.

**Inner loop** is a single `tl.dot(q_tile, item_tile.T)` over the full `D`
axis (no K-loop because `D=128` fits in one tile). The matmul lands on
tensor cores. Tile-blocked reduction order means scores can drift ~1e-4
fp32 vs cuBLAS's row-major reduction — bench tests use a tolerant
top-K matcher; bit-exact parity is **not** asserted for this kernel.

**Mask path** (`HAS_MASK=True`): a single `tl.load` of `mask[m, n]` and
`tl.where(m, scores, -inf)` in the same program. No second pass.

**Autotune** searches `(BLOCK_M ∈ {16, 32}, BLOCK_N ∈ {64, 128, 256},
num_warps ∈ {4, 8})` keyed on `(B, N, D)`.

## `fused_masked_knn_topk` — V2 sparse path

[`kernels/triton/linr/fused_masked_knn_topk.py`](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py).

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
`tl.where(in_count, dots, -inf)`. Positions past `P` (block tail) are
masked at store time. The host then `torch.topk(scores, min(k, P))` and
gathers global ids from `positive_indices`.

**Autotune** searches `(BLOCK_N ∈ {32, 64, 128, 256}, num_warps ∈ {4, 8})`
keyed on `(P, D)`.

## `oporp_1bit_match_topk` — V3 (all paths)

[`kernels/triton/linr/oporp_1bit_match_topk.py`](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py).

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

**Popcount** is the [SWAR bit-twiddle](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py)
(`_popcount_int64`): five mask-shift-add steps, no libdevice dependency.
The matching torch reference [`popcount_int64`](../../retrieve/src/retrieve/layers/utils/quantize.py)
uses the exact same algorithm so torch and Triton produce **bit-exact**
identical scores.

**Launch grid** `(B, cdiv(n, BLOCK_N))`, modeled on `bloom_match` (the
already-winning kernel of identical shape: int64-word reduction over `W`).

**HAS_INDICES path**: replaces the contiguous `n_off` load with an
indirect `pos_indices[b, n_off]` lookup; otherwise identical. Used for
candidate-set rerank and for the masked V3 path (after `compact_mask`).

**Autotune** searches `(BLOCK_N ∈ {64, 128, 256, 512}, num_warps ∈ {4, 8})`
keyed on `(n, W, HAS_INDICES)`.

## `clause_compact` — fused clause eval + stream compaction

[`kernels/triton/filters/clause_compact.py`](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py).

Powers `ClauseIndex.evaluate_indices`. Avoids materializing the dense
`[B, N]` bool that `evaluate_mask` would otherwise produce, then doing a
host-side argsort to compact it. One launch produces the `(positive_indices,
counts)` pair the V2/V3 sparse paths consume.

```
inputs:   item_clause_attrs   [N, C, A_max]  int64
          clause_is_reverse   [C]            bool
          query_clause_attrs  [B, C]         int64
output:   out_indices         [B, N]         int64  (worst-case scratch)
          counts              [B]            int64
return:   positive_indices    [B, P]         int64  (P = max(counts.max(), 1))
          counts              [B]            int64
```

**Launch grid** `(B, cdiv(N, BLOCK_N))`. Each program owns one
`(query, n-tile)` cell and produces:

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

**No autotune.** A single fixed config (`BLOCK_N=256`, `num_warps=4`) is
hard-coded — `tl.atomic_add` accumulates across autotune trials and would
corrupt `counts`. If a sweep is needed later, re-enable autotune with
`reset_to_zero` covering both `counts_ptr` and `out_indices_ptr`.

## Layer dispatch

Each LinR Triton subclass is a thin router from the `forward()` signature
to one of the three kernels above. The dispatch is **design-time** — V1
is always dense, V2 is always sparse, V3 always uses popcount — not a
runtime sparsity heuristic. Pick the version that matches your expected
mask shape, not the one that benches best on a given input.

| layer                       | path         | kernel(s) called                                  |
|-----------------------------|--------------|---------------------------------------------------|
| [`LiNR_V1_Triton`](../../retrieve/src/retrieve/layers/linr/v1_triton.py)   | always dense | `fused_matmul_topk` (with optional inline mask)   |
| [`LiNR_V2_Triton`](../../retrieve/src/retrieve/layers/linr/v2_triton.py)   | masked       | `compact_mask` → `fused_masked_knn_topk`          |
|                                                                | unmasked     | `fused_matmul_topk` (V2 has nothing to pre-filter)|
| [`LiNR_V3_Triton`](../../retrieve/src/retrieve/layers/linr/v3_triton.py)   | full         | `oporp_1bit_match_topk` (HAS_INDICES=False)       |
|                                                                | masked       | `compact_mask` → `oporp_1bit_match_topk` (HAS_INDICES=True) |
|                                                                | candidates   | `oporp_1bit_match_topk` (HAS_INDICES=True)        |

V3's masked path always compacts and uses HAS_INDICES — popcount is cheap
enough that the gather penalty never crosses the dense-fallback break-even
point that V1 would hit. (The fp32 dot has a much higher per-item cost,
so V1's dense + inline-mask wins; that argument doesn't transfer to V3.)

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

## Numerics

Two failure modes worth keeping in mind:

1. **`tl.dot` reduction order vs cuBLAS.** `fused_matmul_topk` reduces in
   tensor-core tile order; cuBLAS reduces in row-major order. fp32 scores
   drift ~1e-4–1e-3 absolute, which can flip top-K ordering at ties near
   the K-th boundary. Parity tests use [`assert_topk_matches`](../../retrieve/tests/parity/conftest.py)
   (set + sorted-score tolerance); the bench harness uses [`topk_matches`](../../retrieve/tests/bench/conftest.py).
   Bit-exact parity is **not** an invariant for V1/V2 and never was.

2. **OPORP popcount is bit-exact.** V3's torch reference and the Triton
   kernel both use the same SWAR popcount on the same packed bits, so
   `recall@K_torch` and `recall@K_triton` agree to the third decimal
   in [bench.md](bench.md). If they ever diverge, a popcount or packing
   bug has been introduced — the kernel's correctness depends on bit
   identity here.

## SilverTorch kernels (out of scope, kept for completeness)

[`codesigned_probe_score`](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py)
is the actively-used SilverTorch kernel and powers
[`SilverTorch.forward`](../../retrieve/src/retrieve/layers/silvertorch/main.py).
Fuses three steps into one launch: probe the top-`n_probe` IVF clusters,
score the union of their items via INT8 dequantize + dot, AND the bloom
attribute filter inline (skipped entirely when `query_clause_attrs=None`).
Autotune key includes `HAS_QB` and `HAS_MASK` so the cluster-only,
bloom+cluster, and externally-masked paths each get their own configs.

[`bloom_match`](../../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py)
is the standalone bloom-filter subset test — `(qb & sigs) == qb` reduced
over W int64 words. 2-D grid, the structural model the OPORP kernel was
based on. Wins 10–20× over the torch broadcast version. SilverTorch fuses
this primitive inline into `codesigned_probe_score`, so the standalone
kernel currently has no production caller — only bench tests.

[`int8_ann_fused`](../../retrieve/src/retrieve/kernels/triton/silvertorch/int8_ann_fused.py)
is the standalone int8 candidate scorer (gather codes → cast fp32 → dot
→ scale). After V3's pivot to 1-bit, no LinR layer consumes this kernel
either; it remains in the SilverTorch tree pending a focused SilverTorch
pass that should either move it to a `dp4a`/`mma.s8` int32 accumulator
path (per the SilverTorch paper) or remove it. Untouched in this run.
