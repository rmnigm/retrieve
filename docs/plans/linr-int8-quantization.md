# `linr_v4` — int8-quantized KNN (single-stage + cascade variants)

> **Status: planned.** Adds two new algos to the LiNR family at the int8
> quantization level — between fp16/fp32 (full quality, full memory) and
> 1-bit `linr_v3` (lossy, needs fp32 rescore). Builds on existing
> `quantize_int8` and the `codesigned_probe_score` int8 kernel pattern;
> no new quantization math required.

## Context

LiNR today ships three algos:

| Algo | Stage 1 | Stage 2 | Memory vs fp16 |
|---|---|---|---|
| [`linr_v1_filter_mask`](../../evaluation/retrieval/algos/linr_v1.py) | fp32 dense + mask + topk | — | 1× |
| [`linr_v2`](../../evaluation/retrieval/algos/linr_v2.py) | filter → sparse fp32 rescore | — | 1× |
| [`linr_v3`](../../evaluation/retrieval/algos/linr_v3.py) | 1-bit Sign-OPORP Hamming → top-`candidate_pool` | fp32 rescore via `PrefilterKNN` | ~0.06× (1-bit) + 1× rescore index |

The 1-bit cascade gives ~16× memory savings on the prefilter index but
requires a full-precision rescore stage to recover recall. Int8 fills the
middle: ~4× memory cut vs fp16 with near-exact recall (≥0.95 typical) in a
single pass. We ship **two** algos so the sweep can compare the tradeoff:

- **`linr_v4`** — single-stage int8 full scan. Analog of `linr_v1` but with
  int8 codes. No fp32 index resident, lowest memory.
- **`linr_v4_rescore`** — two-stage cascade: int8 prefilter →
  `PrefilterKNN` fp32 rescore. Analog of `linr_v3` with int8 swapped in
  for 1-bit Hamming. Higher recall ceiling, holds both indexes in HBM.

A large fraction of the machinery already exists.

## Reuse — do NOT reimplement

- [`quantize_int8(embs)`](../../retrieve/src/retrieve/layers/utils/quantize.py#L25-L33)
  — symmetric per-item: returns `(codes[N, D] int8, scales[N] fp32)`.
  Already covered by `test_quantize.py::test_int8_*`.
- [`PrefilterKNN`](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py)
  — used unchanged as stage-2 of `linr_v4_rescore`, exactly like
  `linr_v3` does.
- [`compact_mask`](../../retrieve/src/retrieve/layers/utils/compact.py)
  — converts optional `[B, N]` bool masks to `(positive_indices, counts)`
  for the kernel's indirect path.
- [`RetrievalModule`](../../retrieve/src/retrieve/interfaces.py#L60-L79)
  contract: `register_index` + `forward → (ids[B, K], scores[B, K])`.
- Kernel body for int8 dequant + dot + scale + topk —
  [`_codesigned_probe_score_kernel`](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py#L109-L122):

  ```python
  codes = tl.load(item_codes_ptr + ...).to(tl.float32)
  scales = tl.load(item_scales_ptr + ...)
  dots = tl.sum(codes * q[None, :], axis=1) * scales
  dots = tl.where(keep, dots, float("-inf"))
  ```
  Copy this body verbatim into the new kernel; drop the bloom branch;
  add the `HAS_INDICES` full-scan path.
- Structural template for one kernel covering full-scan + candidate +
  mask paths via a `HAS_INDICES: tl.constexpr` flag —
  [`oporp_1bit_match_topk`](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py).

## New files

### 1. `retrieve/src/retrieve/kernels/triton/linr/int8_match_topk.py`

New Triton kernel + wrapper. Modeled on `oporp_1bit_match_topk.py`:

- `@triton.jit` `_int8_match_topk_kernel` with `HAS_INDICES: tl.constexpr`
  toggling between contiguous full scan (`item_codes_ptr[n_off, :]`) and
  indirect gather (`item_codes_ptr[pos_indices[b, n_off], :]`).
- Per (B, n_tile): load `q[D]` fp32 once, load `codes[BLOCK_N, D]` int8
  and cast to fp32 in registers, load `scales[BLOCK_N]` fp32, compute
  `dots = tl.sum(codes_fp * q[None, :], axis=1) * scales`, write `-inf`
  for invalid lanes, store into `[B, N]` (full scan) or `[B, P]`
  (indirect).
- Autotune keys: `["N", "D", "HAS_INDICES"]` plus `do_not_specialize`
  on `P_REAL` and a `P_BUCKET` rounding ladder for the indirect path,
  matching [`fused_masked_knn_topk`](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L42-L65)
  to avoid recompile-per-distinct-P.
- Python wrapper `int8_match_topk(query, item_codes, item_scales, k,
  *, positive_indices=None, counts=None)` mirrors
  `oporp_1bit_match_topk`'s wrapper signature. Pads to K with
  `(-1, -inf)` when `P < K`. Decorate with `@torch._dynamo.disable`
  consistent with the existing wrappers (the upcoming
  `triton_op` migration in [02-triton-op-migration.md](02-triton-op-migration.md)
  will replace these decorators uniformly).

### 2. `retrieve/src/retrieve/layers/linr/int8_knn.py`

`Int8KNN(RetrievalModule)` — structural twin of
[`OneBitKNN`](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py) with
these substitutions:

- `register_index` calls `quantize_int8(item_embs)` and stores
  `item_codes [N, D] int8` and `item_scales [N] fp32` as buffers.
  **No OPORP signs/perm** — int8 with per-item symmetric scaling
  preserves cosine signal directly; no random projection needed.
- `_project_query` does not exist; the query stays fp32 throughout.
- Torch eager path:
  - Full scan: `(self.item_codes.float() * self.item_scales[:, None]) @
    query.T`, then mask + topk like
    [`SimilarityMasking.forward`](../../retrieve/src/retrieve/layers/linr/similarity_masking.py).
  - Candidate path: gather `codes[ids]` and `scales[ids]`, `bmm` with
    `query.unsqueeze(1)`, multiply by scales, topk — mirrors
    `PrefilterKNN._forward_torch_eager`.
- Triton path: three sub-cases like `OneBitKNN._forward_triton` —
  full scan / `candidate_ids` (build trivial `counts = P`) / `mask →
  compact_mask → indirect`. All call the new `int8_match_topk`.
- Constructor `Int8KNN(k: int, backend: Backend = "triton")`. No `seed`
  — int8 is deterministic.

### 3. `evaluation/retrieval/algos/linr_v4.py`

`LinrV4Algo` — single-stage. Direct twin of
[`linr_v1.py`](../../evaluation/retrieval/algos/linr_v1.py):

```python
self.idx = Int8KNN(k=k, backend=backend).to(item_embs.device)
self.idx.register_index(item_embs)
self.filter_mod = filter_mod
self.algo_modules = collect_modules(self.idx, filter_mod=filter_mod)
self.compile(dynamic=True, mode="reduce-overhead")
```
`forward(q, qa_narrow)` → `self.idx(q, mask=make_mask(self.filter_mod,
qa_narrow))`.

### 4. `evaluation/retrieval/algos/linr_v4_rescore.py`

`LinrV4RescoreAlgo` — two-stage cascade. Twin of
[`linr_v3.py`](../../evaluation/retrieval/algos/linr_v3.py) with
`OneBitKNN` replaced by `Int8KNN` in stage 1; `PrefilterKNN` stage 2
unchanged. Constructor takes `candidate_pool: int = 5000`; drop
`v3_seed` analog.

### 5. `evaluation/conf/deep_sweeps/goodreads-d128-linr_v4.yaml`

Clone [`goodreads-d128-linr_v3.yaml`](../../evaluation/conf/deep_sweeps/goodreads-d128-linr_v3.yaml):

```yaml
algorithms:
  - linr_v4
  - linr_v4_rescore

algo_params:
  linr_v4:
    - {}
  linr_v4_rescore:
    - {candidate_pool: 500}
    - {candidate_pool: 1000}
    - {candidate_pool: 2000}
    - {candidate_pool: 4000}
    - {candidate_pool: 8000}
```

Pool sizes deliberately smaller than v3's (2k–32k) — int8 stage-1 is
much closer to exact, so a 500–2000 pool should already saturate
recall.

## Modified files

### `evaluation/retrieval/algos/__init__.py`

- Append `"linr_v4"`, `"linr_v4_rescore"` to `ALGORITHMS` and
  `BACKEND_CAPABLE_ALGOS`.
- Add factory branches in `build_algorithm` mirroring the existing
  `linr_v1` / `linr_v3` branches. `linr_v4` takes no extra params;
  `linr_v4_rescore` reads `candidate_pool` from `params` (default 5000).

### `evaluation/conf/goodreads/d128-quality.yaml`

Add `linr_v4` and `linr_v4_rescore` (with 2–3 representative pool sizes)
to `algorithms` and `algo_params` so the standard quality sweep picks
them up alongside `linr_v3` / `silvertorch`.

### `retrieve/tests/correctness/test_linr.py`

Extend, don't rewrite:

- New `TestInt8KNN` class mirroring
  [`TestOneBitKNN`](../../retrieve/tests/correctness/test_linr.py)
  with a tighter recall floor: **≥0.95 recall@K=200** vs the exact
  reference (vs 0.4 for 1-bit). Reuses the existing `data` fixture and
  `BACKENDS` parametrization.
- Extend
  [`TestCrossBackendAgreement`](../../retrieve/tests/correctness/test_linr.py#L111)
  to include `Int8KNN` for full-scan and candidate-id paths via
  `assert_topk_id_sets_match`.
- Extend `TestEdgeCases` to parametrize over `Int8KNN`: all-true mask,
  all-false mask, P=0, B=1, K=N.

## Correctness notes

- **No OPORP for int8.** 1-bit needs Sign-OPORP to spread signal before
  sign quantization; per-item symmetric int8 preserves cosine directly.
  Skipping OPORP also avoids carrying `signs[D]` / `perm[D]` state and
  keeps the query path fp32 end-to-end. If a future variant wants
  OPORP-int8 it can be layered on later.
- **Score formula** matches
  [`codesigned_probe_score`](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py#L121):
  `dots = tl.sum(codes_fp * q, axis=1) * scale_i` — per-row multiply
  *after* the reduction, not folded into the codes load. This keeps
  the inner reduce in fp32 with a single scalar multiply per row.
- **`linr_v4_rescore` holds both indexes resident** (`Int8KNN` int8
  codes + scales; `PrefilterKNN` fp32 embeddings). Memory is
  `N·D` int8 + `N·4` fp32 + `N·D·4` fp32 ≈ 1.25× fp32 baseline — the
  expected tradeoff vs `linr_v4` alone (~0.25× fp32, ~0.5× fp16).
- **No backward-compat layer needed**: `Int8KNN` is brand new, no
  callers, no migrations.

## Verification

1. **Correctness suite:** `cd retrieve && uv run pytest
   tests/correctness/test_linr.py -k "Int8 or Cross or Edge" -v` —
   new `TestInt8KNN`, extended cross-backend and edge-case
   parametrizations all pass. Recall floor ≥0.95 holds on the standard
   fixture.
2. **Quantize unit tests** already cover `quantize_int8` roundtrip,
   range, clamping, zero-input — no new tests needed there unless a
   gap surfaces.
3. **Registry smoke:** from `evaluation/retrieval/`:
   `uv run python -c "from algos import ALGORITHMS; print(ALGORITHMS)"`
   shows `linr_v4` and `linr_v4_rescore`.
4. **Sweep:** `cd evaluation && uv run evaluate --config
   conf/deep_sweeps/goodreads-d128-linr_v4.yaml` — produces a results
   JSON. Expect `linr_v4` recall within ~1 point of `torch_knn`;
   `linr_v4_rescore` essentially matching `torch_knn` already at
   `candidate_pool=1000–2000`. Latency for `linr_v4` should land
   between `linr_v1_filter_mask` (more memory bandwidth) and
   `linr_v3`'s stage-1 (less arithmetic work).
5. **Cross-algo sweep:** `uv run evaluate --config
   conf/goodreads/d128-quality.yaml` — `linr_v4*` show up alongside
   `linr_v3` and `silvertorch` in the joint output.
6. **Manual sanity:** REPL, `N=2048, D=128`, random unit-norm index;
   compare `Int8KNN` top-100 ids vs `FullScanKNN` top-100 ids;
   confirm recall ≥ 0.95. Automated by the unit test but worth
   eyeballing once.