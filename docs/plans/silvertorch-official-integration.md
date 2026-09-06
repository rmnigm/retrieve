# Official SilverTorch ops as the reference backend — integration, parity, Triton-vs-official

> **Status:** planned 2026-09-05 on `feat/cute-dsl-scorer` (nothing implemented, no code touched).
> Target box: A100-SXM4-80GB, torch 2.10.0+cu128, CUDA 12.x toolchain, triton 3.6.0, Python 3.11.
> Authored on the Mac (no GPU): every "the official op does X" claim cites `silvertorch/ops/csrc/<file>:<line>`
> in the clone of [meta-recsys/silvertorch](https://github.com/meta-recsys/silvertorch) at `21aa35e`
> (2026-07-23, `main`, no tags, Apache-2.0); every "we do Y" claim cites a file:line in this repo.
> Scope decision taken by the user the same day: **the official ops become the reference backend for
> Algorithm 1 phases 2+3; the hand-written CUDA C++ backend and its CuTe DSL port are both deleted**
> (git history and archived plans keep their record; the CuTe port's no-`nvcc` advantage vanishes
> once the official package needs `nvcc`); `triton` stays as our easier-to-maintain reimplementation,
> `torch` stays as the readable eager reference, and kernel effort goes into Triton (§8).
>
> Question this plan answers: *how do we run Meta's own kernels inside our layer and harness so that
> (1) they are the ground truth our Triton kernels are checked against, and (2) the paper can say, on
> public data and the same A100, whether a ~600-line Triton reimplementation is competitive with the
> vendor's ~9.4k-line CUDA C++?*
>
> **Ordering authority:** [00-roadmap.md](00-roadmap.md) §1 — WP-0/1 are Phase A2/A3, WP-2..5
> are Phase B, WP-6 folds into Phase C1/C4, WP-7 into D1, WP-8 (TF-1) is Phase G-a, WP-9 is F2.
> Validation records go in a "§13 Validation record" section appended to this file.

## 1. Where it starts

**Official repo**: 12 commits (2026-04-21 → 2026-07-23), 9,427 lines of `.cpp/.cu/.cuh/.h` under
`silvertorch/ops/csrc/` (≈ 3.5k for `fused_kmean_ann*` + `simple_index_mm.cuh` + `index_mm_helpers.cuh`,
≈ 4.5k for the bloom index/search/parser, the rest `is_topk`, fresh-index merge, `faster_repeat_interleave`),
29 stars, 7 open issues, **not on PyPI** (`https://pypi.org/pypi/silvertorch/json`
→ 404; the README's `pip install silvertorch` does not work, install is from git). One extension
`silvertorch._C` via `setup.py` (`CUDAExtension` when `CUDA_HOME` is set, else CPU-only, `setup.py:62-83`),
`--no-build-isolation` required (`pyproject.toml` build-requires `torch>=2.0`), ops under
`torch.ops.st.*` after `import silvertorch.ops._load_ops` (`_load_ops.py:34`). No fake/meta kernels
anywhere (repo-wide grep for `register_fake|impl_abstract|custom_op|DispatchKey::Meta` is empty).

**Ours**: `SilverTorch` ([main.py](../../retrieve/src/retrieve/layers/silvertorch/main.py)),
`backend ∈ {triton, torch, cuda, cute}` ([interfaces.py:8](../../retrieve/src/retrieve/interfaces.py)),
phase 1 host-side (main.py:311-319), phases 2+3 in one Triton launch
([codesigned_probe_score.py](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py)
310 lines, exact variant 294, [common.py](../../retrieve/src/retrieve/kernels/common.py) 110,
[bloom_hash.py](../../retrieve/src/retrieve/layers/filters/bloom_hash.py) 165), plus the two-kernel
CUDA/CuTe backends this plan removes (§7).

### 1.1 Inventory — Algorithm 1 phases vs the official ops

| phase (paper Alg. 1) | official op | ours | for `backend="official"` |
|---|---|---|---|
| k-means / IVF build | **none** — README "Index build flow" (`fused_kmean_ann.cpp:331-337`): bring your own k-means, sort items by cluster, CSR `cluster_offsets` | `KMeansTorch` + padded layout (main.py:171-210) | ours; needs a **cluster-sorted** table + `cluster_offsets[n_lists+1]` |
| int8 quantization | **none**; op takes int8 embeddings *and int8 queries* (`fused_kmean_ann_cuda.cu:1296`), scale via `divisor_for_int8` (int) or `per_embedding_scale` (fp16 `[N]`, divides) | `quantize_int8_global` + per-row `quantize_int8` ([quantize.py:24-43](../../retrieve/src/retrieve/layers/utils/quantize.py)) | ours; scale applied host-side (§4.2) |
| bloom index build | `bloom_index_build(feature_ids int32[F], feature_offsets int64[N·F+1], feature_values int64, b_multiplier>1.0, k, fast_build)` — CPU `bloom_indexer.cpp:27-112`, CUDA `bloom_indexer_cuda.cu:755-878` | `build_signatures` `[N, W]` | **official** (their hash) |
| query predicate | `parse_expression_query_batch(str[], ks, hash_k, return_plan, max_sub_queries)` — CPU parser, AND/OR/NOT/parens (`EXPRESSION_SYNTAX.md`); `ks` is unused (`expression_query_parser.cpp:395`) | `[B, C]` attrs → `build_query_signatures` | official; plans cached (§4.4) |
| phase 2 partial mask | `bloom_index_search_batch_return_partial_response(index, b_offsets, plans, selected_cluster_offsets [B,P], selected_cluster_lengths [B,P], k, hash_k, plan_index?) → (column_counts_cumsum int32, first_item_offset_in_column int8, column_mask_response int64)` (`bloom_index_search.cpp:560-571`, cuda `:1213-1262`) | fused row-wise subset test | official |
| phase 2 full mask | `bloom_index_search_batch(...) → [B, padded_N] bool` or packed `[B, padded_N/64] int64` (`:940-1030`) | `bloom_match` | official; S9 baseline |
| phase 3 scoring | `fused_kmean_ann(cluster_offsets, cluster_ids [B,P], cluster_length [B,P], embeddings, queries, max_tensor_size_per_row, filtering_bit_mask?, invalid_index_value, divisor_for_int8, filtering_bit_index?, per_embedding_scale?) → (scores [B, M32], indices int32)` and `fused_kmean_ann_with_partial_masks(... column_counts_cumsum, first_item_offset, column_results, ...)` (`fused_kmean_ann.cpp:400-434`) | Triton scorer | **official = ground truth** |
| top-k | **none** — caller runs `torch.topk`; `is_topk` only marks top-k positions (full sort, ties may mark > k; `is_topk.cpp:22-39`, `is_topk.cu:78-91`) | host `masked_topk` | ours, identical in both arms |
| exact / clause | **none** | `codesigned_probe_score_exact`, `clause_mask` | ours; on official via `filtering_bit_mask` (§5.1) |
| NOT | `NOT`/`!` in expressions | exact mode only | official bloom gains Goodreads `c1_lang_reverse` |
| live update | `take_top_k_and_gather_from_main_and_fresh` (`fresh_index_post_processing.cpp:273-285`) | none | out of scope ([live-update-api.md](live-update-api.md)) |

Facts a reimplementer needs: `DIM ∈ {16,32,64,96,128,192,256,384,512,768,1024,1280}` else
`TORCH_CHECK` (`fused_kmean_ann_cuda.cu:1431-1487`); int8 rows read as `int4` (`simple_index_mm.cuh:80,172-173`,
so DIM % 16 == 0 and allocator-aligned rows); `__dp4a` gated on `__CUDA_ARCH__ >= 610`; the
`__launch_bounds__(256, 2)` double-buffered `process_cluster_v4_pipelined` (`:878-962`) is **dead
code** — `try_launch_pipelined_kernel` (`:1188`) has no call site; the live int8 scorer is
`process_cluster<int8_t, Half, DIM, FILTER, int32_t, false>` (`:964-1030`): one **thread per
document**, each lane streams its own `DIM/16` `int4` words, query read from global per warp, 32
rows in flight per warp; cluster tails `< 32` go to `process_cluster_remaining` (`:1033-1074`).
`max_tensor_size_per_row` is rounded up to a multiple of 32 (`fused_kmean_ann.cpp:107`,
`cuda.cu:1873-1874`); output slots per row are all warp-aligned segments in probe order, then all
remainders (`:535-541`, `:402-407`) — fine for `topk`, not a concatenation. Score dtype for int8:
**fp16** with a divisor/scale, else **int32** (`fused_kmean_ann.cpp:29-39`); padding = dtype
minimum (`-65504`, `INT32_MIN`; `simple_index_mm.cuh:99-100,150-152`), indices `invalid_index_value`.
Bloom: `k ≤ 10` is an **unchecked** hard limit (`MAX_K_V2`, `bloom_index_util.h:45`, fixed-size
arrays at `bloom_index_search_cuda.cu:254`); build with `k` as the README does — the module builder
passes `hash_k` as the build-time k (`bloom_index_search_module_builder.py:92-98`), which sets
extra bits per term. **Neither official CPU reference is an oracle**: `fused_kmean_ann_cpu` reads
masks LOW-bit-first (§4.3) and `fused_kmean_ann_cpu_with_partial_masks` ignores its masks entirely
(`fused_kmean_ann.cpp:236-260`); their tests only ever pass all-pass masks (`tests/test_fused_kmean_ann.py:621-652`).
Other audit notes for the paper's build-friction/code-audit paragraph: every "bloom index v1"
template branch is unreachable (all registered ops instantiate `<true>` = v2); three functions are
declared but never defined or registered (`generate_bloom_column_mask_for_selected_clusters`,
`generate_cluster_column_info_for_jagged_flow`, `mask_marginal_bits_from_column_response`,
`bloom_index_util.cuh:294-333`); the bloom-index bit helpers exist (`is_document_valid_with_*_mask`,
`bloom_index_util.cuh:236-268`) but the scorer re-implements the arithmetic inline at six sites; the
CUDA entry points check device/contiguity/dims but not dtypes of `cluster_*`/`embeddings`
(`fused_kmean_ann_cuda.cu:1797-1832`), so a wrong dtype fails inside `packed_accessor64`. Python
side: `BloomIndexSearchModule` holds `bloom_index`/`bloom_bundle_b_offsets` as buffers and wraps
`bloom_index_search_batch` (`modules/bloom_index_search_module.py:22-73`); `FilterQueryParserModule`
is stateless; neither adds anything our adapter needs, so we call `torch.ops.st.*` directly.

## 2. Decisions

- **D1 — `backend="official"`** joins `triton` and `torch`; `Backend = Literal["torch", "triton", "official"]`.
  It is the *reference*: `triton` correctness is defined against it (D5) and it is the vendor arm of
  the paper's comparison (D8).
- **D2 — Pinned git dependency (option A), not vendored, not a submodule.** The paper wants "Meta's
  code at commit X, unmodified"; vendoring 9.4k lines of C++ is the maintenance burden being
  removed; a submodule adds nothing over a sha in `uv.lock`. The adapter imports only `torch.ops.st`,
  so a switch to a path source (C) is a one-line change (§6.1). Zenodo snapshot includes the sdist.
- **D3 — One layer, shared phase 1.** `SilverTorch(backend="official")` keeps our k-means,
  quantization and probe selection and routes phases 2+3 to `torch.ops.st.*`: both arms score the
  *same clusters and the same int8 codes*; only kernels differ. A separate module would re-derive an
  IVF the official repo does not ship.
- **D4 — Buffers are cluster-sorted for `official`** (`item_codes` in CSR order + `sort_perm`/`inv_perm`,
  no `padded_cluster_items`); state dicts are not portable to/from this backend (as `bloom_sigs_t`
  already was, architecture.md:301-311).
- **D5 — Correctness contract.** Phase 3: official `divisor_for_int8=-1` returns the raw int32 dot;
  our scores must equal `(dot.float() * q_scale) * global_scale` with `torch.equal`, ids equal up to
  ties. Phase 2 bloom: semantic — official mask ⊇ exact mask (no false negatives) on both backends,
  FPR measured on both, `_with_partial_masks` scores `== filtering_bit_mask` scores (paper §4.3 "the
  results remain the same"). Exact/clause keeps its torch-reference gate (no counterpart upstream).
- **D6 — Bloom on `official` is the official bloom** (their hash, bundles, parser); Triton keeps
  `bloom_hash.py`. Fairness = matched FPR *and* matched memory, both measured (§4.3).
- **D7 — `official` runs eager only.** Every op syncs the host (§3), plan buffers are re-decoded and
  re-uploaded from pageable memory per call, partial-response output shapes are data-dependent —
  no CUDA-graph capture, no `torch.compile`. The harness records `mode: graph` as `null` with
  reason `not_capturable`; the paper's headline is **eager vs eager**, Triton's graph number
  alongside as deployed-best-case (harness-v2 §2.7).
- **D8 — The paper question is Triton vs official** (phase 3 kernel, phase 2 kernel, end to end),
  `torch` eager as the floor. CUDA/CuTe numbers appear only as quotes from the archived
  [cuda-silvertorch-handoff.md §13](cuda-silvertorch-handoff.md) and [cute-dsl-scorer.md §5](cute-dsl-scorer.md).
- **D9 — Deletion happens after the official parity gate (WP-3), never before** (§9).
- **D10 — Timing protocol is harness-v2 §2 verbatim**, extended with the official sha, `nvcc --version`,
  and the parse-cost row (§8e).

## 3. Host-side behaviour of the official ops (decides fairness)

| op | syncs per call | launches / allocations per call | capturable |
|---|---|---|---|
| `fused_kmean_ann` (`fused_kmean_ann_cuda.cu:1797-1926`) | `repeat_interleave(cluster_warp_size)` without `output_size` (`:575`, hidden D2H) and `cluster_remaining_length_cumsum[-1].item<int32_t>()` (`:465`) — **2, unavoidable** | `round_cluster_to_warp` 3 tensors + 1 kernel (`:314-353`); 2 `cumsum`; `arange`+`repeat_interleave`; `cumsum`; 2 payload buffers + 2 kernels (`:560-608`, `:452-494`); `at::full` ×2 (`:1272-1276`, `:1421-1427`); `process_cluster` + `process_cluster_remaining` — **≈ 12 launches, ≈ 11 allocations** | no |
| `fused_kmean_ann_with_partial_masks` (`:1928-2100`) | `.item()` at `:2054` and `:2072` unless the caller passes the four precomputed tensors and both `total_cluster_*` ints (`:2052-2072`); `repeat_interleave` at `:697` **always** — 1–3 | as above | no |
| `bloom_index_search_batch` (`bloom_index_search_cuda.cu:1030-1110`) | `plans_data.cpu()` (`:1045-1046`) — a D2H copy only if plans live on CUDA (the README's `.cuda()` advice adds one; keep plans on CPU); host loop decoding every plan (`:1051-1083`) and a fresh pageable `cudaMemcpyAsync` upload **every call** (`:117-126`, `:169-173`) | 1 `at::empty` + 1 `process_documents` kernel, one thread per (query, 64-doc column) (`:575-615`) | no |
| `..._return_partial_response` (`:1213-1262`, `:1174-1210`) | plan decode/upload as above; `repeat_interleave(column_counts)` (`:1186-1190`) — ≥ 1; output size data-dependent | `generate_column_info_for_clusters` 3 tensors + 1 kernel (`:896-938`); `cumsum`; `at::zeros`; 1 process kernel | no |
| `bloom_index_build` (CUDA) | `[-1].item()` ×2 (`bloom_indexer_cuda.cu:796-798`); output size data-dependent | CUB reduce-by-key + build kernels; `fast_build=True` selects a single signature-free warp-cooperative build (`:838-878`, "SAVE_MEM") instead of the two-pass signature-then-transpose build; the CPU build ignores the flag (`bloom_indexer.cpp:32`) | n/a (build time) |
| `*_multiple` variants (`fused_kmean_ann_with_partial_masks_multiple`, `bloom_index_search_batch_return_partial_response_multiple`) | `all_totals_gpu.cpu()` (`fused_kmean_ann_cuda.cu:2285`); `per_chunk_cumsum[-1].item()` per chunk (`bloom_index_search_cuda.cu:1495`) | pointer-array uploads per call; `fuse=True` runs one multi-chunk kernel | no — sharded-index API, not needed here |
| `generate_column_info_for_clusters` alone | none; output shape = `selected_cluster_lengths.numel()` | 3 tensors + 1 kernel | yes (the one capturable op) |
| `parse_expression_query_batch` | CPU: `std::stoll` tokenizer, `hash_k` murmur3 hashes per term at parse time (`expression_query_parser.cpp:359-362`), plans cloned into tensors (`:382-387`) | — | cache |

The syncs are not intrinsic: the repo ships a `faster_repeat_interleave` op family whose
`_with_cumsum_raw` variants take an explicit host output size (`faster_repeat_interleave.cuh:24-73`,
`faster_repeat_interleave_helper.h:49-53`), but `fused_kmean_ann_cuda.cu` never calls it — with it
plus the precomputed `total_cluster_*` ints the scorer's shapes would already be static
(§3 table) and capture would be possible. Worth an upstream issue, not our patch (D2).

`max_tensor_size_per_row` is a host `int`; for our padded IVF it is the static `n_probe · max_cluster_size`
(main.py:185), so no per-batch `cluster_length.sum(1).max().item()` and the official output has our
`[B, P]` width (rounded to 32) — the host `topk` epilogue costs the same in both arms. A
`backend="official"` forward is ≈ 12 launches + 2 syncs (no filter) or ≈ 20 launches + ≥ 3 syncs
+ a host plan decode (bloom) against Triton's 1 launch: expect the official arm to lose at small
`P`/`B=1` for host reasons; only at `B=16, P ≈ 58k` is the comparison about kernel bandwidth. The
kernel-only tier (§8a) separates the two.

## 4. Numerics and attribute mapping

### 4.1 Layout: padded IVF → official CSR

`sort_perm = argsort(assignments)` (main.py:195) and `cluster_offsets` (main.py:197-198) already
exist. Official needs `embeddings = item_codes[sort_perm]`, `cluster_ids = probe_ids [B, n_probe]`,
`cluster_length = cluster_sizes[probe_ids]` (int64). Returned `indices` are sorted-table positions
→ `ids = where(idx >= 0, sort_perm[idx.clamp_min(0)], -1)`. Empty clusters contribute no slots;
pads carry `-65504`/`-1` and map to `-inf`/`-1` for `masked_topk`.

### 4.2 Scale convention

Ours: `(dot.to(f32) * q_scale[b]) * global_scale` (two left-associated fp32 multiplies, bit-identical
across triton/torch — kernels.md "Numerics"). Official consumes our `q_codes` from `quantize_int8(query)`
and offers: (i) `divisor_for_int8 = -1` → **int32 raw dot** (`fused_kmean_ann.cpp:38`, `cuda.cu:1498-1502`);
the host epilogue above reproduces our scores **bit-exactly** — the parity path; (ii)
`divisor_for_int8 = 2^k` → fp16 `float(dot) / divisor` (`index_mm_helpers.cuh:53-56`): `|dot| ≤ 127²·D`
= 2,064,512 at D=128 > 65,504, so the divisor must be ≥ 32 (D=64: 16, D=256: 64); a power of two
keeps the division exact, leaving only fp16 rounding (rel. 2⁻¹¹); host epilogue
`scores.float() * (divisor * q_scale * global_scale)` — the **timed path** (the instantiation Meta
ships for int8 serving), ranking agreement vs (i) gated at `jaccard@k ≥ 0.99` and reported; (iii)
`per_embedding_scale` is **unusable**: the kernel casts the raw int32 dot to fp16 *before* dividing
(`index_mm_helpers.cuh:50-52`), overflowing to `±inf` for any D ≥ 5 at full-range codes — a
reproducibility finding to report; the S6/S16 global-vs-per-row ablation therefore runs on path (i)
with per-row scales applied in the host epilogue (same kernel, same dots).

### 4.3 Bloom: never bit-comparable — match FPR and memory

Official: bundles of 2048 docs (32 columns × 64, `bloom_index_util.h:42-44`); per bundle
`B = int(max_terms_per_doc_in_bundle · k · b_multiplier)` bits per doc (`bloom_indexer.cpp:52-66`),
index = `Σ_bundles B · 32` int64 words = **`B/8` bytes per doc** (B = 1024 → 128 B, our `m_bits=1024`
row); term → `k` positions by `murmur_hash3_2x64(feature_id, value, seed++) % B` with duplicate
rejection (`bloom_index_util.cuh:184-206`; the parser stores the `hash_k` raw hashes, `% B_bundle`
happens at search time, `bloom_index_search_cuda.cu:258-269`); doc `d` at `1 << (63 - d % 64)`
(`bloom_indexer.cpp:107`). Ours: splitmix-style mix, per-clause salt, fixed `m_bits`
(bloom_hash.py:63-95). Their width tracks the bundle *maximum* term count, ours is fixed, so
matched memory ≠ matched FPR. Per dataset/sweep, on the real attributes and query pool: (1)
**matched memory** — the global `b_multiplier` at which `bloom_index.numel() == bloom_sigs.numel()`
within 2 %; (2) **matched FPR** — bisect `b_multiplier ∈ [1.5, 20]` until the measured pass rate on
the sweep's pool equals ours ±10 % relative (`k = k_hash = 5`, `hash_k = 7`); the paper's own
`bits = max_values · K · 3` heuristic (S8) is a third point. Report all.

Search semantics of `k` vs `hash_k`: the plan carries `hash_k` raw hashes per term; a term ANDs
the first `k` *distinct* positions among them (`while (p < p_k && resolved < k)`,
`bloom_index_search_cuda.cu:258-276`), so with fewer than `k` distinct positions available it ANDs
fewer rows — a weaker filter, never a false negative — which is why `hash_k` must exceed `k` by a
margin (README uses 7 for `k=3`; we use 7 for `k=5`). `AND`/`OR`/`NOT`/`EMPTY` combine whole
64-doc words with `&`, `|`, `~`, `~0` on a small on-thread stack (`:294-461`; a fixed `Stack<16>`
fast route when the plan's max depth ≤ 16, `:887-891`). Partial-response outputs: `column_counts_cumsum
[B·P] int32` = running count of the 64-doc columns each `(query, probe)` range touches
(`ceil((len + start % 64) / 64)`, `:620-643`), `first_item_offset_in_column [B·P] int8 = start % 64`,
`column_mask_response [Σ columns] int64` in `(query, probe, column)` order — the scorer indexes it
at `column_results + cumsum[idx-1]`, bit `first_offset[idx] + doc_offset` (`fused_kmean_ann_cuda.cu:661-673`).
`filtering_bit_index [B] int64` (both scorer entry points) remaps query rows onto shared mask rows —
useful if several queries share one predicate; `query_plan_index` does the same for plans.

**Mask bit order — a trap.** The GPU scorer reads HIGH-bit-first: `get_bit_32_bit_mask` /
`get_next_32_bit_mask` (`bloom_index_util.cuh:125-134,156-180`, "lower doc id put at higher bits"),
and the packed `return_bool_mask=False` output stores the raw word in that order
(`bloom_index_search_cuda.cu:552-556`). The CPU reference reads LOW-bit-first
(`fused_kmean_ann.cpp:71-81`). Our packer (§5.1) follows the GPU; parity test T3 pins it.

### 4.4 Attributes → features and expressions

`item_attrs_narrow.pt` is `[N, C, A_max]` int64 with `-1` pads (Goodreads and arXiv `[N, 5, 4]`,
datasets.md:105, `eval_datasets/arxiv.py:67,781-835`). Mapping: `feature_ids = arange(C).int32`
(clause index as feature id — the same `(clause, value)` keying as our salt), `feature_values` = the
non-`-1` entries in `(doc, clause)` order, `feature_offsets` = the CSR over `N·C` slots, all in
cluster-sorted doc order so `selected_cluster_offsets = cluster_offsets[probe_ids]` addresses the
same doc space. Queries `[B, C]` → `" AND ".join(f"{c}:{v}")` over active clauses, reverse clauses
→ `NOT c:v`, none active → `""` (EMPTY = match all, `expression_query_parser.cpp:400-405`). Grammar:
`feature_term = NUMBER ":" NUMBER [":" NUMBER]` (the optional weight is parsed and discarded,
`:345-357`), precedence NOT > AND > OR, parentheses; `max_sub_queries` (default 5) bounds the fan-out of
one AND/OR node by folding operands into nested compound ops (`:195-284`), semantics unchanged. Plans
depend only on `(strings, hash_k, max_sub_queries)` (`tests/test_expression_query_parser.py:183-187`
round-trip; the CUDA registration of the parser just forwards to the CPU one,
`expression_query_parser_cuda.cu:38-53`), so the harness pre-parses its 4,096-batch pool once and
caches `(plans_data, plans_offsets)` on **CPU** per batch; the per-call decode/upload (§3) stays
inside the timed region because it is inside the op. Parse cost at `B=16` is measured with
`torch.utils.benchmark.Timer` as `tests/bloom_index_bench.py:141-165` does (that bench is CPU-only
and parses one large expression per call) and reported as its own row.

## 5. Adapter design

### 5.1 Files, buffers, filter matrix

- `retrieve/src/retrieve/kernels/silvertorch/official.py` (new, ≈ 250 lines): `is_available()` /
  `ensure_loaded()` (distinguishes "not installed" from "op missing", as
  [codesigned_probe_score_cuda.py:161-176](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_cuda.py)
  does today); `attrs_to_features(attrs_sorted)`; `queries_to_expressions(qa, clause_is_reverse)`;
  `parse_plans(expressions, hash_k)` with an LRU keyed by the string tuple; `pack_mask_high_first(bool
  [B, N]) -> int64 [B, ceil(N/64)]`; eager `official_scores(...)` / `official_bloom_partial(...)`
  returning `[B, P]` fp32 scores + int64 ids (`-inf`/`-1` pads) so `masked_topk` applies unchanged.
- `main.py`: `_forward_official`; `register_index` registers `item_codes` (sorted), `global_scale`,
  `cluster_offsets`, `cluster_sizes`, `sort_perm`, `inv_perm` (candidate path
  `item_codes[inv_perm[candidate_ids]]`); bloom: `bloom_index [W] int64`, `bundle_b_offsets`, attrs
  `b_multiplier`, `hash_k`; exact: `item_clause_attrs` **sorted** + `clause_is_reverse`. Frozen order
  `centroids, item_codes, global_scale`, then backend-specific.
- `filter_mode` on `official`: `none` → `fused_kmean_ann`; `bloom` → `_return_partial_response` +
  `_with_partial_masks` (default) or `bloom_index_search_batch(return_bool_mask=False)` +
  `filtering_bit_mask` when `official_bloom_path="full"` (S9 ablation); `exact` → our Triton
  `clause_mask` over sorted attrs + `pack_mask_high_first` + `filtering_bit_mask` (phase 2 ours,
  full-N, labelled so in every table).
- `torch.compile`: unsupported for `official`; `SilvertorchAlgo` passes `compile=False` through a
  new `AlgoBase._finalize(..., compile=True)` flag (harness-v2: `PATHS[("silvertorch", *, "official")]
  = "official"`, `bench.latency` skips `graph` with a reason). `torch.library.register_fake("st::…")`
  from our side is possible but pointless while the ops sync — deferred.

### 5.2 Tests — `retrieve/tests/parity/test_official.py` (+ `require_official()` in `tests/conftest.py`)

| # | gate | inputs |
|---|---|---|
| T1 | int32 path: official vs `ref_cps_phase23` int reference **bit-exact**, and vs `_codesigned_probe_score_impl` scores after the host epilogue with `torch.equal`; ids `assert_ids_equal_up_to_ties` | `make_probe_family` with `-1` pads, D ∈ {64, 96, 128}, B ∈ {1, 16} |
| T2 | fp16 path: `jaccard@k ≥ 0.99` vs T1, `max_rel_err ≤ 2⁻¹⁰` on finite slots | same |
| T3 | bit order: a mask with only doc 0 set passes doc 0 on GPU iff HIGH-first; `pack_mask_high_first` round-trips `return_bool_mask=True` vs `False` | README's 4-doc corpus |
| T4 | bloom no-false-negative: official partial mask ⊇ `clause_subset_match`, AND and NOT expressions; FPR printed, `< 5 %` at `b_multiplier=10` | random attrs `[N, 2, 2]`, vocab 50 |
| T5 | co-design invariance: `_with_partial_masks` scores `torch.equal` full-mask scores | same |
| T6 | layer: `SilverTorch("official")` vs `"triton"` on `make_index`: ids up to ties, scores exact (int32 flag) / fp16 (default), all three filter modes; state-dict round trip | as `test_silvertorch.py` |
| T7 | `official` refuses `torch.compile` with a clear error; sync count per op recorded under `set_sync_debug_mode("warn")` | — |

## 6. Integration — option A, exact changes

### 6.1 `pyproject`

```toml
# retrieve/pyproject.toml
[project.optional-dependencies]
official = ["silvertorch"]                       # replaces the removed `cute` extra

# root pyproject.toml (workspace root; sources are inherited by members)
[tool.uv.sources]
silvertorch = { git = "https://github.com/meta-recsys/silvertorch", rev = "<full sha of 21aa35e>" }
[tool.uv]
no-build-isolation-package = ["silvertorch"]     # torch already in the env: uv's two-phase install
[[tool.uv.dependency-metadata]]                  # `uv lock` on the Mac never runs their setup.py
name = "silvertorch"
version = "1.0.0"
requires-dist = ["torch>=2.0"]
```

Verified against the uv docs (build isolation and `dependency-metadata` are documented for exactly
the flash-attn-needs-torch case; git `rev` sources and per-extra sources are supported). README
tested matrix (`README.md:31-43`): Python 3.10–3.11 / torch 2.4–2.6 / CUDA 12.1 or 12.4
("conservative"); Python 3.11–3.12 / torch **2.7–2.10** / CUDA **12.8** ("recommended"); 3.13 /
2.10+ / 12.8 ("experimental"); torch 2.11 + CUDA 13 unsupported (CCCL 3.x vs `at_cuda_detail::cub`).
`setup.py` declares `python_requires>=3.8`, `install_requires=["torch>=2.0"]`, `version="1.0.0"`, and
passes `--expt-extended-lambda --expt-relaxed-constexpr` to nvcc (`setup.py:62-66`). The A100 box
needs `CUDA_HOME` → a 12.x toolkit with `nvcc` on `PATH` (handoff §13 built our extension with
nvcc 12.4 against the cu128 wheel; the README recommends 12.8 — expect a minor-version warning,
not a failure), `ninja`, `setuptools>=61`, `wheel` (add to the dev group), `TORCH_CUDA_ARCH_LIST="8.0"`
to keep the ~30 s build to one arch. Install: `uv sync --extra official`; nothing builds on the Mac.

Rejected: **B (vendor)** — 9.4k lines under Apache-2.0 + NOTICE built via `cpp_extension.load`;
reproducible, but a fork the moment anyone touches it, and "their code" becomes "their code as we
built it". **C (submodule)** — same pin, worse ergonomics, no lock record; the fallback if the git
build through uv misbehaves (`[tool.uv.sources] silvertorch = { path = "third_party/silvertorch" }`).

### 6.2 Harness

Old harness (what runs today): `evaluation/retrieval/cli/evaluate.py:40` choices gain `official`,
lose `cuda`/`cute`; `sweep.py:48` `_TRITON_FILTER_BACKENDS = ("official",)` (filter modules for
official cells stay Triton, the rule that already existed); `SilvertorchAlgo` gets `compile=False`.
Harness-v2 (its WP-1–3): `PATHS`, `bench.latency(mode="graph")` reason field, JSONL `perf[].mode ==
"graph"` → `null` + `reason`. Config backends: `silvertorch: [triton, torch, official]`.

## 7. What gets deleted (after WP-3's gate, in WP-5)

| file | lines | action |
|---|---|---|
| `kernels/silvertorch/cuda/codesigned_probe_score.cu` | 672 | delete |
| `kernels/silvertorch/codesigned_probe_score_cuda.py` | 611 | delete (keep `build_transposed_sigs`, 40 lines, in `bloom_hash.py` for §8 TF-1) |
| `kernels/silvertorch/codesigned_probe_score_cute.py`, `cute/codesigned_probe_score.py`, `cute/__init__.py` | 650 + 763 | delete |
| `tests/parity/test_codesigned_probe_score_cuda.py`, `..._cute.py` | 454 + 527 | delete |
| `layers/silvertorch/main.py` | 538 → ≈ 400 | drop imports 14-27, `_TWO_KERNEL_BACKENDS` 41-44, `bloom_sigs_t` branch 244-271, dispatch 305-308, `_forward_cuda/_forward_cute/_forward_two_kernel` 368-454 |
| `tune.py` | 632 → ≈ 450 | drop imports 37-47, `_cps_cuda_probe_family` 173-198, `_cps_cuda_inputs` 199-239, `_cpse_cuda_inputs` 270-295, specs 442-483, 502-530 |
| `tests/conftest.py` 22-82, `tests/parity/conftest.py` 7,40, `tests/correctness/test_silvertorch.py` 37,42-45,284-360, `test_tune_smoke.py` 21-25,36-43, `tests/compile/test_silvertorch_compile.py` 36-48,53-56, `test_export_kernel_ref.py` | ≈ 160 | `require_cps_*` → `require_official`; cuda/cute rows → official rows |
| `kernels/silvertorch/__init__.py`, `interfaces.py:8`, `evaluation/retrieval/sweep.py:48,145,158`, `cli/evaluate.py:40`, `config/deep_sweeps/arxiv-d128-silvertorch.yaml`, `retrieve/pyproject.toml` `cute` extra | ≈ 20 | as §6.2 |
| `docs/system/kernels.md` §cuda + §cute + Follow-ups (849-1491) | 643 | replace by a 20-line "Historical backends" note + the official-backend section |
| `architecture.md` 20-26, 54-55, 251-262, 289-311, 337-393; `evaluation.md` 371-412; `testing.md` 35-52, 127-145; `README.md:56`; `retrieve/README.md:13,52,66`; `00-roadmap.md` | ≈ 150 | rewrite for `official` |
| `docs/plans/cuda-silvertorch-handoff.md` (822), `cuda-silvertorch-phase2.md` (138), `cute-dsl-scorer.md` (350), `cute-dsl-scorer-artifacts/` (128 files, 2 MB) | 1,310 + artifacts | **move to `docs/plans/archive/`** unchanged |

Net: ≈ 3,680 lines of kernel/host/test code deleted outright, ≈ 550 trimmed from shared files,
≈ 800 lines of live docs replaced by ≈ 120; the `cute` extra and the *runtime* JIT toolchain
requirement (`nvcc` + `ninja` at first forward) go away — the official extension builds once at `uv sync`.

**What stays citable.** Tag the last commit holding both backends (`git tag cuda-cute-backends-final
<WP-5 parent>`) and cite `retrieve@cuda-cute-backends-final`. Results quoted from the archived
plans: handoff §13 — the transposed-index mask reads ~40× fewer filter bytes than the row-wise
Triton form and is 2.1–2.5× faster kernel-only (1.3× wall) *once the scorer had memory-level
parallelism* (the "which SilverTorch" finding, S13); cute §5 — a CuTe DSL port is kernel-for-kernel
within noise of the C++ backend, its cost is launch overhead (63 → 16 µs after trims), results
byte-identical; cute §5.1 — every backend collapses to one latency under CUDA-graph replay.

## 8. Triton kernel follow-ups (where effort goes now)

Each item: expected gain, effort, validation = the parity suite (bit-exact vs `torch` and vs
official T1) plus the §9a/§9b head-to-head; gains are claimed only from the head-to-head.

- **TF-1 — Transposed cluster-major bloom index + 1-bit-mask scorer (phase 2 in Triton). Worth
  writing.** Row-wise bloom reads `W·8 = 128 B` of signature per probed item — at `B=16, P=58 368`
  that is 119 MB, equal to the code rows themselves, which is why Triton bloom costs 117–125 µs vs
  85 µs unfiltered (cute §5 kernel-only). The transposed form reads only the set rows of `QB`:
  `popcount(QB) ≈ k_hash × active clauses` (5–25 bits) × 8 B per 64 items ≈ 0.6–3 B/item, 40–200×
  less (kernels.md §Phase 2); the C++ mask kernel ran in 6–7 µs and the masked scorer in 40–50 µs.
  Triton design: grid `(B, n_probe · wpc)`, one program per 64-item word per query, loop over
  `m_bits` rows in `BLOCK_M` chunks with `tl.load(sigs_t + rows·stride + word, mask=qbit[rows],
  other=-1)` — masked-out lanes issue no transaction, so traffic ∝ set bits without the
  data-dependent `__ffsll` walk Triton cannot express (≈ 8 predicated iterations of overhead) —
  then `tl.reduce(combine_fn=and)` and one int64 store. Scorer: a `HAS_MASK` constexpr path in
  `_codesigned_probe_score_kernel` that reads one mask bit per slot (`p // max_size`, `p % max_size`)
  instead of the `[BLOCK_P, W]` sigs tile; rejected rows already skip the 128 B gather via
  `mask=keep[:, None]` (codesigned_probe_score.py:93-97). Buffers: `bloom_sigs_t` instead of
  `bloom_sigs` (`build_transposed_sigs` kept from the cuda module). Gain: bloom `B=16, P=58k`
  kernel-only 117 → ≈ 55 µs (2×), wall 0.28 → ≈ 0.21 ms. Effort 2 d. Validation: mask `torch.equal`
  to `bloom_subset_match` on the probed items (exact algebra), T4-style ⊇ check vs official, h2h.
- **TF-2 — Salt as a buffer** (harness-v2 §7). `_clause_salt` builds `torch.tensor(_SALT, device=cuda)`
  per call (bloom_hash.py:56-60): a pageable H2D copy per forward that breaks raw CUDA-graph
  capture (cute §5.1) and inflates eager bloom by ≈ 0.4 ms flat (filtering.md). Register seeds and
  salt at `register_index`. Gain: eager bloom at `B=1` toward the `none` row; effort 0.5 h;
  validation `test_bloom_hash.py` bit-equality. Do before WP-7.
  > **Status (roadmap B5):** authored 2026-09-06 on `dev/b1-official-adapter` (Mac-side, no
  > GPU): `generate_clause_salt` → `[C]` int64 `clause_salt` buffer registered by `BloomFilter`
  > and `SilverTorch`, builders take `clause_salt=`; four tests added to `test_bloom_hash.py`
  > (buffer ≡ on-the-fly ≡ pre-B5 inline bits on CUDA and CPU, device independence, shape
  > check, buffer moves with `.to()`). **GPU gate pending** — `test_bloom_hash.py` and the
  > bloom rows of `test_silvertorch.py` on the A100; the raw-capture claim is unmeasured.
- **TF-3 — Memory-level parallelism / retune.** The C++ scorer went 188 → 88 µs by keeping
  `SPW·UNROLL` rows in flight per warp (handoff §13 fix 1). Triton's `[BLOCK_P, D]` int8 tile is
  already one coalesced 128 B row per item with `num_stages` pipelining across programs
  (`DEFAULT_CONFIG = (256, 4, 3)`, codesigned_probe_score.py:24) and sits at 85 µs ≈ 1.4 TB/s, ~90 %
  of A100 HBM; its `tl.dot(q[1, D], codesᵀ)` with `M = 1` cannot use IMMA tensor cores and lowers
  to the dp4a path (codesigned_probe_score.py:99-103), so both arms execute the same instruction
  and the comparison is purely about memory scheduling. Expected ≤ 10 % from `uv run tune-kernels
  codesigned-probe-score --json-out` after TF-1 changes the register budget; effort 1 h.
- **TF-4 — Evict-first row loads.** C++ used `__ldcs`; Triton has `tl.load(..., eviction_policy=
  "evict_first")` on the code gather (rows are touched once per query). Expected 0–5 % at `B=16`
  (40 MB L2 vs 119 MB streamed); effort 15 min; keep only if outside ±1 µs noise.
- **TF-5 — Ids prefetched one iteration ahead** (handoff fix 1b): a Triton program owns a whole
  tile, so its ids are one load and the dependent gather is issued as a whole; the cross-tile
  overlap the C++ loop pipelines by hand is what `num_stages` gives across programs. Nothing to
  write; confirm with `kernel_only.py` that Triton stays within ±5 % of the C++ 90 µs (cute §5).
- **TF-6 — What the official pipelined kernel does, and Triton.** Warp payloads — a flat work list
  `(doc_start, write_index, query_row, 32-bit mask)` per 32-doc chunk built by two prep kernels +
  cumsums — are our `flat_items [B, P]` plus `(cluster, slot)` arithmetic, which the padded layout
  gives for free (cost: `-1` pads at ragged clusters; theirs pays ≈ 12 launches). Double-buffered
  shared-memory queries: moot, a program loads `q_codes [D]` once into registers for its tile.
  `__launch_bounds__(256, 2)`: an occupancy hint, i.e. `num_warps` × register pressure — Triton
  exposes only `num_warps`/`num_stages`. Thread-per-row with 32 rows in flight: not expressible,
  not needed (the tile load does the same via vectorized loads). The kernel is dead code upstream
  anyway. Verdict: nothing to port.
- **TF-7 — fp16 score store** (official writes fp16): 1.9 vs 3.7 MB at `B=16, P=58k`, < 2 % of the
  row traffic, costs bit-exactness. Skip. **TF-8 — Exact mode**: the fused predicate is already the
  winning structure (the C++ standalone clause kernel paid a serial 55 µs gather); TF-1's `HAS_MASK`
  scorer would also accept a `clause_mask` output, giving the S9-style full-mask-vs-fused ablation
  on Triton for exact mode. Optional.

Order: TF-2 (before WP-7), TF-1 (after WP-5, which moves `build_transposed_sigs`), TF-3/TF-4
retune, each followed by the parity suite and a §9a/§9b rerun.

## 9. Perf-comparison plan (Triton vs official; torch eager as floor)

Scripts live in `docs/plans/official-silvertorch-artifacts/` (raw JSON kept, as the cute artifacts
dir). Reused: `wp4/bench_common.py` (add `official` callables: sorted codes, `cluster_offsets`,
`cluster_length`, int8 queries, plan cache), `wp4/h2h.py`, `wp4/kernel_only.py` (`classify` learns
`process_cluster*` → scorer, `generate_*payload*|cumsum|repeat_interleave|arange|fill` → "official
prep", `process_documents*` → mask), `wp4/host_overhead.py` (+ a sync counter). New:
`official_facts.py` (WP-1), `fpr_calibrate.py` (§4.3), `phase2_h2h.py`. `wp6/graphs.py` is not reused (D7).

- **(a) Phase 3, kernel-only + wall**, shared inputs, D=128, B ∈ {1, 16}, P ∈ {1024, 46 720, 58 368}
  (layouts S/B/A), modes none / bloom / exact. Columns `triton`, `official` (fp16), `official-int32`
  (one row: the epilogue cost), `torch` at P=1024. `torch.equal` on int32-path scores asserted before
  every timing, as `h2h.py` does; profiler µs per kernel class + `do_bench` wall incl. the shared `topk`.
- **(b) Phase 2 only**, N = layout A (3.03 M), B ∈ {1, 16}: `bloom_match` (row-wise, full N) vs
  `bloom_index_search_batch` (transposed, full N, packed) vs `_return_partial_response` (probed only)
  at matched FPR and matched memory; `clause_mask` as the forward-index stand-in (S7's GPU baseline);
  the parse row (§4.4); after TF-1, our transposed kernel joins the table.
- **(c) End to end**, harness cells: arXiv-d128 and Goodreads-d128 `filter` (clause + bloom incl.
  Goodreads `c1_lang_reverse` on official bloom) and `quality` (none); `silvertorch × {triton, torch,
  official}`; ks {100, 500, 1000}; bs {1, 8, 16}; seeds {0, 1, 2} on headline sweeps; `n_probe ∈ {24, 32}`
  (24 = the paper's production point). Per cell: recall vs oracle, `jaccard@100` vs the triton row
  (expect ≥ 0.99 on fp16), `pass_rate`/`bloom_fp_rate` for both blooms, `index_mib` (official bloom
  bytes vs `bloom_sigs`), `build_s` (use the CUDA `bloom_index_build`; the CPU one is a single-threaded
  loop), eager latency + QPS for all three, graph for triton/torch, `peak_fwd_mib`.
- **(d) Paper claims tested** (§B.1 numbering): **S9** directly — official full mask → `fused_kmean_ann`
  vs official partial → `_with_partial_masks`, latency + `peak_fwd_mib` vs n_probe (paper: 1.55 →
  0.72 ms, 35.6 → 18.2 MB at probe 32 on 20 M); **S13** — transposed (official) vs row-wise (Triton)
  phase 2; **S8** — FPR and bytes vs `b_multiplier` on real attributes (paper: 6.98 % → 0.067 %
  from 512 to 1024 bits); **S10** — probed fraction; **S6/S16** — per-row vs global scale on the
  int32 path (§4.2); **S12** — no top-k cap. Cannot test: **S7**'s CPU inverted index (cite),
  **S1–S4/S11/S14/S15**; **S5** needs G5's Faiss baselines.
- **(e) Provenance**: harness-v2 §2.1 + `silvertorch` sha, `nvcc --version`, build flags,
  `TORCH_CUDA_ARCH_LIST`; clocks locked (`nvidia-smi -lgc 1410`), 3 windows, `spread > 5 %` flags the
  cell; kernel-only = medians over 20 profiled iterations after 5 warm-ups.

**Expected outcomes and write-up.** (i) *Kernel-only, no filter, large P*: both HBM-bound on
119 MB of codes; Triton at 85 µs (~1.4 TB/s), the official thread-per-row design issues 16 B per
lane across 32 rows — parity within ±20 % is the likely headline ("a 310-line Triton kernel within
X % of the vendor's"); a Triton win is sector efficiency, an official win is 32 rows in flight.
(ii) *Eager wall at B=1 / small P*: official 2–3× slower from ≈ 12 launches + 2 syncs — reported
next to the kernel-only row as host overhead, never as "their kernel is slower". (iii) *Bloom
kernel-only at B=16, P=58k*: official partial mask + masked scorer beats the row-wise fused Triton
kernel by ~2× until TF-1 lands (S13 confirmed with Meta's code); if not, FPR calibration or the
plan decode is the first suspect. (iv) *Quality*: identical candidate sets up to fp16 ties.
Reviewer questions pre-empted: eager-only fairness (D7); same clusters/codes in both arms (D3);
FPR matching (§4.3); fp16 vs fp32 (§4.2); the shipped pipelined kernel is dead code (§1.1) — we
benchmark what the pinned sha runs; parse cost excluded and reported; identical `topk` epilogue.

## 10. Work packages (in this order; deletion gated on WP-3)

- **WP-0 — Pin + build on the A100 (0.5 d, GPU).** §6.1; `uv sync --extra official`; assert
  `torch.ops.st.fused_kmean_ann` exists; run `pytest silvertorch/` from the pinned checkout. Gate:
  build < 5 min, their suite green, sha + nvcc + build log in the artifacts dir; the 7 open
  upstream issues read before pinning.
- **WP-1 — Op facts (0.5 d, GPU).** `official_facts.py`: bit-order probe, syncs per op under
  `set_sync_debug_mode("warn")`, launches per op from `torch.profiler`, a `torch.cuda.CUDAGraph`
  capture attempt per op (expected failure, message recorded), parse µs per query at B=16, the
  `per_embedding_scale` overflow reproduced. Gate: §3 confirmed or corrected here.
- **WP-2 — Adapter + tests (2 d, Mac-authored).** §5.1, `require_official`, T1–T7. Gate: `ruff`
  clean, suite collects and skips on the Mac.
  > **Status (roadmap B1):** authored 2026-09-06 on `dev/b1-official-adapter` (Mac-side, no GPU,
  > official package not installed — A2 runs in parallel). Shipped: `kernels/silvertorch/official.py`
  > (availability probe splitting "missing" from "broken", `OfficialConfig`, CSR + feature +
  > expression mapping, `pack_mask` / `unpack_partial_mask` / `reverse_bits64`, the raw and
  > dequantised scoring wrappers, bloom partial / full search), `Backend` gains `"official"`,
  > `SilverTorch(backend="official", official=OfficialConfig(...))` with the §5.1 buffers and
  > filter matrix, `compile()` / traced-forward refusal (D7), `require_official()`,
  > `tests/parity/test_official.py` T1–T7 (bit-order tests parametrised over both candidate
  > orders, `OFFICIAL_BIT_ORDER = None` until A3 pins it), `"official"` rows in
  > `test_silvertorch.py`. Mac gate green: `ruff check` / `ruff format --check` clean on the
  > library, `pytest --collect-only` collects 683 tests with no errors, the official surface
  > skips on `OfficialMissing`. **GPU gate pending** (WP-3 / roadmap B2): nothing here has run
  > against the real ops. Deviations from §5 found while matching the upstream source: (1)
  > T4's "official partial mask ⊇ `clause_subset_match` … AND and NOT expressions" cannot hold
  > for NOT — a bloom NOT is the complement of a bloom term, so it has no false positives and
  > *may* have false negatives; the test asserts ⊆ for NOT and records the false-negative
  > rate. (2) The `k` passed to `bloom_index_build` is a knob (`OfficialConfig.build_k`,
  > default = the search `k` as the README does; the upstream module builder passes `hash_k`).
  > (3) `m_bits` is optional on the official backend (the width is `b_multiplier`); `k_hash`
  > is validated `≤ 10` (`MAX_K_V2`). (4) The layer applies the `-1` id sentinel to every
  > non-finite slot (`masked_topk`, the `RetrievalModule` contract); the Triton `_impl`s do
  > not, so the bit-exact tests normalise both sides before comparing ids.
- **WP-3 — Parity gate (0.5 d, GPU).** `tests/parity/test_official.py` + `test_silvertorch.py` green,
  T1 `torch.equal` on every regime, T4 FPR at matched memory recorded. **Unblocks deletion.**
- **WP-4 — Kernel head-to-head (1 d, GPU).** §9a/§9b + `fpr_calibrate.py`; JSON + a §12
  validation record appended to this plan.
- **WP-5 — Delete cuda/cute (1 d).** §7 table, tag, archive, docs. Gate: suite green on the A100 and
  collect-only on the Mac; `git grep -il "cute\|codesigned_probe_score_cuda"` hits only
  `docs/plans/archive/`; `ruff` clean.
- **WP-6 — Harness backend (0.5–1 d).** §6.2 on whichever harness is live. Gate: an `official` cell
  runs end to end on Goodreads d128 `c0_genre` clause + bloom with `jaccard_vs_first@100 ≥ 0.99`.
- **WP-7 — TF-2, then campaign cells (≈ 4 h GPU + 0.5 d).** §9c. Gate: no `unstable` cell,
  `bloom_fp_rate` for both blooms, S9 ablation cells present.
- **WP-8 — TF-1 (2 d) + retune (TF-3/4), rerun §9a/§9b.** Gate: parity suite bit-exact, bloom
  `B=16, P=58k` kernel-only within 1.3× of official.
- **WP-9 — Paper section (1 d).** "Official vs reimplementation" table (kernel-only, eager wall,
  quality, memory, FPR), the S9/S13/S8 replications, G1's deviations table updated (official bloom
  hash ≠ ours; official eager-only; the `per_embedding_scale` finding).

≈ 10 focused days plus GPU time (G2 in reproducibility-paper §B.3 budgeted 2–4 days for the
comparison alone; the rest is the backend swap, the deletion and TF-1).

## 11. Risks

- **Build on torch 2.10 + cu128 with a 12.4 nvcc**: `BuildExtension` runs torch's
  `_check_cuda_version` — minor mismatch warns, major raises. Prefer `CUDA_HOME=/usr/local/cuda-12.8`;
  CUDA 13 / torch 2.11 is unsupported upstream (README "Troubleshooting").
- **Host syncs make `official` eager-only** (§3, D7); passing the precomputed `cluster_warp_*`
  tensors removes two of three syncs but not the `repeat_interleave` — not worth it for a reference.
- **fp16 scores** blur top-k boundaries (T2/jaccard quantify it; the int32 path is exact). **Hash
  mismatch** → no bit-comparison of blooms ever; both §4.3 calibration points in every table.
- **`k ≤ 10`** unchecked upstream (our `k_hash = 5`; the FPR sweep must not exceed it). **CPU
  references are not oracles** (§1.1) — every official-side check runs on GPU. D=96 is
  official-supported but not Triton's power-of-two path (torch reference there, as today).
- **Upstream drift**: no tags, 7 open issues; the pin is a sha, bumps are deliberate PRs rerunning WP-3.
  **Deleting before parity** would leave the transposed-index result with no in-tree owner (D9).

## 12. Claims unlocked (reproducibility-paper §B.1)

Directly measurable: **S9** (co-design ablation, both arms official), **S13** (transposed vs
row-wise, official vs Triton — after TF-1, Triton vs Triton), **S8** (FPR/bytes vs width), **S10**
(probed fraction), **S6/S16** (per-row vs global scale, int32 path), **S12** (no top-k cap). **G1**'s
deviations table gains "official bloom hash ≠ ours", "official is eager-only" and the
`per_embedding_scale` overflow; **G2** closes. Still elsewhere: **S5/S7** (Faiss, CPU inverted index; G5/G6).
