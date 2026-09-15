# Official SilverTorch ops as the reference backend — integration, parity, Triton-vs-official

> **Status:** planned 2026-09-05 on `feat/cute-dsl-scorer`. **WP-0 and WP-1 executed 2026-09-06**
> on `dev/a0-a3-deps-official` (roadmap A2/A3): the package is pinned and built, and §3's host-side
> table and §4's numerics are now measured rather than read — see the §13 validation record, which
> corrects §3's sync and launch counts and §1.1's op inventory. **WP-2 (adapter + T1–T7) and
> WP-3 (the parity gate) executed 2026-09-06 on `dev/integration`** (roadmap B1/B2/B5): int32
> path `torch.equal` vs Triton on every regime, 43/43 tests, suite green — see §14. **WP-5 (the
> deletion, roadmap B4) authored 2026-09-06 on `dev/b4-delete-cuda-cute`, GPU suite pending** —
> see §15. **WP-4 (the head-to-head, roadmap B3) executed 2026-09-15** on
> `dev/b3-head-to-head`: kernel-only, phase-2-only and end to end on both datasets, each
> speed statement beside its parity statement — see §16, which corrects §9's expected
> outcomes (i) and (iii). WP-6 onward: nothing implemented.
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

> **Steer 2026-09-15 (user), recorded in every plan it touches.**
> *"We don't care about reproducing old results now, we're improving all code
> and rewriting, then testing and profiling, then running the full evals step
> by step."* The A1 golden baseline stops being a gate and becomes
> information; roadmap **C4 closed** on what the harness proves about itself
> (graph capture 20/20, kill-and-resume, cross-backend parity, the official
> cell end to end), not on equality with the pre-v2 harness. Order of work:
> **code first, then tests and profiling, then the evals one step at a time.**
> Authority: [00-roadmap.md](00-roadmap.md) §1 status block;
> [evaluation-harness-v2.md](evaluation-harness-v2.md) WP-4's amendment block.
>
> **A finding for B3 and F2, from C4's run (2026-09-15, H §12).** `official` vs
> `triton` on `silvertorch` clears the O WP-6 / roadmap gate on **goodreads**
> — `jaccard_vs_first@100` **0.999849** with the plan cache off — but reaches
> only **0.985 / 0.9849 on arxiv**, where the roadmap's clause names goodreads
> only and so does not bite. `score_max_abs_diff` is ~10× *smaller* on arxiv
> while jaccard is lower.
>
> **Answered by B3, 2026-09-15 (§16), and the guess in the original note was
> wrong.** The note above supposed "more near-ties at the top-100 boundary
> *under a looser filter* rather than a numerics problem". It is near-ties, but
> it has nothing to do with filtering and it *is* numerics: the split appears
> identically on the **unfiltered** cells (arxiv 0.9838 vs goodreads 0.9999),
> and the official int32 path is **bit-exact** against Triton in all three
> filter modes on both datasets. The cause is the shipped **fp16 score path
> meeting arxiv's score distribution**: the rank-100/101 gap there is 8.6e-5 on
> a score of ≈0.83, so **95.3 % of arxiv queries have that gap inside one fp16
> ulp**, against 3.1 % of goodreads queries (22× their ulp). Cost in the metric
> that matters — recall@100 against the exact oracle over 10 k queries — is
> **3.1e-4 on arxiv and 4e-6 on goodreads**. §10 WP-9 should report the
> dataset dependence as a property of fp16 resolution against a corpus's score
> distribution, not as a filter effect.
>
> **Answered by B3 (§16.4) — and this reading of it is wrong.** It is not the
> filter: the *unfiltered* cells split the same way (arxiv 0.9838, goodreads
> 0.9999), and on the **int32** score path both datasets are bit-exact (jaccard
> 1.0, `score_max_abs_diff` 0.0) in all three filter modes. The whole deficit is
> the shipped fp16 score path meeting arxiv's score distribution: **95 % of arxiv
> queries have their rank-100/101 score gap inside one fp16 ulp**, against 3 % of
> goodreads ones. Cost in the metric that matters: 3.1e-4 recall@100 on arxiv,
> 4e-6 on goodreads.

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

**Ours**: `SilverTorch` ([main.py](../../retrieve/src/retrieve/modules/silvertorch.py)),
`backend ∈ {triton, torch, cuda, cute}` ([interfaces.py:8](../../retrieve/src/retrieve/interfaces.py)),
phase 1 host-side (main.py:311-319), phases 2+3 in one Triton launch
([codesigned_probe_score.py](../../retrieve/src/retrieve/ops/triton/codesigned_probe_score.py)
310 lines, exact variant 294, [common.py](../../retrieve/src/retrieve/ops/triton/common.py) 110,
[bloom_hash.py](../../retrieve/src/retrieve/indexing/bloom_hash.py) 165), plus the two-kernel
CUDA/CuTe backends this plan removes (§7).

### 1.1 Inventory — Algorithm 1 phases vs the official ops

| phase (paper Alg. 1) | official op | ours | for `backend="official"` |
|---|---|---|---|
| k-means / IVF build | **none** — README "Index build flow" (`fused_kmean_ann.cpp:331-337`): bring your own k-means, sort items by cluster, CSR `cluster_offsets` | `KMeansTorch` + padded layout (main.py:171-210) | ours; needs a **cluster-sorted** table + `cluster_offsets[n_lists+1]` |
| int8 quantization | **none**; op takes int8 embeddings *and int8 queries* (`fused_kmean_ann_cuda.cu:1296`), scale via `divisor_for_int8` (int) or `per_embedding_scale` (fp16 `[N]`, divides) | `quantize_int8_global` + per-row `quantize_int8` ([quantize.py:24-43](../../retrieve/src/retrieve/indexing/quantize.py)) | ours; scale applied host-side (§4.2) |
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
  [cuda-silvertorch-handoff.md §13](archive/cuda-silvertorch-handoff.md) and [cute-dsl-scorer.md §5](archive/cute-dsl-scorer.md).
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

- `retrieve/src/retrieve/ops/official/adapter.py` (new, ≈ 250 lines): `is_available()` /
  `ensure_loaded()` (distinguishes "not installed" from "op missing", as
  `codesigned_probe_score_cuda.py:161-176` did at the time — that wrapper went at B4, tag
  `cuda-cute-backends-final`); `attrs_to_features(attrs_sorted)`; `queries_to_expressions(qa, clause_is_reverse)`;
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
  > check, buffer moves with `.to()`). **GPU gate passed 2026-09-06** (roadmap B5, §14.2:
  > `test_bloom_hash.py` and the bloom rows of `test_silvertorch.py` green on the A100 in the
  > full-suite run 3); the raw-capture claim is still unmeasured — WP-7's.
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
  > skips on `OfficialMissing`. **GPU gate passed 2026-09-06** (WP-3 / roadmap B2, §14: 43/43
  > after 11 test-side fixes, no adapter change). Deviations from §5 found while matching the upstream source: (1)
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

## 13. Validation record — WP-0 and WP-1, 2026-09-06, A100-SXM4-80GB, nvcc 12.4 / torch 2.10.0+cu128, triton 3.6.0

Roadmap steps A2 (WP-0) and A3 (WP-1), on `dev/a0-a3-deps-official`. Script,
raw JSON and logs: [official-silvertorch-artifacts/](official-silvertorch-artifacts/README.md)
(`official_facts.py`, `official_facts.json`, `official_facts.txt`,
`wp0_build.txt`, `wp0_environment.txt`, `wp0_upstream_pytest.txt`).
Everything below is measured on this box; nothing is read off the source.

### 13.1 WP-0 — the pin and the build

`silvertorch @ 21aa35e28b6dd9a91e9ee35efb0857715e86bda7` (the full sha of
`21aa35e`, still `main` HEAD on 2026-09-06). `uv sync --extra official` built
`silvertorch._C` in **2 m 17 s** with `ninja` (10 translation units,
`CUDA_HOME=/usr/local/cuda`, `TORCH_CUDA_ARCH_LIST="8.0"`, `MAX_JOBS=32`);
`torch.ops.st.fused_kmean_ann` exists. **Gate passed** (< 5 min).

Four corrections to §1, §6.1 and §11:

- **§11's "prefer `CUDA_HOME=/usr/local/cuda-12.8`" can be relaxed to 12.x.**
  This box has only 12.4. Torch warns
  (`cpp_extension.py:525 ... minor version mismatch ... (12.8)`) and builds;
  it raises only on a *major* mismatch. Every CUDA test upstream ships passes
  on the resulting extension.
- **Nine `st::` ops, not eleven.** `is_topk.{cpp,cu}` and
  `fresh_index_post_processing.{cpp,cu}` carry `TORCH_LIBRARY_FRAGMENT(st, …)`
  registrations but are **absent from `setup.py`'s `cpu_sources`/`cuda_sources`**,
  so `torch.ops.st.is_topk` and `torch.ops.st.take_top_k_and_gather_from_main_and_fresh`
  exist in no OSS build. §1.1's top-k and live-update rows should say "dead
  source at this sha", alongside the dead `process_cluster_v4_pipelined` and the
  unreachable v1 template branches §1.1 already lists. Nothing in §5 changes.
  `faster_repeat_interleave.cu` compiles but registers no op, consistent with
  §3's note that `fused_kmean_ann_cuda.cu` never calls it.
- **`pytest silvertorch/` — the README's own verification command — fails at
  collection**, 3 errors, before a single test runs. Meta's internal
  `@oss-disable` comment-stripping is line-based and mangled three test files,
  each of which loads a Buck target that cannot resolve outside Meta:
  `test_fresh_index_post_processing.py` and `test_bloom_search_integration.py`
  have the *closing paren* of a multi-line `torch.ops.load_library(...)` left
  commented (`SyntaxError: '(' was never closed`); `test_is_topk.py:24` has a
  single-line call that was not commented at all
  (`OSError: Could not load this library: /silvertorch/oss/ops/csrc:is_topk`).
  Excluding those three: **99 passed, 3 subtests passed in 11.6 s**, every CUDA
  test included. With the bogus lines removed in a scratch copy,
  `test_bloom_search_integration.py` passes in full (9 tests) — the file is
  sound; the other two fail on the ops that are never compiled. **Gate reading:
  green for everything the OSS build ships.** Two paper-ready build-friction
  facts for §9's write-up.
- **"7 open issues" are 7 pull requests.** The repo has zero issues, open or
  closed, in its whole history; the 7 open items (`#5`–`#10`, `#16`) are PRs
  exported from Phabricator. None argues against this pin: `#16` is CCCL-3 /
  CUDA-13 compatibility (the incompatibility we avoid by staying on 12.x),
  `#5`/`#6` a MovieLens benchmark, `#7` lazy imports in internal test rules,
  `#8`/`#9` README wording, `#10` a JAX/TPU bloom path.

§6.1's `pyproject` recipe works as written with two additions: the workspace
root is virtual, so it must re-export the extra
(`official = ["retrieve[official]"]`) for `uv sync --extra official` to resolve
against it; and `no-build-isolation` needs `setuptools`/`wheel`/`ninja` already
in the shared `.venv`, which a member's dev group does not give — they are now a
root `[dependency-groups] dev`. `[[tool.uv.dependency-metadata]]` works: `uv lock`
takes 1 s and never executes upstream's `setup.py`. The `cute` extra **stays**
until B4 (roadmap rule 5); `official` is added alongside it, not in place of it.

### 13.2 WP-1 — measured op facts

Config for every measurement below: `N = 16 384` (64 clusters × 256),
`D = 128`, `n_probe = 8` (`P = 2048`), `B = 16`, `k = 5`, `hash_k = 7`,
`b_multiplier = 8.0`, int8 codes, query plans held on **CPU** as §3 advises.

**Bit order — §4.3 CONFIRMED, three independent ways.** The official bloom is
**HIGH-bit-first**: document `d` lives at bit `63 - (d % 64)` of word `d // 64`.

| probe | result |
|---|---|
| A — packed `bloom_index_search_batch(return_bool_mask=False)` with a predicate matching only doc 0 | word 0 = `0x8000000000000000`, i.e. bit 63 |
| B — hand-built `filtering_bit_mask` into `fused_kmean_ann` | bit 63 set → doc 0 survives; bit 0 set → doc **63** survives |
| C — round trip: A's packed word used as B's mask | surviving ids `==` the bool mask's passing docs |

**This is the constant B1 was told to parameterise over (roadmap A3 → B1 note):
take the HIGH-first branch.** `pack_mask_high_first(bool [B, N]) -> int64
[B, ceil(N/64)]` sets bit `63 - (i % 64)` for item `i`, and T3 pins it.

**Host syncs per op — §3 measured, and it undercounts.** The instrument matters:
`warnings.catch_warnings(record=True)` around
`torch.cuda.set_sync_debug_mode("warn")` reports **zero** syncs for every
`torch.ops.st.*` call, which is an artefact — a `TORCH_WARN` raised inside a C++
custom op is handled by c10's own warning handler and printed to fd 2, never
converted to a Python warning. Capturing fd 2 and counting
`warn_or_error_on_sync` lines (validated against a `t.item()` control visible to
both instruments) gives:

| op | §3 predicted | measured | verdict |
|---|---|---|---|
| `fused_kmean_ann` | 2, unavoidable | **3** | more than predicted |
| `fused_kmean_ann_with_partial_masks` | 1–3 | **4** | above the range |
| `bloom_index_search_batch` | plan `.cpu()` only if plans are on CUDA | **0** | §3's "keep plans on CPU" advice confirmed |
| `..._return_partial_response` | ≥ 1 | **2** | consistent |
| `bloom_index_build` (CUDA) | `[-1].item()` ×2 | **2** | exact |
| `generate_column_info_for_clusters` | none | **0** | exact |

**Kernel launches per op — §3's "≈ 12 launches" for the scorer is low.**
`torch.profiler`, one profiled call after 3 warm-ups:

| op | launches | distinct | D2H | H2D | aten ops |
|---|---|---|---|---|---|
| `fused_kmean_ann` (no filter) | **19** | 16 | 3 | 0 | 58 |
| `fused_kmean_ann` (full mask) | **19** | 16 | 3 | 0 | 58 |
| `fused_kmean_ann_with_partial_masks` | **19** | 16 | 4 | 0 | 62 |
| `bloom_index_search_batch` (packed, full N) | **1** | 1 | 0 | **2** | 4 |
| `..._return_partial_response` | **13** | 13 | 2 | 2 | 43 |
| `generate_column_info_for_clusters` | **1** | 1 | 0 | 0 | 3 |
| `bloom_index_build` (CUDA) | **16** | 10 | 2 | 3 | 47 |

Only 2 of the scorer's 19 launches are the actual `process_cluster` /
`process_cluster_remaining` work; the rest is the payload prep §3 describes
(`generate_cluster_warp_size`, two `generate_*payload*` kernels, four cub scans,
`arange`, a `repeat_interleave` `compute_cuda_kernel`, a scatter-gather, two
fills, a bool reduce). The two H2D copies on `bloom_index_search_batch` are the
per-call pageable plan upload §3 predicts, and they are the whole reason that op
is not free even at 1 launch. **Direction of §3's conclusion holds and gets
worse: a `backend="official"` bloom forward is ≈ 32 launches + ≥ 5 syncs against
Triton's 1 launch.** The §9(ii) framing — report host overhead next to the
kernel-only row, never as "their kernel is slower" — is the right one.

**CUDA-graph capture — §3 corrected, D7 unchanged and stronger.** §3 marks
`generate_column_info_for_clusters` as the one capturable op. Measured, **two**
of seven capture; but the second one is a trap:

| op | capture | note |
|---|---|---|
| `generate_column_info_for_clusters` | **CAPTURED** | as §3 says |
| `bloom_index_search_batch` | **CAPTURED** | …and **replaying it raises `AcceleratorError: CUDA error: an illegal memory access was encountered`** |
| the other five | FAILED | `cudaErrorStreamCaptureInvalidated` — "operation failed due to a previous error during capture" |

The replay probe captures with predicate A (passing docs `[0, 664, 1170, …]`),
overwrites the host plan buffer in place with predicate B (`[1, 2015, 5911, …]`),
and replays: the replay does not return A's stale answer, it faults. The host
plan decode and pageable upload simply are not in the graph. So capture
"succeeding" for this op means *nothing usable*, and **D7 ("`official` runs
eager only", `mode: graph` recorded as `null` with reason `not_capturable`)
stands as written** — with a sharper reason for the harness: `bloom_index_search_batch`
must be on the not-capturable list explicitly, or it will capture and then crash.

Operational fact for anyone extending this: **a failed capture attempt leaves
the CUDA context poisoned** — the next unrelated `torch.cuda.synchronize()`
raises `cudaErrorIllegalAddress`. `official_facts.py` therefore runs each graph
probe in its own subprocess; a single-process sweep reports the first failure
and then junk.

**Parse cost (§4.4) — CPU, `torch.utils.benchmark`, plans returned to CPU:**

| batch of expressions | µs / call | µs / query |
|---|---|---|
| B=16, `c:v AND c:v` | 58.7 | **3.67** |
| B=16, `c:v` | 41.7 | 2.61 |
| B=16, `c:v AND NOT c:v` | 66.8 | 4.18 |
| B=1, `c:v AND c:v` | 12.6 | 12.62 |

IQR ≤ 0.2 µs. At B=16 the parse is ~59 µs against a bloom forward in the
hundreds of µs — a real line item, and the reason §4.4's per-pool plan cache is
worth having. It is reported as its own row and excluded from the kernel
comparison (§9e).

**Scale conventions (§4.2) — all three CONFIRMED.** With saturated codes
(`±127`, D=128) the raw dot is `127² · 128 = 2 064 512`:

- (i) `divisor_for_int8 = -1` returns **int32** and the value is exactly
  2 064 512 — the bit-exact parity path D5 relies on.
- (ii) `divisor_for_int8 = 64` returns **fp16** `32 256.0` where the exact
  quotient is `32 258.0`: pure fp16 rounding (the representable spacing at
  2¹⁵ is 16), relative error 6.2 × 10⁻⁵, inside §4.2's 2⁻¹¹ bound.
- (iii) `per_embedding_scale = ones(N, fp16)` returns **`inf` in every one of
  the 256 output slots**. The kernel casts the int32 dot to fp16 *before*
  dividing (`index_mm_helpers.cuh:50-52`), so the option is unusable for any
  input whose dot exceeds 65 504 — at full-range int8 codes that is **D ≥ 5**,
  exactly as §4.2 predicts. The S6/S16 per-row-vs-global-scale ablation must run
  on path (i) with the per-row scale in the host epilogue.

### 13.3 What this changes, and what it does not

Confirmed as written: §4.3's bit order (the one fact B1 was blocked on), §4.2
(i)–(iii), §3's sync counts for `bloom_index_build` and
`generate_column_info_for_clusters`, §3's "keep plans on CPU", D5's int32
contract, D7's eager-only conclusion.

Corrected: §3's sync counts for both scorer entry points (3 and 4, not 2 and
1–3) and its launch estimate (19, not ≈ 12); §3's capturable column
(`bloom_index_search_batch` captures but its replay faults); §1.1's `is_topk`
and `take_top_k_and_gather_from_main_and_fresh` rows (never compiled);
§1.1/§11's "7 open issues" (7 PRs, 0 issues); §11's CUDA 12.8 preference
(12.4 is fine).

Not done in A2/A3, by scope: no adapter, no parity test, no timing — WP-2/WP-3
own those. The bloom FPR and memory calibration of §4.3 is WP-4's
`fpr_calibrate.py`, not measured here. Numbers in this section are host-side
counts and CPU parse times; **no kernel-speed claim is made and none of this is
citable as a performance result** (roadmap rule 2).

## 14. Validation record — WP-2 GPU gate + WP-3 parity gate, 2026-09-06, A100-SXM4-80GB, nvcc 12.8 / torch 2.10.0+cu128, triton 3.6.0

Roadmap steps **B1** (WP-2's GPU gate), **B2** (WP-3) and **B5** (TF-2's GPU gate), on
`dev/integration` (worktree; parent `563a0f3` = `development`), commits `2b09f8e`
(three pre-existing red cells), `0521a67` (stable argsort), `aadc380` (`test_official.py`
fixes + the per-forward sync test) and the docs commit that carries this section. Raw
outputs: [official-silvertorch-artifacts/wp3/](official-silvertorch-artifacts/README.md)
(`full_suite_run{1,2,3}.txt`, `test_official_run3.txt`, `argsort_stable_probe.{py,txt,json}`,
`parity_gate_probe.{py,txt,json}`) and `wp0_upstream_pytest_cu128.txt`.

**Environment.** A100-SXM4-80GB (sm_80), driver 570.195.03, Python 3.11.10, torch
2.10.0+cu128, triton 3.6.0; `silvertorch._C` **rebuilt with nvcc 12.8** (V12.8.93,
`/usr/local/cuda-12.8` — upstream README's tested row; `wp0_build_cu128.txt`, 169 s, no
warnings) at `21aa35e28b6dd9a91e9ee35efb0857715e86bda7`; our CUDA C++ backend JIT-built
with the same nvcc. **SM clock unlocked** (`nvidia-smi -lgc` is denied on this box;
samples during the session 210–1140 MHz). Nothing in this section is a timing: every
number is a bit comparison or a count, so the clock is irrelevant to it.

### 14.1 Upstream suite on the 12.8 build

Fresh clone at the pin, built `_C.so` copied in, the three uncollectable files of §13.1
excluded: **99 passed, 3 subtests passed in 2.58 s** — the same 99 as the 12.4 build
(`wp0_upstream_pytest_cu128.txt`).

### 14.2 Library suite (`uv run --directory retrieve pytest tests/`)

| run | state | result |
|---|---|---|
| 1, `-x` | tip `563a0f3` | 337 passed, 42 skipped, **stopped at `test_silvertorch.py::TestEdgeCases::test_n_lists_equals_n[cute]`** (2 m 03 s incl. the 78 s CUDA JIT build) |
| 2 | + `2b09f8e`, `0521a67` | 561 passed, **11 failed — all in `test_official.py`**, 127 skipped, 57 s |
| 3, final | + `aadc380` | **574 passed, 0 failed, 127 skipped, 52 s** |

Every skip is a `cute` cell: `nvidia-cutlass-dsl` (the `cute` extra) is not installed in
the B2 venv, so those rows skip exactly as `require_cps_cute` promises; the last full cute
validation stays [cute-dsl-scorer.md §5](archive/cute-dsl-scorer.md) (2026-09-02) and B4 deletes
the backend. The `cuda` rows ran (parity + compile + export).

**The three pre-existing red cells** (A2's finding 7; all red on `development` before B1):

- `test_topk_util.py::test_gather_ids_mapping` and `::test_gather_ids_with_mask_sentinel`
  compared `out_scores.tolist()` with Python literals (`0.9`, `0.7`, `0.1`) while the
  score tensor is fp32 (`0.8999999761581421 != 0.9`): a test that could never pass. Fixed
  as **exact equality** — `torch.equal` against the fp32 input elements the winners were
  gathered from — no tolerance introduced anywhere.
- `test_silvertorch.py::TestEdgeCases::test_n_lists_equals_n[cute]` was the one test in
  the file that builds a module without `_require_backend`, so without the extra it
  raised `CuteMissing` from inside the forward instead of skipping. Gated.

### 14.3 The parity gate — per test (`test_official.py`, 43 tests, run 3 all green)

First run against the real ops: 32 of 43 green as authored; the 11 red were all
**test-side**, none a mismatch of the adapter or of the kernels (`aadc380`). No tolerance
was loosened; T1's `torch.equal` gates are unchanged.

| test | cells | outcome | what B2 changed |
|---|---|---|---|
| **T1** int32 path `torch.equal` vs `ref_cps_phase23` and vs `_codesigned_probe_score_impl`, ids up to ties | 4 layouts (`(16,64,4,D64)`, `(64,96,8,D128)`, `(32,64,8,D96)` ref-only, `(32,90,8,D128)` remainder path) × B ∈ {1, 16} = 8; exact mask via `clause_mask` → `pack_mask` 4 (D ∈ {64, 128} × reverse none/mixed); raw op contract 1 | **PASS — bit-exact on every regime** | exact cells asserted `mask.shape == (b, N)`; `make_probe_family` pads ~10 % of slots so the CSR doc space is `sort_perm.numel()` (5507 of 6144). Assertion corrected; the bit-exactness held as authored |
| **T2** fp16 path | 2 layouts × B ∈ {1, 16} + the divisor bound | **PASS**: `max_rel_err` 4.76e-4 / 4.85e-4 / 4.88e-4 / 4.88e-4 (bound 2⁻¹⁰ = 9.77e-4), `jaccard@32` 1.0 / 1.0 / 1.0 / **0.9924** (D=128, B=16) | — |
| **T3** bit order | packed-output round trip + README hits 1; one-doc mask over a 70-doc cluster, docs {0, 5, 40, 69} 4; partial decode 2; pack/unpack/reverse 1 | **PASS — HIGH-first confirmed** (adapter constants equal the A3 pin) | the negative control claimed a low-first mask scores *nothing*; it scores the **mirrored doc** `63 − d % 64` of the same word (0 → 63, 5 → 58, 40 → 23; 69 → 122, past the cluster → nothing) — A3's probe B verbatim. The control now asserts exactly that |
| **T4** bloom ⊇ exact, FPR; NOT ⊆ exact; expressions | 3 | **PASS**: AND — no false negative on the full mask nor on the partial masks (which equal the full mask on the probed docs), FPR 0.0000 at `b_multiplier=10` on a 0.09 % pass-rate predicate; NOT — no false positive, false-negative rate 0.0000; LRU / `cache=False` behave | — (FPR table in §14.4) |
| **T5** `_with_partial_masks` ≡ full mask ≡ unfiltered ∘ mask | B ∈ {1, 16} | **PASS**, `torch.equal` on scores, ids and valid | — |
| **T6** the layer | int32 vs Triton 3 (none / exact / exact-reverse, + all-inactive); fp16 2; bloom 2 (partial / full, + `cache_plans=False`); contract 1 | **PASS**: int32 `torch.equal` on every path; fp16 `jaccard@64` **1.0000** on `none` and `exact`; bloom **0 false positives among 1024 returned slots** on both paths, partial ≡ full ≡ uncached bit for bit; state-dict order / round trip, candidates through `inv_perm` bit-equal to Triton, bare bloom index, `k_hash > 10` rejected | fp16-exact read `jaccard = 0.09` and the bloom cells crashed on `.min()` of an empty set because the **Triton epilogue leaves the padded item id at `-inf` slots** while the official epilogue writes `-1` (B1's deviation 4). The int32 cells already normalised through `_sentinel_ids`; the fp16 and bloom cells now do too |
| **T7** eager-only; syncs | compile refusal 1; per op 1; per forward × `cache_plans` 2 | **PASS**: `module.compile()` and `fullgraph` raise; syncs per op **parse 0, `fused_kmean_ann` 3, `_return_partial_response` 2, `_with_partial_masks` 4, `bloom_index_search_batch` 0** — §13.2's numbers exactly; per forward see §14.5 | `_count_syncs` used `warnings.catch_warnings`, which reads 0 for every C++ op (§13.2's instrument artefact) — it now counts c10's `warn_or_error_on_sync` lines on fd 2 **plus** Python-side warnings (disjoint: an `.item()` control is seen only by the latter, the ops only by the former). New `test_t7_layer_forward_sync_count` |

### 14.4 T4 — FPR at matched memory (`parity_gate_probe.py`, both blooms `k = 5`, official `hash_k = 7`)

Matched memory is **exact**, not "within 2 %": the official width per bundle is
`int(max_terms · k · b_multiplier)` bits per doc (`bundle_b_offsets` is its running sum,
`bloom_indexer.cpp:46-66`), so at `b_multiplier = m_bits / (max_terms · 5)` the official
index has the same byte count as our `[N, m_bits/64]` signatures — 131 072 / 262 144 /
524 288 B for 256 / 512 / 1024 bits on the T4 corpus (4 terms per doc, `b_multiplier` 12.8 /
25.6 / 51.2) and at `b_multiplier` 6.4 / 12.8 / 25.6 on the dense corpus (8 terms).

T4's corpus (N = 4096, C = 2, A_max = 2, vocab 50, pad 0.3; 32 two-term AND queries, exact
pass rate 0.0009) — 4 096 docs, so the "index B" column is the real buffer size:

| arm | width | bytes/doc | index B | FPR | FN |
|---|---|---|---|---|---|
| official | `b_mult=1.5` (30 bits) | 3.8 | 15 360 | 0.0010 | 0 |
| official | `b_mult=2.0` (40) | 5.0 | 20 480 | 0.0002 | 0 |
| official | `b_mult=3.0` … `10.0` (60 … 200) | 7.5 … 25 | 30 720 … 102 400 | 0.0000 | 0 |
| official | **`b_mult=12.8` (256) — matched to `m_bits=256`** | 32.0 | **131 072** | 0.0000 | 0 |
| official | **`b_mult=25.6` (512) — matched to `m_bits=512`** | 64.0 | **262 144** | 0.0000 | 0 |
| official | **`b_mult=51.2` (1024) — matched to `m_bits=1024`** | 128.0 | **524 288** | 0.0000 | 0 |
| ours | `m_bits=256` | 32.0 | **131 072** | 0.0000 | 0 |
| ours | `m_bits=512` | 64.0 | **262 144** | 0.0000 | 0 |
| ours | `m_bits=1024` | 128.0 | **524 288** | 0.0000 | 0 |

Dense corpus (A_max = 4, vocab 8, pad 0.1, up to 8 terms per doc; 32 one-term queries,
exact pass rate 0.3787):

| arm | width | bytes/doc | index B | FPR | FN |
|---|---|---|---|---|---|
| official | `b_mult=1.5` (60 bits) | 7.5 | 30 720 | 0.0043 | 0 |
| official | `b_mult=2.0` (80) | 10.0 | 40 960 | 0.0067 | 0 |
| official | `b_mult=3.0` … `5.0` (120 … 200) | 15 … 25 | 61 440 … 102 400 | 0.0000 | 0 |
| official | **`b_mult=6.4` (256) — matched to `m_bits=256`** | 32.0 | **131 072** | 0.0000 | 0 |
| official | **`b_mult=12.8` (512) — matched to `m_bits=512`** | 64.0 | **262 144** | 0.0000 | 0 |
| official | **`b_mult=25.6` (1024) — matched to `m_bits=1024`** | 128.0 | **524 288** | 0.0000 | 0 |
| official | `b_mult=40.0`, `51.2` (1600, 2048) | 200, 256 | 819 200, 1 048 576 | 0.0000 | 0 |
| ours | `m_bits=256` / `512` / `1024` | 32 / 64 / 128 | 131 072 / 262 144 / 524 288 | 0.0000 | 0 |

Reading: at matched memory **both blooms are at FPR 0.0000 on synthetic attributes** —
with ≤ 8 terms × 5 hashes per doc a 256-bit row is ≤ 16 % full and a term's 5 positions
collide with probability ≈ 10⁻⁴; the official bloom shows a measurable FPR only below
~100 bits per doc. The gate's requirement (FPR recorded at matched memory, no false
negatives on either arm) is met; the *informative* comparison — the paper's S8 point,
6.98 % → 0.067 % from 512 to 1024 bits — needs real attributes with tens of terms per
doc and is WP-4's `fpr_calibrate.py` / roadmap D3, unchanged.

### 14.5 T7 — launches, memcpys and host syncs per layer forward (`parity_gate_probe.py`)

N = 4096, D = 128, B = 16, K = 64, `n_lists = 64`, `n_probe = 8`; one profiled forward
after 3 warm-ups (`torch.profiler`), syncs by the two disjoint instruments (C++ + Python):

| forward | launches | distinct | D2H | H2D | syncs |
|---|---|---|---|---|---|
| triton none / exact / bloom | 17 / 18 / 39 | 15 / 15 / 31 | 0 | 0 | 0 |
| torch none / exact / bloom | 32 / 41 / 59 | 27 / 34 / 46 | 0 | 0 | 0 |
| official none (fp16 / int32) | 55 / 54 | 42 / 41 | 3 | 0 | **3** (3 + 0) |
| official exact (fp16 / int32) | 63 / 62 | 48 / 47 | 3 | 0 | **3** (3 + 0) |
| official bloom[partial] (fp16 / int32) | 70 / 69 | 44 / 43 | 7 | 2 | **7** (6 + 1) |
| official bloom[full] (fp16 / int32) | 56 / 55 | 43 / 42 | 4 | 2 | **4** (3 + 1) |

`cache_plans=True` and `=False` give **identical rows** in every column — the plan cache
moves only CPU parse time (§13.2: ≈ 59 µs at B = 16), never a launch or a device sync.
The `+ 1` Python-side sync on both bloom paths is the `.tolist()` in
`queries_to_expressions` (the string parser needs host values); the two H2D copies are the
per-call pageable plan upload of §3. The Triton rows count the whole forward — phase 1
(`matmul`, `topk`, gather), `quantize_int8`, the fused kernel and the `topk` epilogue —
the "1 launch" of §3 is the fused kernel alone. `test_t7_layer_forward_sync_count`
asserts the sync column (3 / 3 / 7 / 4) on both cache settings.

### 14.6 `argsort(stable=True)` in `_build_ivf` — applied (`0521a67`)

The review's deferred A3(a). `argsort_stable_probe.py` builds the IVF both ways from one
k-means assignment on nine regimes (N = 256 … 131 072 with `n_lists` 8 … 1024 — the
suite's layer sizes, both sides of torch's 4096-element CUDA small-sort threshold) and
compares the permutation, the padded layout and the forward's ids and scores on `torch`,
`triton` and `official`: **bit-identical in every regime** (the unstable sort was already
id-ordered within clusters on this torch), so no checkpoint or golden output can move.
Applied so the slot order is a property of the assignment, not of the sort implementation.

### 14.7 Findings to carry, and what is not done here

- **Triton's epilogue leaves the padded item id at `-inf` slots** (`_cps_finish`,
  `_cpse_finish`); the `torch` and `official` backends return the `-1` sentinel through
  `masked_topk`, which is what `interfaces.py`'s contract names. Every test normalises
  through score finiteness, so nothing is red, but the two backends' `ids` tensors are
  not `torch.equal` on rows with fewer than K survivors. Not changed in B2: it alters the
  Triton backend's outputs, which A1's golden JSONs and C4's gate own. Recommended: apply
  the sentinel in the two `_finish` helpers together with C1's harness rewrite (one
  `torch.where(isfinite)` — capture-safe), and re-run the golden gate.
  **Landed** on `dev/c4-library-fixes` as `8df7e9a` (2026-09-06): exactly that one op in both
  helpers, gated by `TestFewSurvivorsSentinel` in `tests/correctness/test_silvertorch.py`; T6's
  normalisation stays in place and is now a no-op. The golden re-run is still owed — see the
  C4 record in [evaluation-harness-v2.md §9](evaluation-harness-v2.md).
- The `cute` extra is not in the B2 venv; its 127 cells skipped. B4 deletes them.
- **B4 is unblocked** by the roadmap's rule ("never delete before B2 is green"): T1 is
  `torch.equal` on every regime, the bloom ⊇ / ⊆ checks hold, FPR at matched memory is
  recorded. B3 (the head-to-head) is the other consumer of this gate.
- No timing was taken and none is citable (rule 2; the clock is unlocked anyway). WP-4
  owns the numbers.

## 15. Validation record — WP-5 (roadmap B4), 2026-09-06

§15.1–15.5 are the authoring record (CPU-only, GPU suite pending); **§15.6 is the
A100 gate run, 2026-09-06, green**.

Roadmap step **B4** plus the library review's item 5 (`Backend` reshaping, sequenced for B4),
on `dev/b4-delete-cuda-cute` off `development` at `41d4479`, three commits: `4d92432`
(deletion), `010681d` (`Backend` split + table dispatch) and the docs commit carrying this
section. **Nothing here ran on a GPU**: the box is CPU-only (`CUDA_VISIBLE_DEVICES=""`), so
the A100 suite is the coordinator's and this step's checkbox stays open until it is green.

**Gate check before deleting.** §14.3 T1: official int32 path `torch.equal` vs the
reference and vs Triton on all 8 regime cells + 4 exact-mask cells; §14.7 "B4 is unblocked";
roadmap B2 checked (`aadc380`). Rule 5 of CLAUDE.md is satisfied.

### 15.1 Tag

`git tag cuda-cute-backends-final 41d4479` (local, not pushed) — the last commit holding
both backends; cite `retrieve@cuda-cute-backends-final` for anything quoted from the
archived plans (§7 "What stays citable").

### 15.2 Deleted (`4d92432`)

| file | lines |
|---|---|
| `kernels/silvertorch/cuda/codesigned_probe_score.cu` | 672 |
| `kernels/silvertorch/codesigned_probe_score_cuda.py` | 611 |
| `kernels/silvertorch/codesigned_probe_score_cute.py` | 650 |
| `kernels/silvertorch/cute/codesigned_probe_score.py` | 763 |
| `kernels/silvertorch/cute/__init__.py` | 0 |
| `tests/parity/test_codesigned_probe_score_cuda.py` | 454 |
| `tests/parity/test_codesigned_probe_score_cute.py` | 527 |
| **deleted outright** | **3,677** |

Trimmed (parent → now): `layers/silvertorch/main.py` 781 → 639 (imports, `_TWO_KERNEL_BACKENDS`,
the `bloom_sigs_t` branch and its slack warning, `_forward_cuda` / `_forward_cute` /
`_forward_two_kernel`); `tune.py` 643 → 452 (four specs, `_cps_cuda_probe_family`,
`_cps_cuda_inputs`, `_cpse_cuda_inputs`, `_CPS_CUDA_GRID`); `tests/conftest.py` 247 → 182
(`require_cps_cuda`, `require_cps_cute`); `tests/parity/conftest.py` 195 → 193 (`sigs_t`);
`test_silvertorch.py` 703 → 599 (eight cuda/cute cross-backend tests, the cute-vs-cuda
bit-exact test, `BACKENDS` → three); `test_silvertorch_compile.py` 119 → 97 (twelve rows);
`test_export_kernel_ref.py` 169 → 110 (two backends); `test_tune_smoke.py` 50 → 38;
`kernels/silvertorch/__init__.py` 29 → 5 (re-exports dropped, review A9). `retrieve/pyproject.toml`:
the `cute` extra and the `*.cu` wheel-artifact rule; `uv.lock` re-resolved — nine transitive
packages gone (`nvidia-cutlass-dsl*`, `cuda-python`, `cuda-bindings`, `cuda-core`,
`nvidia-cuda-nvdisasm`, `backports-strenum`), **no torch / triton line changed**. Commit stat:
169 files, +320 / −5,460 (131 of the files are the artifact renames).

**Moved, not deleted.** `build_transposed_sigs` and `words_per_cluster` (§7: "keep …
in `bloom_hash.py` for §8 TF-1"; §8: "WP-5 moves `build_transposed_sigs`") now live in
`layers/filters/bloom_hash.py` with a docstring saying no shipped backend reads the layout;
their layout test `test_build_transposed_sigs_bits` moved to `test_bloom_hash.py` (it needs only
`make_probe_family`). `make_probe_family` stays (`test_official.py` builds its CSR view from it);
`make_bloom` stays minus its `sigs_t` half, with no live caller (kept for TF-1's parity tests,
documented as such in testing.md). Nothing else in the Triton or official path imported from
the deleted modules.

**Plans archived** (`git mv`, contents unchanged): `cuda-silvertorch-handoff.md` (822 lines),
`cuda-silvertorch-phase2.md` (138), `cute-dsl-scorer.md` (350), `cute-dsl-scorer-artifacts/`
(128 files, 9.6 MB on disk) → `docs/plans/archive/`, three rows added to the archive README;
every live link re-pointed (roadmap, this plan, harness-v2 plan, refactor handoff, the
reproducibility paper's path text, CLAUDE.md's model-record pointer). `scripts/check_doc_links.py`
now skips `docs/plans/archive/`: the archived plans reference the deleted sources, and
CLAUDE.md / the archive README already declare the archive "not maintained, links may rot".

**Docs (same commit, rule 4).** kernels.md 1,621 → 973: the cuda and cute sections and their
follow-ups replaced by a 25-line "Historical backends" note (what they were, what they
established, where the record is, that TF-1 inherits the transposed index); the intro,
conventions, op count (16 → 10 ops / 7 files) and tuner list (11 → 7) rewritten.
architecture.md 514 → 486 (backend table, SilverTorch section, state-dict portability, kernels
table); testing.md 682 → 646 (layout, gates, parity helpers, per-file invariants); filtering.md,
evaluation.md, `retrieve/docs/modules.md`, both READMEs, `evaluation/golden/README.md`,
CLAUDE.md rule 5.

### 15.3 `Backend` split and table dispatch (`010681d`, review item 5 / A2 / A9)

`interfaces.py`: `LinrBackend = Literal["torch", "triton"]`, `SilverTorchBackend =
Literal["torch", "triton", "official"]`, `check_backend(backend, literal)`; the five-valued
`Backend` alias is gone from `retrieve.__all__`. Every LiNR layer (`_PackedBitsKNN` covers
`OneBitKNN` / `SimHashKNN`), `PostfilterKNN`, `PostfilterKNNInt8`, `PrefilterKNN`,
`ExactAttributeFilter`, `BloomFilter` and `SilverTorch` validate in `__init__` — an unknown
value, or `"official"` on a LiNR module, raises `ValueError("unknown backend …")` instead of
silently running the torch path (18 new cells in `test_linr.py`). `SilverTorch.forward` calls
`self._forward_impl`, a `{"triton", "torch", "official"} → bound method` table built once in
`__init__`; the `is_compiling()` refusal for official stays in `forward`.
`_register_official_filter_buffers` folded into `_register_filter_buffers(…, perm=sort_perm)`:
the bloom branch keys on the backend (our sigs vs the official index), the exact branch is one
body with the `[perm]` permutation when given — buffer registration order per backend unchanged.

Harness: `algos.py` annotations and its `get_args` guard use the new literals, and its
docstring says the LiNR cells for `official` are `None` because the layers *reject* the value;
`test_algos.py` reads the architecture.md dispatch table's `official` column as "rejected" for
the LiNR rows (the table now says `raises ValueError` there).

**Bit-identity evidence (CPU).** `torch`-backend `SilverTorch` on `none` / `bloom` / `exact`
(same seed, same inputs; forward with attrs and the candidates path) built on the parent tree
and on `010681d`: every `state_dict` key and buffer, and every output tensor, `torch.equal`
(`cpu_identity.py` / `cpu_compare.py`, not kept — 40 lines, reproducible from the description).
`copy.deepcopy` of a module rebinds `_forward_impl` to the copy (`TestStateDict` deep-copies);
`load_state_dict` round trip forwards identically. The Triton and official paths are untouched
by this commit (dispatch only) and are the GPU run's to confirm.

### 15.4 Gates run (all CPU, `CUDA_VISIBLE_DEVICES=""`, `/venvs/integration`)

| gate | result |
|---|---|
| `uvx ruff@0.15.6 check retrieve evaluation/retrieval` | clean (both commits) |
| `ruff format --check retrieve/src retrieve/tests` | clean on every touched file; the three pre-existing unformatted files (`bloom_compact.py`, `clause_compact.py`, and `test_linr.py` before `010681d` reformatted it) are untouched by the deletion |
| `pytest tests/ --collect-only -q` (library) | 486 after `4d92432` (from 701 = 574 + 127 on the parent), 504 after `010681d` (+18 rejection cells); no collection errors |
| `pytest retrieval/tests/ -q` (evaluation, CPU) | 93 passed, 1 skipped — after each commit |
| `python3 scripts/check_doc_links.py` | 0 broken (after each commit) |
| `uv lock` | resolved in 104 ms; diff = the nine `cute` transitive packages only |

**The grep rule.** The roadmap's gate is `git grep -il "cute\|codesigned_probe_score_cuda"`
hitting only `docs/plans/archive/`. Read literally it is unsatisfiable on any tree: `-i` without
`-w` matches "exe**cute**" in `LICENSE`, both frozen `articles/`, `CLAUDE.md` and the harness
docstrings, and the archived plan's own filename (`cute-dsl-scorer.md`) matches wherever the
archive is linked from — which this record, the roadmap's Done entry and the archive README must
do. The reading applied: **no code, test, harness, pyproject, script or live system / sdist doc
mentions the backends.** `git grep -ilw "cute\|codesigned_probe_score_cuda" -- retrieve evaluation
scripts pyproject.toml docs/system retrieve/docs README.md` hits nothing except the mandated tag
name `cuda-cute-backends-final` (kernels.md's historical note; also CLAUDE.md rule 5). The plan
documents that still name them are records, left as written: `00-roadmap.md` (the B2/B4 entries
and this step's Done lines), this plan (§7, §14, §15), the library review (its own text),
`evaluation-harness-v2.md` (its amendment "read every cuda / cute as official"),
`reproducibility-paper.md`, `official-silvertorch-artifacts/` and `evaluation-harness-v2-artifacts/`
(raw outputs), `refactor-validation-handoff.md` (one link to the archived artifacts).

### 15.5 What the A100 run must confirm (the checkbox waits for it)

- `uv run --directory retrieve pytest tests/` green on the 504 collected: expected 0 skips
  with the `official` extra installed (the 127 `cute` skips of §14.2 are gone), otherwise the
  official cells skip via `require_official`.
- `test_silvertorch.py` on every backend × filter mode, in particular `TestStateDict` (deep copy
  of a module carrying the `_forward_impl` bound method) and `TestCrossBackend`'s official rows
  (the merged `_register_filter_buffers`: state-dict key order and the cluster-sorted attrs).
- `test_official.py` 43/43 as in §14.3 — T6's state-dict key-order assertion is the direct
  check on the merged registration.
- `test_silvertorch_compile.py` (3 rows): compiled == eager and **zero graph breaks** with the
  forward now calling `self._forward_impl` — dynamo through a bound-method attribute is the one
  thing in this step a CPU cannot vouch for.
- `test_export_kernel_ref.py` (Triton only now), `test_bloom_hash.py::test_build_transposed_sigs_bits`
  (moved), `test_tune_smoke.py` (7 specs), the 18 `test_unknown_backend_is_rejected` cells.
- No golden re-run is needed: no Triton kernel or epilogue changed (§14.7's sentinel note is
  untouched, as instructed), and the harness's `PATHS` table is unchanged in shape.

### 15.6 A100 gate run — 2026-09-06, green

The GPU half of B4's gate, on `dev/b4-delete-cuda-cute` at `a435179` (the three authored
commits `4d92432` / `010681d` / `6698b4a`, plus one test fix found by this run). Box: the
A100-SXM4-80GB VM, `torch 2.10.0+cu128`, `triton 3.6.0`, `nvcc 12.8` (V12.8.93 — the
`official` extra is built with `CUDA_HOME=/usr/local/cuda-12.8`), Python 3.11. Environment:
a dedicated `/venvs/b4` built from this worktree with `uv sync --extra official`
(`silvertorch==1.0.0` from `meta-recsys/silvertorch@21aa35e`); `retrieve.__file__` verified to
resolve inside `/workspace/wt/b4-delete-cuda-cute/`. **SM clocks cannot be locked in this
container** (`nvidia-smi -lgc` denied, no sudo; sampled 210 MHz idle against a 1410 MHz max),
so the wall times below are run metadata only — **nothing here is a citable timing**.

#### 15.6.1 Library suite (`uv run --directory retrieve pytest tests/ -q`)

| run | state | result |
|---|---|---|
| 1 | `6698b4a` (as authored) | 501 passed, **3 failed**, 0 skipped, 39.4 s (44.6 s wall) |
| 2, final | `a435179` | **504 passed, 0 failed, 0 skipped, 37.8 s** (42.2 s wall) |

**The three red cells (run 1), and the fix.** `test_linr.py::test_unknown_backend_is_rejected[
{official,cuda,foo}-simhash]` — the `simhash` row of §15.3's new rejection cells built
`SimHashKNN(k=K, backend=backend)`, but `SimHashKNN.__init__` takes `k_bits` as a required
positional, so the call raised `TypeError: SimHashKNN.__init__() missing 1 required positional
argument: 'k_bits'` inside the `pytest.raises(ValueError)` block instead of the rejection the
cell asserts. A test-authoring bug invisible to §15.4's `--collect-only` gate (parametrized
lambdas are not called at collection). Fixed in `a435179` by passing `k_bits=64`; the
`ValueError` still comes from `_PackedBitsKNN.__init__`'s `check_backend`, before `k_bits` is
stored, so the cell tests what it claims. **No tolerance was loosened and no test was skipped
or deleted**; no library source changed for it.

**Zero skips**, as §15.5 required: the 127 `cute` skips of §14.2 are gone with the backend, and
with the `official` extra installed no `require_official` cell skipped either — the official
rows genuinely ran. §15.5's list confirmed green in run 2: `test_official.py` 43/43,
`test_silvertorch.py` on all three backends × filter modes (incl. `TestStateDict`'s deep copy
of a module carrying the `_forward_impl` bound method and `TestCrossBackend`'s official rows),
`test_silvertorch_compile.py` 3 rows compiled == eager with zero graph breaks (dynamo through
the bound-method attribute — the one thing CPU could not vouch for), `test_export_kernel_ref.py`,
`test_bloom_hash.py::test_build_transposed_sigs_bits` (moved), `test_tune_smoke.py` 7 specs,
and the 18 rejection cells.

#### 15.6.2 The other gates

| gate | result |
|---|---|
| `pytest retrieval/tests/ --ignore=…test_silvertorch_algo_reverse.py -q` (evaluation, `CUDA_VISIBLE_DEVICES=""`) | **93 passed, 1 skipped, 78.7 s** — the skip is `test_algos.py:315` "official backend integrated in retrieve — its cell is C4's gate", unchanged from §15.4 and not B4's |
| `uvx ruff@0.15.6 check retrieve evaluation/retrieval` | see below |
| `uvx ruff@0.15.6 format --check retrieve/tests/correctness/test_linr.py` | clean |
| `python3 scripts/check_doc_links.py` | **0 broken** |

`ruff` is not in the `b4` venv (it is not a runtime dependency), so the pinned
`uvx ruff@0.15.6` of `.pre-commit-config.yaml` was used. `check retrieve evaluation/retrieval`
and `format --check` on the one file this run changed are clean. CLAUDE.md's wider
`ruff check retrieve evaluation` reports **11 pre-existing `E501`s** in
`evaluation/eval_datasets/{arxiv,goodreads,hf_io,synth_arxiv,timesplit}.py` and
`evaluation/training/train_sasrec.py`, and `ruff format --check retrieve` the two
pre-existing unformatted files of §15.4 (`kernels/filters/{bloom_compact,clause_compact}.py`).
All eight files are **byte-identical to `41d4479`** (`git diff --stat 41d4479..HEAD` over them is
empty): red on `development` before B4, untouched by it, not this step's to fix.

#### 15.6.3 The grep gate

`git grep -il "cute\|codesigned_probe_score_cuda"` outside `docs/plans/archive/`, read as
§15.4 prescribes (**no code, test, harness, pyproject, script or live system / sdist doc
mentions the backends**):

- `git grep -ilw … -- retrieve evaluation scripts pyproject.toml uv.lock docs/system README.md`
  hits **exactly one file**, `docs/system/kernels.md:973`, and it is the mandated tag name
  ("backends is tagged `cuda-cute-backends-final`"). Nothing else in code, tests, the harness,
  any `pyproject.toml`, `uv.lock` or `scripts/`.
- The remaining non-archive hits of the broad `-il` form are the substring "exe**cute**"
  (`docs/system/testing.md:551`, `evaluation/golden/README.md:6`,
  `evaluation/retrieval/run.py:3`, `CLAUDE.md`, `LICENSE`, both frozen `articles/`, six
  `docs/presentation/` binaries) and the plan documents §15.4 lists as records
  (`00-roadmap.md`, this plan, the library review, `evaluation-harness-v2*`,
  `reproducibility-paper.md`, `refactor-validation-handoff.md`, `kernels-layers-design.md`,
  `torch-export-refactor.md`, `official-silvertorch-artifacts/wp3/full_suite_run*.txt`).

**Gate status: green.** B4's checkbox is the coordinator's to flip after merge.

## 16. Validation record — WP-4, the Triton vs official head-to-head (roadmap B3), 2026-09-15, A100-SXM4-80GB

Roadmap step **B3** (§9a kernel-only, §9b phase 2, §9c end to end), executed on
`dev/b3-head-to-head` off `development` @ `e23309c`. Scripts, raw JSON, harness
records and the full table dump:
[official-silvertorch-artifacts/b3/](official-silvertorch-artifacts/b3/)
(`b3_e2e_run.sh`, `b3_kernel_h2h.py`, `b3_tables.py`, `kernel_{goodreads,arxiv}.json`,
`e2e/**/*.jsonl`, `tables.md`). Everything below was measured on this box in this
session. **Nothing here is citable until the orchestrator re-runs the gate**
(CLAUDE.md rule 2); what follows is what the box did, with its estimator and its
spread.

### 16.1 Environment, protocol, and what makes the arms comparable

A100-SXM4-80GB, torch 2.10.0+cu128, triton 3.6.0, CUDA runtime 12.8 / **nvcc 12.4**
build of `silvertorch @ 21aa35e28b6dd9a91e9ee35efb0857715e86bda7`, Python 3.11,
`/venvs/b3`, `code_version 0e67780…` (the library subtree tree hash; `dirty: false`
on all 30 records). Datasets at d128: goodreads-work-id (797,084 items, sweep
`c0_genre`) and arxiv-papers (2,988,996 items, sweep `c0_maincat`), `users_limit
10000`, seed 0, `n_lists 1024`, `n_probe ∈ {24, 32}` end to end and 24 kernel-only,
bloom `m_bits 1024, k_hash 5`, official `b_multiplier 10.0, hash_k 7,
bloom_path="partial"`.

- **Estimator.** Both tiers use the harness's `bench.measure.latency` (H §2.5): 50
  warm-ups, 20 calls to size the window, then **3 windows** of `clamp(2 s / median,
  1000, 5000)` calls, per-call CUDA events, and the **median of the three window
  medians**; `spread = (max − min) / median` of those three, `unstable` above 5 %.
  Every arm rotates the *same* fixed-seed pool of query batches. Host-side rows
  (§16.3's two parser rows) are `perf_counter` walls, because CUDA events would time
  an empty GPU timeline; they say so in the table.
- **Clocks.** `nvidia-smi -lgc` is denied on this box, so every table reports the SM
  clock **sampled under load right after the last window's sync** (`sm_mhz`).
  Under load the box runs at **1410 MHz**; the rows that read 1155–1395 MHz are not
  noise and not idle-sample contamination — they are arms whose GPU is *idle inside
  the measured call* (official at `bs = 1`, where the host does a `.tolist()`
  sync and a CPU expression parse per forward), so the card drops its clock. Read
  those rows as host-bound by construction.
- **`bs = 1` is noise-dominated here** (this session's C4 finding): **31 of the 468**
  end-to-end perf entries are `unstable`, **26 of them at `bs ∈ {1, 8}`**, spreads to
  22.5 %. No claim below rests on a `bs = 1` difference smaller than that; the
  headline comparisons are at `bs = 16`, where 5 of 156 entries exceed the 5 %
  threshold (worst 19.1 %, an arxiv `clause` `triton` `k = 500` cell; the worst
  `k = 100` one is 8.1 %, arxiv bloom official `n_probe = 24`).
- **Plan cache.** Every timed official forward runs with
  `OfficialConfig(cache_plans=False)` — the harness sets it in `run.perf` and
  `b3_kernel_h2h.py` sets it on both official arms — so the CPU expression parse is
  inside every timed call, in both tiers. `cache_plans: false` is recorded on **all
  156 official perf entries**.
- **Fairness (D3), checked rather than assumed.** For every mode and dataset the
  script asserts the two arms share an index: `centroids_equal` **True**,
  `item_codes` of the official arm `torch.equal` to `item_codes[sort_perm]` of the
  Triton arm **True**, `global_scale` equal **True** (6/6 checks, both datasets ×
  3 filter modes). Both arms therefore probe the same clusters and score the same
  int8 codes; only phases 2+3 differ.
- **What is inside a timed call.** *Kernel-only (§16.2)*: phase 1 (centroid matmul +
  probe top-k, identical code in both arms) is precomputed per pool batch and
  excluded; the call covers query quantization, the mask op(s) the filter mode
  needs, the scorer, the host epilogue and the shared `masked_topk`. *End to end
  (§16.5)*: the whole `module.forward`, phase 1 included, through the harness.

### 16.2 Kernel-only — Algorithm 1 phases 2+3 (§9a), `bs = 16`, n_probe 24

`device µs` columns are one `torch.profiler` call classified by kernel name
(`scorer` = `process_cluster*` / `_codesigned_probe_score*`, `mask` =
`process_documents*`, `topk` = the shared epilogue, `prep` = the official op's
payload build — scans, `repeat_interleave`, fills, gathers). **The classifier is a
name heuristic**; the raw per-kernel lists are in the JSON, and one row is
mislabelled on purpose below.

| dataset | mode | arm | wall median ms | spread | sm_mhz | device µs | scorer | mask | prep | topk | launches | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goodreads | none | `triton` | **0.7517** | 0.000 | 1410 | 721 | 337.0 | — | 41 | 335 | 38 | 37.7 |
| goodreads | none | `official-fp16` | 1.3074 | 0.006 | 1410 | 1146 | **30.9** | — | 710 | 338 | 75 | 214.6 |
| goodreads | none | `official-int32` | 1.3750 | 0.003 | 1410 | 1203 | 33.1 | — | 771 | 335 | 73 | 233.3 |
| goodreads | bloom | `triton` | **1.0069** | 0.000 | 1410 | 960 | 520.8 | — | 88 | 339 | 58 | 37.7 |
| goodreads | bloom | `official-fp16` | 2.1563 | 0.070 | 1410 | 1181 | **29.2** | 9.1 | 730 | 331 | 93 | 214.7 |
| goodreads | exact | `triton` | **0.8177** | 0.000 | 1410 | 778 | 397.6 | — | 37 | 339 | 35 | 37.7 |
| goodreads | exact | `official-fp16` | 3.0496 | 0.001 | 1410 | 1343 | 26.8 | — | 834 | 324 | 79 | 228.3 |
| arxiv | none | `triton` | **0.4977** | 0.005 | 1410 | 302 | 129.8 | — | 40 | 124 | 38 | 10.8 |
| arxiv | none | `official-fp16` | 0.9362 | 0.006 | 1410 | 529 | **113.3** | — | 268 | 120 | 75 | 60.9 |
| arxiv | bloom | `triton` | **0.8953** | 0.009 | 1410 | 452 | 228.6 | — | 86 | 124 | 58 | 10.8 |
| arxiv | bloom | `official-fp16` | 1.6064 | 0.001 | 1395 | 543 | **76.8** | 10.4 | 294 | 120 | 93 | 61.0 |
| arxiv | exact | `triton` | **0.4770** | 0.003 | 1410 | 423 | 260.8 | — | 33 | 125 | 34 | 10.8 |
| arxiv | exact | `official-fp16` | 7.2117 | 0.001 | 1410 | 815 | 81.7 | — | 271 | 119 | 77 | **826.7** |

`bs = 1` rows are in the JSON: Triton 0.380–0.869 ms against official 0.833–1.467 ms,
i.e. **1.4–2.5× for host reasons**, exactly the §9(ii) framing, and the arms where
the card drops below 1410 MHz.

Four things this says, and one it does not.

1. **Wall, phases 2+3, `bs = 16`: Triton is 1.7–2.1× faster unfiltered and bloom, and
   3.7× (goodreads) / 15.1× (arxiv) faster in exact mode.** The exact-mode gap is *not*
   a statement about Meta's kernel — see point 4.
2. **Meta's scoring kernel is faster than ours in every cell**, by **1.15–3.2× on
   arxiv** (113 vs 130 µs unfiltered, 77 vs 229 µs bloom) and by **10.9–17.8× on
   goodreads** (31 vs 337 µs unfiltered, 29 vs 521 µs bloom). This
   was not the expected outcome (§9 expected "parity within ±20 %") and the cause is
   **our padded IVF layout, not bandwidth**: `max_tensor_size_per_row = n_probe ×
   max_cluster_size` is 611,520 slots on goodreads where the 24 probed clusters hold
   ≈ 18.7 k items — **97 % of the slots our kernel walks are `-1` pads** — against
   171,648 slots and ≈ 59 % pads on the less skewed arxiv IVF. The official op reads a
   CSR and visits only real items. The comparison "at equal `n_probe`" is therefore
   equal *recall* but not equal *work*, and that is the finding: on a skewed IVF the
   padded layout, not the kernel, is what costs us.
3. **The official arm gives all of that back in payload prep**: 268–896 µs of scans,
   `repeat_interleave`, fills and gathers across 73–95 launches per forward against
   Triton's 34–58, so its *total device time* is 1.2–2.4× ours even where its scorer
   is 10× faster. §3/§13.2's launch-count conclusion is confirmed at the timing level.
4. **The arxiv exact cell's 7.21 ms is ours, not Meta's.** Its device time is 0.50–0.82 ms;
   the rest is the `filtering_bit_mask` our adapter builds — `clause_mask` over all
   2.99 M items plus `pack_mask`, whose `[B, N/64, 64]` intermediate is the 826 MiB
   peak and the 316 µs `reduce_kernel<long>` the classifier files under "quantize" in
   the raw JSON. Plan §5.1 already labels this path "phase 2 ours, full-N"; the
   measurement says it is also the dominant cost of that path, and that the official
   backend's exact mode is an adapter artefact, not a vendor result. **Reported, not
   fixed** (B3 measures; a transposed or chunked packer is Phase G work).
5. The `topk` epilogue is identical in both arms by construction (324–339 µs on
   goodreads, 119–125 µs on arxiv) and dominates the unfiltered Triton cell — which
   is why the wall ratios are smaller than the scorer ratios.

### 16.3 Phase 2 alone (§9b), `bs = 16`, full `N`

| dataset | op | arm | median ms | p99 ms | spread | timer |
|---|---|---|---|---|---|---|
| goodreads | `bloom_match` (row-wise, full N) | ours | 0.4517 | 0.5610 | 0.060 | CUDA events |
| goodreads | `bloom_index_search_batch` (transposed, full N, packed) | official | **0.2279** | 0.2588 | 0.003 | CUDA events |
| goodreads | `…_return_partial_response` (probed clusters only) | official | 0.4488 | 0.4971 | 0.029 | CUDA events |
| goodreads | `build_query_signatures` (our query bloom bits) | ours | 0.3687 | 0.4191 | 0.004 | CUDA events |
| goodreads | `queries_to_expressions` (host, incl. its D2H) | official | 0.0272 | 0.0376 | — | `perf_counter` |
| goodreads | `parse_expression_query_batch` (host) | official | 0.0468 | 0.0594 | — | `perf_counter` |
| arxiv | `bloom_match` (row-wise, full N) | ours | 1.4809 | 1.4898 | 0.000 | CUDA events |
| arxiv | `bloom_index_search_batch` (transposed, full N, packed) | official | **0.2443** | 0.2814 | 0.003 | CUDA events |
| arxiv | `…_return_partial_response` (probed clusters only) | official | 0.4513 | 0.5039 | 0.016 | CUDA events |
| arxiv | `build_query_signatures` (our query bloom bits) | ours | 0.3780 | 0.4580 | 0.015 | CUDA events |
| arxiv | `queries_to_expressions` (host, incl. its D2H) | official | 0.0273 | 0.0363 | — | `perf_counter` |
| arxiv | `parse_expression_query_batch` (host) | official | 0.0468 | 0.0559 | — | `perf_counter` |

- **S13 replicated with Meta's code, and it scales the way the transposed-index
  argument predicts.** Full-`N` mask, same batch: official is **2.0× faster on
  goodreads (0.8 M items) and 6.1× on arxiv (3.0 M)**. Our row-wise `bloom_match`
  grows 3.3× from 0.8 M to 3.0 M items, the official transposed search grows 1.07×.
  This is the strongest argument in the run for TF-1 (§8), and it is now measured
  against the vendor rather than inferred from the deleted CUDA backend.
- **The official partial-response path is *slower* than the official full-`N` search**
  at these sizes (0.449 vs 0.228 ms, 0.451 vs 0.244 ms) — 13 launches, 2 syncs and a
  `repeat_interleave` against one launch. The co-design still wins end to end (§16.5)
  because it shrinks what the *scorer* then reads, not because phase 2 is cheaper.
- **Our query-side bloom hashing costs 0.37 ms at `bs = 16`** — more than the whole
  official mask search — and it is inside our fused bloom forward: it is most of the
  gap between the Triton `none` (0.75 ms) and `bloom` (1.01 ms) kernel-only cells.
  Under CUDA graph it largely disappears (§16.5's graph column). A Triton-side
  opportunity, recorded here, not acted on.
- **Parse cost**: 46.8 µs/call at `bs = 16` (2.9 µs/query), plus 27.2 µs for
  `queries_to_expressions` including its `.tolist()` sync — ≈ 74 µs of host work per
  official bloom forward. A3 measured 58.7 µs for the parser alone on a different
  expression shape (§13.2); same order, and still a real line item at `bs = 1`.

**Bloom selectivity and memory, real attributes, shipped settings**

| dataset | exact pass rate | our bloom FP rate | official bloom FP rate | our bloom MiB | official index MiB |
|---|---|---|---|---|---|
| goodreads `c0_genre` | 0.3323 | **0.000000** | **0.000000** | 97.3 (`m_bits = 1024`) | **33.3** (`b_multiplier = 10`) |
| arxiv `c0_maincat` | 0.1357 | **0.000000** | **0.000000** | 364.9 | **71.3** |

Both blooms have **zero false positives** on these sweeps (8 batches × 16 queries,
counts against the exact mask), so §4.3's matched-FPR bisection is **undefined here**
— there is no FPR to match. What is comparable is memory at equal (zero) FPR, and
the official index is **2.9× / 5.1× smaller**. Our `m_bits = 1024` is simply
over-provisioned for single-clause sweeps at these vocabularies; the FPR-vs-width
curve (S8) needs a wider sweep and stays D3's job.

### 16.4 Parity alongside speed — and where the goodreads/arxiv jaccard split comes from

Kernel-only, 512 queries per cell (32 pool batches at `bs = 16`), against the Triton
arm on the same batches:

| dataset | mode | score path | jaccard@100 | `score_max_abs_diff` |
|---|---|---|---|---|
| goodreads | none | int32 | **1.000000** | **0.0** |
| goodreads | none | fp16 | 1.000000 | 2.885e-03 |
| goodreads | bloom / exact | int32 | 0.999961 | **0.0** |
| goodreads | bloom / exact | fp16 | 0.999961 | 2.916e-03 |
| arxiv | none | int32 | **1.000000** | **0.0** |
| arxiv | none | fp16 | 0.982928 | 4.814e-04 |
| arxiv | bloom / exact | int32 | **1.000000** | **0.0** |
| arxiv | bloom / exact | fp16 | 0.985788 | 4.814e-04 |

**The int32 path is bit-exact against Triton on both datasets, in all three filter
modes** — D5's contract, now confirmed on real data at scale as well as in B2's unit
regimes. The goodreads 0.999961 comes with `score_max_abs_diff = 0.0`: identical
scores, ids differing only where scores tie.

**So the whole of the arxiv deficit is the fp16 score path**, and the steer's
hypothesis for it is falsified. O's steer (from C4) read the 0.985-vs-0.9998 split as
"more near-ties at the top-100 boundary *under a looser filter*". It is not the
filter: the unfiltered (`none`) cells show the same split, 0.9838 end to end and
0.9829 kernel-only on arxiv against 0.9999 / 1.0000 on goodreads. It is the score
distribution of the dataset:

| dataset | mode | rank-100/101 gap p10 | gap median | score@100 median | one fp16 ulp there | **rows with gap < 1 ulp** |
|---|---|---|---|---|---|---|
| goodreads | none | 9.46e-04 | 6.97e-03 | 0.5623 | 2.75e-04 | **3.3 %** |
| goodreads | bloom / exact | 1.10e-03 | 6.85e-03 | 0.6320 | 3.09e-04 | **3.1 %** |
| arxiv | none | 1.38e-05 | 7.92e-05 | 0.8363 | 4.08e-04 | **95.3 %** |
| arxiv | bloom / exact | 1.41e-05 | 8.57e-05 | 0.8303 | 4.05e-04 | **94.5 %** |

arxiv's nomic text embeddings put the 100th and 101st candidate **8.6e-5 apart on a
score of 0.83**, a fifth of an fp16 ulp; goodreads' gSASRec scores are 22× further
apart than their ulp. Under a score path that rounds to fp16, 95 % of arxiv queries
*can* swap their boundary ranks and ~3 % of goodreads ones can. That is the
mechanism, it is a property of the embedding, and it is the number F2 should quote
rather than the jaccard alone.

What it costs in the metric anyone cares about: **recall@100 against the exact oracle,
end to end, 10,000 queries** — arxiv `c0_maincat` n_probe 24, official 0.883735 vs
Triton 0.884044 (**Δ 3.1e-4**); goodreads `c0_genre`, 0.912796 vs 0.912800 (**Δ 4e-6**).
`torch` matches `triton` at **jaccard 1.0, `score_max_abs_diff` 0.0** on all six
filter cells (the post-L5 state).

### 16.5 End to end (§9c) — `k = 100`, seed 0, harness `filter` + `quality` suites

30 records (24 `ok`, 6 `partial` — the `quality` suite was narrowed to `--k 100 --bs
1 --bs 8 --bs 16` so the unfiltered arm is timed at the same batch sizes; the
`filter` suite ran unnarrowed at `ks {100,500,1000} × bs {1,8,16}`). All at one
`code_version`, `dirty: false`, `env.sm_mhz_load` **1410 MHz on all 30**.

| dataset | filter | n_probe | backend | eager bs=1 | eager bs=8 | eager bs=16 | graph bs=16 | qps bs=16 | peak MiB bs=16 | index MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| arxiv | bloom | 24 | `triton` | 1.083 | 1.075 | 1.094 | 0.428 | 14518 | 32 | 786 |
| arxiv | bloom | 24 | `official` | 1.176 | 1.370 | 1.292 | n/a (not_capturable) | 12355 | 61 | 482 |
| arxiv | bloom | 24 | `torch` | 1.162 | 4.253 | 8.170 | 2.319 | 1957 | 1725 | 786 |
| arxiv | bloom | 32 | `triton` | 1.069 | 1.087 | 1.084 | 0.526 | 14751 | 42 | 786 |
| arxiv | bloom | 32 | `official` | 1.300 | 1.372 | 1.451 | n/a (not_capturable) | 11102 | 81 | 482 |
| arxiv | bloom | 32 | `torch` | 1.240 | 5.565 | 10.806 | 3.017 | 1480 | 2299 | 786 |
| arxiv | clause | 24 | `triton` | 0.560 | 0.689 | 0.701 | 0.452 | 22750 | 32 | 786 |
| arxiv | clause | 24 | `official` | 1.167 | 3.956 | 7.110 | n/a (not_capturable) | 2255 | 827 | 776 |
| arxiv | clause | 24 | `torch` | 0.735 | 4.076 | 7.863 | 2.276 | 2033 | 2060 | 786 |
| arxiv | clause | 32 | `triton` | 0.695 | 0.704 | 0.709 | 0.560 | 19981 | 42 | 786 |
| arxiv | clause | 32 | `official` | 1.150 | 3.924 | 7.131 | n/a (not_capturable) | 2240 | 827 | 776 |
| arxiv | clause | 32 | `torch` | 0.902 | 5.334 | 10.410 | 2.990 | 1536 | 2745 | 786 |
| arxiv | none | — | `triton` | 0.541 | 0.659 | 0.668 | 0.348 | 23821 | 32 | 421 |
| arxiv | none | — | `official` | 0.834 | 0.834 | 0.846 | n/a (not_capturable) | 18851 | 61 | 411 |
| arxiv | none | — | `torch` | 0.589 | 2.549 | 4.852 | 2.128 | 3294 | 1722 | 421 |
| goodreads | bloom | 24 | `triton` | 0.999 | 1.020 | 1.112 | 0.970 | 14305 | 112 | 394 |
| goodreads | bloom | 24 | `official` | 1.330 | 1.411 | 2.063 | n/a (not_capturable) | 7501 | 215 | 143 |
| goodreads | bloom | 24 | `torch` | 2.088 | 14.325 | 28.516 | 7.192 | 561 | 6140 | 394 |
| goodreads | bloom | 32 | `triton` | 1.055 | 1.096 | 1.350 | 1.206 | 11797 | 150 | 394 |
| goodreads | bloom | 32 | `official` | 1.344 | 1.637 | 2.260 | n/a (not_capturable) | 7088 | 287 | 143 |
| goodreads | bloom | 32 | `torch` | 2.668 | 19.065 | 37.870 | 9.346 | 422 | 8186 | 394 |
| goodreads | clause | 24 | `triton` | 0.519 | 0.649 | 0.921 | 0.842 | 17253 | 112 | 394 |
| goodreads | clause | 24 | `official` | 1.053 | 1.677 | 3.044 | n/a (not_capturable) | 5240 | 228 | 207 |
| goodreads | clause | 24 | `torch` | 1.980 | 13.843 | 27.575 | 6.811 | 580 | 7335 | 394 |
| goodreads | clause | 32 | `triton` | 0.648 | 0.655 | 1.119 | 1.037 | 14224 | 150 | 394 |
| goodreads | clause | 32 | `official` | 1.051 | 1.837 | 3.291 | n/a (not_capturable) | 4849 | 300 | 207 |
| goodreads | clause | 32 | `torch` | 2.544 | 18.407 | 36.638 | 9.022 | 437 | 9779 | 394 |
| goodreads | none | — | `triton` | 0.534 | 0.667 | 0.857 | 0.894 | 18539 | 112 | 297 |
| goodreads | none | — | `official` | 0.897 | 0.904 | 1.325 | n/a (not_capturable) | 11995 | 215 | 110 |
| goodreads | none | — | `torch` | 1.274 | 8.490 | 16.918 | 6.639 | 945 | 6131 | 297 |

| dataset | filter | n_probe | backend | recall@100 vs oracle | jaccard_vs_first@100 | score_max_abs_diff | parity | unstable |
|---|---|---|---|---|---|---|---|---|
| arxiv | bloom | 24 | `triton` | 0.884044 | reference | — | reference | True |
| arxiv | bloom | 24 | `official` | 0.883735 | 0.984950 | 9.510e-02 | vs_triton | True |
| arxiv | bloom | 24 | `torch` | 0.884044 | 1.000000 | 0.000e+00 | vs_triton | False |
| arxiv | bloom | 32 | `triton` | 0.903351 | reference | — | reference | True |
| arxiv | bloom | 32 | `official` | 0.903010 | 0.984797 | 9.795e-02 | vs_triton | True |
| arxiv | bloom | 32 | `torch` | 0.903351 | 1.000000 | 0.000e+00 | vs_triton | True |
| arxiv | clause | 24 | `triton` | 0.884044 | reference | — | reference | True |
| arxiv | clause | 24 | `official` | 0.883757 | 0.985033 | 4.827e-04 | vs_triton | False |
| arxiv | clause | 24 | `torch` | 0.884044 | 1.000000 | 0.000e+00 | vs_triton | False |
| arxiv | clause | 32 | `triton` | 0.903351 | reference | — | reference | True |
| arxiv | clause | 32 | `official` | 0.903036 | 0.984882 | 4.827e-04 | vs_triton | False |
| arxiv | clause | 32 | `torch` | 0.903351 | 1.000000 | 0.000e+00 | vs_triton | False |
| arxiv | none | — | `triton` | — (`none` cell) | reference | — | reference | False |
| arxiv | none | — | `official` | — (`none` cell) | 0.983778 | 4.827e-04 | vs_triton | False |
| arxiv | none | — | `torch` | — (`none` cell) | 1.000000 | 0.000e+00 | vs_triton | True |
| goodreads | bloom | 24 | `triton` | 0.912800 | reference | — | reference | True |
| goodreads | bloom | 24 | `official` | 0.912796 | 0.999849 | 5.517e-03 | vs_triton | True |
| goodreads | bloom | 24 | `torch` | 0.912800 | 1.000000 | 0.000e+00 | vs_triton | False |
| goodreads | bloom | 32 | `triton` | 0.936908 | reference | — | reference | True |
| goodreads | bloom | 32 | `official` | 0.936910 | 0.999805 | 5.622e-03 | vs_triton | True |
| goodreads | bloom | 32 | `torch` | 0.936908 | 1.000000 | 0.000e+00 | vs_triton | False |
| goodreads | clause | 24 | `triton` | 0.912800 | reference | — | reference | False |
| goodreads | clause | 24 | `official` | 0.912796 | 0.999849 | 5.517e-03 | vs_triton | True |
| goodreads | clause | 24 | `torch` | 0.912800 | 1.000000 | 0.000e+00 | vs_triton | False |
| goodreads | clause | 32 | `triton` | 0.936908 | reference | — | reference | False |
| goodreads | clause | 32 | `official` | 0.936910 | 0.999805 | 5.622e-03 | vs_triton | True |
| goodreads | clause | 32 | `torch` | 0.936908 | 1.000000 | 0.000e+00 | vs_triton | False |
| goodreads | none | — | `triton` | — (`none` cell) | reference | — | reference | False |
| goodreads | none | — | `official` | — (`none` cell) | 0.999885 | 3.794e-03 | vs_triton | False |
| goodreads | none | — | `torch` | — (`none` cell) | 1.000000 | 0.000e+00 | vs_triton | False |

Reading the end-to-end tables:

- **Triton is the fastest arm in every cell of the matrix**, eager and under graph.
  Against official at `bs = 16`: **1.5× (goodreads none), 1.9× (goodreads bloom),
  3.3× (goodreads clause), 1.2× (arxiv none), 1.2× (arxiv bloom), 10.1× (arxiv
  clause)**. At `bs = 1` the margin is 1.1–2.0× and sits inside this box's `bs = 1`
  noise for the bloom cells — stated as such, not as a result.
- **The co-design is visible in Meta's own numbers, and it inverts the two arms'
  filter ordering.** For official, bloom (partial masks over probed clusters) is
  *cheaper* than exact (full-`N` `filtering_bit_mask`): 1.29 vs 7.11 ms on arxiv,
  2.06 vs 3.04 ms on goodreads. For Triton the order is the other way (1.09 vs 0.70,
  1.11 vs 0.92) because our exact predicate is fused into the scorer and our bloom
  pays the query-hash of §16.3. This is the paper's §4.3 claim (S9) reproduced from
  the vendor side, though not as the controlled full-vs-partial ablation §9(d) wants
  (that needs `bloom_path="full"` cells, **not run** — see §16.6).
- **Graph mode is Triton-only** (D7): **all 78 official `graph` entries** are `null`
  with `reason: not_capturable`, and `torch.compile(mode="reduce-overhead")` gives
  Triton 0.35–1.21 ms at `bs = 16`: **1.6–2.6× over its own eager cell on arxiv**,
  1.09–1.15× on the goodreads filter cells, and **0.96× — i.e. slightly slower —
  on goodreads `none`** (0.894 vs 0.857 ms), the one cell where capture does not
  pay. Triton is still the fastest arm in every graph cell (`torch` never wins one).
  The eager column is the comparable number; the graph column is the
  deployed-best-case one (H §2.7).
- **`torch` is the floor and stays bit-exact**: 1.0 jaccard and `score_max_abs_diff`
  0.0 against Triton on all six filter cells, at 4.9–37.9 ms per `bs = 16` forward
  (**7.3–32.7× Triton**) and 1.7–9.6 GiB of peak forward memory against Triton's
  32–150 MiB.
- **Memory.** The official arm's *index* is smaller wherever the padded layout bites:
  **2.70×** on goodreads `none` (110 vs 297 MiB), 2.75× goodreads bloom, 1.90×
  goodreads clause, 1.63× arxiv bloom — but only **1.01–1.02×** on the arxiv `none`
  and `clause` cells, where the CSR's two permutation vectors nearly cancel the
  padded table they replace. The skew that costs us kernel time in §16.2 is the same
  thing that costs us ~200 MiB on goodreads. The official *forward* peak is ~1.9×
  ours on the bloom and `none` cells and **26× ours on arxiv clause** (827 vs 32 MiB,
  the adapter's packer again).
- **`bloom_fp_rate` is 0.000000 on all four of our bloom cells** at
  `m_bits = 1024, k_hash = 5`, consistent with §16.3.
- **Stability.** 14 of 30 cells carry `unstable: true`; every one of them is either a
  `bs ∈ {1, 8}` eager entry with spread 5–22 % or a `clocks_drift` flag raised by the
  same host-bound arms dropping the card to 1155–1395 MHz (10 cells, all bloom or
  arxiv-clause). Of the 156 `bs = 16` entries, **5 exceed the 5 % threshold**; the
  only one inside a headline ratio above is arxiv bloom official `n_probe = 24` at
  **8.1 %**, so read that cell's 1.2× as 1.1–1.3×. The six `torch` cells and the
  goodreads `triton` `none` / `clause` cells are stable throughout. Per-entry
  `spread` and `sm_mhz` are in the JSONL.

### 16.6 What this settles, what it corrects, and what is still open

**§9's expected outcomes, checked one by one.**

| §9 expectation | verdict |
|---|---|
| (i) kernel-only, no filter, large P: "parity within ±20 %, a 310-line Triton kernel within X % of the vendor's" | **Wrong in both directions.** Meta's scorer kernel is 1.15× (arxiv) to 10.9× (goodreads) faster than ours; our *forward* is still 1.7× faster because their op spends 268–896 µs in prep. The goodreads factor is our padded layout's pad tax (97 % pads), not bandwidth. |
| (ii) eager wall at `bs = 1` / small P: official 2–3× slower from launches + syncs | **Confirmed, milder**: 1.4–2.5× kernel-only, 1.1–2.0× end to end, and the card visibly drops clock in those cells. |
| (iii) bloom kernel-only at `bs = 16`: official partial mask + masked scorer beats row-wise fused Triton by ~2× until TF-1 | **Wrong as stated.** Their *phase 2* beats ours by 2.0–6.1× (full-`N` search), but their bloom *forward* is **1.8× (arxiv) / 2.1× (goodreads) slower** than our fused one. The partial-response op is also slower than their own full-`N` search. |
| (iv) quality: identical candidate sets up to fp16 ties | **Confirmed and quantified** (§16.4): int32 bit-exact; fp16 costs 3.1e-4 recall@100 on arxiv, 4e-6 on goodreads. |
| **S13** (transposed vs row-wise phase 2) | **Replicated against Meta's code**, and the advantage grows with `N` (2.0× at 0.8 M, 6.1× at 3.0 M). |
| **S9** (co-design) | **Directionally reproduced from the official side** (their partial-mask path beats their own full-mask path by 1.5–5.5× end to end); the controlled ablation is not run. |
| **S8** (FPR vs bits) | **Not measurable at these settings** — both blooms are at FPR 0.0. Needs D3's wider sweep. |

**Corrections to this plan's text.** §9's "expected outcomes" (i) and (iii) are wrong
as written and should be read against the table above. §8's TF-1 case is
*strengthened* — the transposed index is worth 2–6× on phase 2 — while TF-3/TF-4
(scorer retunes) are now clearly second-order next to **the padded layout itself**,
which is what costs the Triton scorer 10× on a skewed IVF. A "TF-9: CSR or
capped-pad probe layout" belongs in §8 for Phase G; the orchestrator owns that edit.

**What was skipped, and why.**
- `fpr_calibrate.py` and the §4.3 matched-FPR bisection: undefined at FPR 0.0 for both
  blooms on these sweeps (§16.3). Matched *memory* is reported instead.
- §9(d)'s S9 ablation with `OfficialConfig(bloom_path="full")`, S10 (probed fraction),
  S6/S16 (per-row vs global scale), S12: not run — WP-7 / D1 / D3 scope.
- P ∈ {1024, 46 720, 58 368} synthetic layouts of §9a: replaced by the two real
  datasets' own `P` (611,520 goodreads, 171,648 arxiv at `n_probe = 24`), which is
  where the pad-tax finding came from; the synthetic ladder was not run.
- `ncu` is blocked on this box, so every kernel number is `torch.profiler` device
  time, not an occupancy or sector analysis.
- Seeds 1 and 2, `k ∈ {500, 1000}` ratios, and the third dataset dimension: out of
  scope here (D1's).

**Unverified / carried forward.**
- The 826 MiB peak and 7.2 ms wall of the official **exact** path are the adapter's
  full-`N` `pack_mask`, diagnosed from the kernel list but **not fixed and not
  re-measured after a fix**; every official-exact number is an upper bound on what
  that path could cost with a better packer.
- Our 0.37 ms query-bloom-hash at `bs = 16` is measured but not attributed to a
  specific kernel inside `build_query_signatures`.
- The kernel-class split is a name heuristic (§16.2); the `quantize` column of the
  arxiv official-exact row is really the packer's reduction.
- Parity is measured on 512 queries kernel-only and 10,000 end to end, at `k = 100`;
  the `k = 500 / 1000` jaccards are in the JSONL but not analysed here.
- **Nothing in this section is citable until the orchestrator re-runs the gate**
  (rule 2), and B3's checkbox is the orchestrator's to flip.
