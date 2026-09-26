---
chain: "silvertorch-official-integration"
branch: "main"
nextStep: "WP-0: pin and build meta-recsys/silvertorch on the A100 (uv sync --extra official), then WP-1: measure the op facts (bit order, syncs, launches, capture, parse cost, per_embedding_scale overflow)."
created: "2026-09-05T12:00:00Z"
---

# Plan O: Meta's official SilverTorch ops as the reference backend

Source: `docs/plans/silvertorch-official-integration.md` §1-§12 (plan "O"), planned 2026-09-05 on `feat/cute-dsl-scorer`, authored on the Mac against a clone of `meta-recsys/silvertorch` at `21aa35e` (2026-07-23, `main`, no tags, Apache-2.0). Raw artifacts: `docs/artifacts/official-silvertorch/`. Ordering: WP-0/1 = roadmap A2/A3, WP-2..5 = Phase B, WP-6 folds into C1/C4, WP-7 into D1, WP-8 (TF-1) = G-a, WP-9 = F2.

Scope decision (user, same day): the official ops become the reference backend for Algorithm 1 phases 2+3; the hand-written CUDA C++ backend and its CuTe port are deleted (the CuTe port's no-nvcc advantage vanishes once the official package needs nvcc); `triton` stays as our reimplementation, `torch` as the readable eager reference; kernel effort goes into Triton.

Question: run Meta's kernels inside our layer and harness so they are (1) the ground truth Triton is checked against and (2) the vendor arm of the paper's comparison: is a ~600-line Triton reimplementation competitive with ~9.4k lines of vendor CUDA C++?

## 1. Upstream inventory (as read)
12 commits (2026-04-21 -> 2026-07-23), 9,427 lines of C++/CUDA under `silvertorch/ops/csrc/` (~3.5k `fused_kmean_ann*`, ~4.5k bloom index / search / parser), not on PyPI (README's `pip install silvertorch` 404s), one extension `silvertorch._C`, `--no-build-isolation`, ops under `torch.ops.st.*`, no fake / meta kernels anywhere.

| Alg. 1 phase | official op | ours | for `official` |
|---|---|---|---|
| k-means / IVF | none (bring your own, CSR `cluster_offsets`) | `KMeans` + padded layout | ours, cluster-sorted CSR |
| int8 quantization | none; op takes int8 embeddings and int8 queries | `quantize_int8_global` / `quantize_int8` | ours, scale host-side |
| bloom build | `bloom_index_build(feature_ids, feature_offsets, feature_values, b_multiplier, k, fast_build)` | `build_signatures` | official (their hash) |
| predicate | `parse_expression_query_batch` CPU parser, AND/OR/NOT/parens | `build_query_signatures` | official, plans cached |
| phase 2 partial | `bloom_index_search_batch_return_partial_response` | fused row-wise subset | official |
| phase 2 full | `bloom_index_search_batch` (bool or packed int64) | `bloom_match` | official (S9 baseline) |
| phase 3 | `fused_kmean_ann`, `fused_kmean_ann_with_partial_masks` | Triton scorer | official = ground truth |
| top-k | none (`is_topk` only marks) | host `masked_topk` | ours, identical in both arms |
| exact / clause | none | `codesigned_probe_score_exact`, `clause_mask` | ours via `filtering_bit_mask` |

Reimplementer facts: DIM in {16,32,64,96,128,192,256,384,512,768,1024,1280}; int8 rows read as `int4` (DIM % 16 == 0); the double-buffered `process_cluster_v4_pipelined` is dead code (no call site); the live scorer is `process_cluster` (thread per document, `DIM/16` int4 words per lane, 32 rows in flight per warp); `max_tensor_size_per_row` rounded up to 32; int8 score dtype fp16 with a divisor, else int32; bloom `k <= 10` is an unchecked hard limit (`MAX_K_V2`); neither official CPU reference is an oracle (the CPU scorer reads masks low-bit-first, `_with_partial_masks` CPU ignores its masks). Audit notes: all bloom-v1 template branches unreachable; three functions declared but never defined; the scorer re-implements bit helpers inline at six sites; CUDA entry points do not check dtypes.

## 2. Decisions
- D1 `backend="official"` joins `triton` and `torch` as the reference.
- D2 pinned git dependency, not vendored, not a submodule (the paper wants "Meta's code at commit X, unmodified"). Rejected: vendoring 9.4k lines (a fork the moment anyone touches it), submodule (same pin, worse ergonomics, no lock record; fallback only).
- D3 one layer, shared phase 1: our k-means, quantization and probe selection; phases 2+3 to `torch.ops.st.*`. Both arms score the same clusters and the same int8 codes.
- D4 cluster-sorted buffers for `official` (`item_codes` in CSR order + `sort_perm` / `inv_perm`, no `padded_cluster_items`); state dicts not portable.
- D5 correctness: `divisor_for_int8=-1` returns the raw int32 dot; our `(dot.float() * q_scale) * global_scale` must `torch.equal`, ids up to ties. Bloom: official ⊇ exact, FPR measured, partial-mask scores == full-mask scores (paper §4.3).
- D6 bloom on `official` is Meta's bloom (their hash, bundles, parser); fairness = matched FPR and matched memory, both measured.
- D7 `official` runs eager only (every op syncs, pageable plan uploads per call, data-dependent output shapes); `graph` recorded `null` with reason `not_capturable`; the headline is eager vs eager, Triton's graph number alongside.
- D8 the paper question is Triton vs official; `torch` is the floor; CUDA / CuTe only as quotes from the archived records.
- D9 deletion only after the official parity gate.
- D10 timing protocol is harness v2 §2 plus the official sha, `nvcc --version`, the parse-cost row.

## 3. Host behaviour predicted (corrected in §13)
`fused_kmean_ann` ~2 syncs (`repeat_interleave` without output size, a `.item()`), ~12 launches; `_with_partial_masks` 1-3 syncs; `bloom_index_search_batch` a host plan decode and a pageable upload every call; only `generate_column_info_for_clusters` capturable. The syncs are not intrinsic: upstream ships `faster_repeat_interleave._with_cumsum_raw` (explicit output size) but the scorer never calls it (worth an upstream issue, not our patch).

## 4. Numerics
- Layout: `sort_perm = argsort(assignments)`, `embeddings = item_codes[sort_perm]`, `cluster_ids = probe_ids`, `cluster_length = cluster_sizes[probe_ids]`; returned indices are sorted positions mapped back through `sort_perm`.
- Scale paths: (i) `divisor_for_int8=-1` int32 raw dot, bit-exact parity path; (ii) `divisor = 2^k` fp16 `dot / divisor` (|dot| <= 127²·D = 2,064,512 at D=128, so divisor >= 32; 16 / 32 / 64 at D = 64 / 128 / 256), the shipped serving instantiation and the timed path, gated at `jaccard@k >= 0.99`; (iii) `per_embedding_scale` unusable: the int32 dot is cast to fp16 before dividing, overflowing for D >= 5 at full-range codes.
- Bloom: bundles of 2048 docs; per bundle `B = int(max_terms_per_doc · k · b_multiplier)` bits per doc, index `B/8` bytes per doc; murmur3 `(feature_id, value, seed++) % B` with duplicate rejection; doc `d` at `1 << (63 - d % 64)`. Width tracks the bundle maximum term count (ours fixed), so matched memory != matched FPR. A term ANDs the first `k` distinct positions among `hash_k` raw hashes (hence `hash_k > k`; README 7 for k=3, we use 7 for k=5).
- Mask bit order trap: the GPU scorer reads high-bit-first ("lower doc id put at higher bits"), the CPU reference low-bit-first. Our packer follows the GPU; T3 pins it.
- Attributes -> features: `feature_ids = arange(C)` (clause index as feature id, same `(clause, value)` keying as our salt), values the non-`-1` entries in `(doc, clause)` order, in cluster-sorted doc order. Queries -> `"0:v0 AND 1:v1"`, reverse -> `NOT c:v`, none active -> `""` (match all). Precedence NOT > AND > OR.

## 5. Adapter and tests
Adapter `ops/official/`: `is_available` / `ensure_loaded` (missing vs broken), `attrs_to_features`, `queries_to_expressions`, `parse_plans` (LRU), `pack_mask_high_first`, eager scorers returning `[B, P]` fp32 + ids with `-inf` / `-1` pads. Filter matrix: `none` -> `fused_kmean_ann`; `bloom` -> partial response + `_with_partial_masks` (default) or full search + `filtering_bit_mask` (`bloom_path="full"`, S9); `exact` -> our Triton `clause_mask` over sorted attrs packed into `filtering_bit_mask` (phase 2 ours, full N, labelled so).

Tests T1-T7 (`tests/parity/test_official.py`): T1 int32 bit-exact vs reference and Triton; T2 fp16 `jaccard@k >= 0.99`, `max_rel_err <= 2^-10`; T3 bit order; T4 bloom ⊇ exact, FPR < 5 % at `b_multiplier=10`; T5 partial ≡ full masks; T6 the layer vs Triton in all three modes + state dict; T7 compile refused, syncs counted.

## 6. Integration (option A)
`retrieve[official] = ["silvertorch"]`; root `[tool.uv.sources] silvertorch = { git = ..., rev = <sha> }`, `no-build-isolation-package = ["silvertorch"]`, `[[tool.uv.dependency-metadata]]` so `uv lock` never runs their `setup.py`. README tested matrix: recommended Python 3.11-3.12 / torch 2.7-2.10 / CUDA 12.8; torch 2.11 + CUDA 13 unsupported.

## 7. Deletion (after WP-3)
Planned: ~3,680 lines of kernel / host / test code, `.cu` 672, cuda host 611, cute 650 + 763, parity tests 454 + 527; keep `build_transposed_sigs` in `bloom_hash.py` for TF-1; tag `cuda-cute-backends-final`; archive the three plans.

## 8. Triton follow-ups (effort goes here)
- TF-1 transposed cluster-major bloom index + 1-bit-mask scorer: row-wise reads 128 B per probed item (119 MB at B=16, P=58,368); transposed reads ~0.6-3 B/item. Triton design: grid `(B, n_probe · wpc)`, masked loads over `m_bits` rows (traffic proportional to set bits without the `__ffsll` walk Triton cannot express), `HAS_MASK` scorer path. Gain est. bloom B=16 kernel-only 117 -> ~55 µs. 2 d.
- TF-2 salt as a buffer (a pageable H2D per forward that broke raw capture, ~0.4 ms eager): done as roadmap B5.
- TF-3 retune after TF-1 (Triton already ~1.4 TB/s, ~90 % of HBM; M=1 `tl.dot` cannot use IMMA). TF-4 `eviction_policy="evict_first"` (0-5 %). TF-5 id prefetch: nothing to write. TF-6 the official pipelined kernel is dead code: nothing to port. TF-7 fp16 score store: skip (costs bit-exactness). TF-8 exact mode: fused predicate already wins; optional S9-style ablation.

## 9. Perf-comparison plan (§9)
(a) phase 3 kernel-only + wall, D=128, B in {1, 16}, P in {1024, 46,720, 58,368}; (b) phase 2 only at N=3.03M; (c) end to end on arxiv / goodreads d128 filter + quality, three backends, `n_probe` in {24, 32}; (d) paper claims S9, S13, S8, S10, S6/S16, S12; (e) provenance. Expected outcomes as written: (i) kernel-only unfiltered parity within +-20 %; (ii) official 2-3x slower at B=1 for host reasons; (iii) official bloom kernel-only ~2x faster until TF-1; (iv) identical candidate sets up to fp16 ties. (i) and (iii) were falsified by B3.

## 10. Work packages
WP-0 pin + build (gate: < 5 min, upstream suite green); WP-1 op facts; WP-2 adapter + T1-T7 (Mac); WP-3 parity gate (unblocks deletion); WP-4 head-to-head; WP-5 delete cuda / cute; WP-6 harness backend (official cell on goodreads d128 `c0_genre` with `jaccard_vs_first@100 >= 0.99`); WP-7 TF-2 then campaign cells; WP-8 TF-1 + retune (bit-exact, bloom kernel-only within 1.3x of official); WP-9 paper section. ~10 focused days.

## 11. Risks
Minor nvcc mismatch warns, major raises; syncs make it eager-only; fp16 blurs top-k boundaries; no bloom bit comparison ever; `k <= 10` unchecked; CPU references are not oracles; upstream drift (bumps are deliberate PRs rerunning WP-3).

## 12. Claims unlocked
S9, S13, S8, S10, S6/S16, S12; G1's deviations table gains "official bloom hash != ours", "official eager-only", the `per_embedding_scale` overflow; G2 closes.
