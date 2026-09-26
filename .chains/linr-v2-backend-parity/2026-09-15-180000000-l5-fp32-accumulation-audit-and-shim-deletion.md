---
chain: "linr-v2-backend-parity"
branch: "main"
parent: "2026-09-15-120000000-l4-plan-and-record-linr-v2-divergence.md"
nextStep: "None; L5 merged (0469741). Unmeasured: arxiv and bloom cells after the fix, and V3 stage 2 (same kernel)."
created: "2026-09-15T18:00:00Z"
---

# L5: fp32 accumulation, the reduction-width audit, and the shim deletion

## Decision (user, 2026-09-15, on L4's numbers)
Ship fp32 accumulation and sweep the other kernels for the same defect; §3's "change nothing" default is overridden because there is no trade-off:
| | shipped fp16 acc | fp32 acc |
|---|---|---|
| kernel B=1, P=195,023 | 0.05235 ms | 0.05226 ms |
| kernel B=16, P=442,864 | 0.9636 ms | 0.9408 ms (2.4 % faster) |
| max abs error vs fp64 | 0.0276 | ~1e-5 |
| linr_v2 torch-vs-triton jaccard@100 | 0.998743 (624 rows) | 0.999751 (124 rows) |
The documented contract already said fp32: the code contradicted it. The audit is the point: `tl.sum` inherits its operand dtype, so every fp16 x fp16 reduction has this defect by default, and the parity suite is structurally blind (it compares against `ops.reference` at the same input dtype). It was found only because cuBLAS in the torch backend accumulates differently. Hence a parity file against an fp64 oracle.

WP-1 (`fable`, `dev/l5-fp32-accumulation`): fix `fused_masked_knn_topk`; audit every kernel (stop and ask if a fix would move SilverTorch); fp64 parity file asserting a bound fp32 meets and fp16 does not; restore the docs; re-measure. WP-2: delete `retrieve.layers` / `retrieve.kernels` (the golden stopped being a gate).

## §8.1 Record, 2026-09-15
Driver 580.159.04, nvcc 12.4, Python 3.11, torch 2.10.0+cu128, triton 3.6.0, `/venvs/l5` off `development` @ `81c55d3`, worktree `/workspace/wt/l5`, inductor `/tmp/inductor-l5`, `flock`; SM 1410 sampled (idle 1155). Artifacts `docs/artifacts/linr-v2-backend-parity/l5/` (`parity_and_cost.py`, json, `fused_masked_knn_topk_fp32.ptx`; logs gitignored). Merged `0469741`.

Fix: two `.to(tl.float32)` on the loads in `_fused_masked_knn_topk_kernel`; schema, buffer dtypes, config and tune spec untouched. PTX (D=128):
| body | add.f16 | add.f32 | fma.rn.f32 | mul.f32 | cvt.f32.f16 |
|---|---|---|---|---|---|
| pre-L5 | 8 | 0 | 0 | 0 | 1 |
| shipped | 0 | 8 | 14 | 2 | 24 |

Audit of every reduction in `ops/triton/`:
| file | reduction | operand | accumulates in | verdict |
|---|---|---|---|---|
| fused_masked_knn_topk.py | `tl.sum(emb_rows * q, axis=1)` | fp16/fp32, cast to fp32 first | fp32 | fixed |
| codesigned_probe_score.py | `tl.dot(q_codes[None, :], codes.T, out_dtype=tl.int32)` | int8 x int8 | int32 exact (`|dot| <= 127²·D < 2^24` for D <= 1024) | correct |
| codesigned_probe_score.py | `tl.sum(dots_2d, axis=0)` | int32 | int32 | correct |
| codesigned_probe_score.py | `dots_i32.to(tl.float32) * q_scale * global_scale` | fp32 scalars | two fp32 multiplies | correct; 1.1e-7 rel vs fp64 |
| codesigned_probe_score_exact.py | same three | int8 / int32 / fp32 | int32 then fp32 | correct |
| oporp_1bit_match_topk.py | `tl.sum(pop_words, axis=1)` | int32 | int32; `(D_TOTAL - 2 * hamming)` exact < 2^24 | correct |
| common.py popcount_int64 | SWAR | int64 | int64 -> int32 | correct |
| common.py bloom_subset_pass | `tl.reduce(..., or_combine)` | int64 | OR | boolean |
| common.py clause_pass | `|`, `&`, `^` | int1 | int1 | boolean |
| common.py compact_store | `tl.cumsum` | int32 | int32 | exact |
| common.py compact_stash | `tl.sum(where(pass, 1, 0))` | int32 | int32 -> int64 | exact |
| bloom_match, bloom_compact, clause_mask, clause_compact | none of their own | - | - | - |
| compact_scatter_kernel, _host.py | host `torch.cumsum` int64 | int64 | int64 | exact |
The only floating-point reduction in the tree was the one fixed; no SilverTorch number moves.

fp64 parity file `tests/parity/test_accumulation.py`, 8 tests (k = P, ids mapped back to an fp64 computation; asserts a bound and that an fp16 model misses it). Goodreads-like unnormalised fp16 inputs (|score| up to ~140):
| kernel | error vs fp64 | bound | fp16 model |
|---|---|---|---|
| fused_masked_knn_topk D 64 / 128 / 256 | 7.5e-6 / 9.7e-6 / 1.4e-5 abs | 1e-4 abs | 6.5e-2 / 9.0e-2 / 1.4e-1 |
| codesigned_probe_score D 64 / 128 | 1.1e-7 rel | 1e-6 rel | fp16 cannot hold the int32 dot (overflow) |
| codesigned_probe_score_exact D 64 / 128 | 1.0e-7 rel | 1e-6 rel | same |
| oporp_1bit_match_topk_indirect W=4 | 0 | torch.equal | integer |
Against the pre-L5 body the file fails at every D (7.2e-2 / 8.9e-2 / 1.4e-1).

Parity and cost re-measured (`parity_and_cost.json`), all 9,859 rows:
| | L4 fp16 acc | L5 fp32 acc |
|---|---|---|
| jaccard@100 | 0.998743 | 0.999751 (predicted 0.999751) |
| jaccard@500 / @1000 | - | 0.999500 / 0.999379 |
| rows whose top-100 differ | 624 | 124 |
| of which torch's swapped-pair scores exactly equal | - | 124 / 124 |
| score_max_abs_diff | 0.009766 | 0.003904 (1 fp16 ulp at |s| in [2, 4)) |
vs fp64 over chunk 0 (P = 442,864, |s| max 31.4): pre-L5 max abs 0.027644, mean 0.003362; shipped 3.2e-6, 4.0e-7.
do_bench (500 reps, warm-up 100): kernel B=1 0.05315 -> 0.05302 ms; B=16 0.9631 -> 0.9404 ms (-2.4 %); `LiNRV2` eager triton 0.497 / 3.271 ms (B=1 / 16), torch 1.518 / 18.28 ms.

WP-2: `layers/__init__.py` and `kernels/__init__.py` deleted with `KMeansTorch`, `build_silvertorch` and the module aliases; the promising paragraphs removed; `git grep -l "retrieve\.layers\|retrieve\.kernels" -- ':!docs/plans'` empty.

Gates: ruff clean (75 files); links 0; full suite 653 passed, 0 failed, 0 skipped in 123 s (645 + 8); existing parity files untouched (`test_fused_masked_knn_topk.py` uses fp32 inputs, where the cast is a no-op).

Unverified: only goodreads `c0_genre`; arxiv and bloom not re-measured; V3 stage 2 inherits the fix unmeasured; torch's fp16 output rounding left as is (cuBLAS's `bmm` output dtype; the 124 rows are its exact ties).
