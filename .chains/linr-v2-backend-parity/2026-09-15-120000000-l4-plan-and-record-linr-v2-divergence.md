---
chain: "linr-v2-backend-parity"
branch: "main"
nextStep: "User decides on L4's numbers whether to ship fp32 accumulation (L5)."
created: "2026-09-15T12:00:00Z"
---

# L4: why do two implementations of an exact algorithm disagree?

Source: `docs/plans/linr-v2-backend-parity.md` §1-§6. Planned 2026-09-15 on `development` at `c1ae1b5` by the orchestrator from C4's run; user decision: settle it as a correctness question before D1 (V2 is the paper's exact baseline). Executed on `dev/l4-linr-v2-parity` off `ea1243f`; merged `91fd352` at `d9a3200`. Artifacts: `docs/artifacts/linr-v2-backend-parity/`.

## Evidence
| where | measurement |
|---|---|
| C4 goodreads d128 `c0_genre` | `jaccard_vs_first@100` 0.998743, `score_max_abs_diff` 9.77e-3 |
| the golden, old harness | recall@100 torch 0.99969470 vs triton 0.99927376 (4.2e-4) |
| control, `linr_v1` / `linr_v4` | exactly 0.0 |
| SilverTorch same run | 1.0, diff 0.0, both datasets |

## Plan
The one question: do both backends score the same candidate set? After L3 they must. If identical, the divergence is scoring precision: document it and correct the "exact algo" gate wording (exact describes filtering, not arithmetic); default is to change nothing in the kernels (changing width would move every V2 number; bring it back with numbers). If different, stop: a bug that invalidates a baseline. Carried from C4: L4-b (run `linr_v4` at quality chunk 64) and L4-c (`clocks_drift` / `clocks_locked` artifacts).

## §6.1 Record, 2026-09-15
Driver 580.159.04, nvcc 12.4, Python 3.11, torch 2.10.0+cu128, triton 3.6.0, `/venvs/l4`, worktree `/workspace/wt/l4`, inductor `/tmp/inductor-l4`, `flock`, SM 1410 MHz sampled. Artifacts: `probe_v2_parity.py` (+ json, log, `fused_masked_knn_topk_fp16.ptx`), `boundary_and_fp32_variant.py`, `torch_side_rounding.py`, `library-suite.log`. Inputs are the C4 cell's (goodreads d128 SASRec cache, first 10,000 users, `c0_genre`, 141 skipped -> 9,859 kept, chunks of 16, `k_max = 1000`); the probe reproduces C4 to the digit (jaccard 0.9987427, diff 0.009765625); both backends deterministic on repeat.

Gate 1: candidate sets identical: 0 / 9,859 rows differ on `counts` or ids.

What each backend computes (chunk 0, P = 442,864, vs an fp64 dot of the same fp16 inputs; goodreads queries have norm ~20, scores to |s| 31, median 12, top scores near 0 to -1):
| path | max abs err | mean abs err | note |
|---|---|---|---|
| torch: cuBLAS bmm, fp16 out | 0.00782 | 0.00203 | <= 1.0 ulp; pure output rounding |
| triton: shipped kernel | 0.0276 | 0.00336 | every written score is an fp16 value |
| fp16 sequential model | 0.155 | 0.0136 | the kernel's reduction is a tree |
| fp32 sequential model | 1.6e-5 | 1.6e-6 | what fp32 accumulate gives |
PTX: 8 `add.f16`, 0 f32 adds / FMAs / muls, 1 `cvt.f32.f16`. Cause: `tl.sum` keeps the operand dtype (Triton 3.6.0 `_pick_sum_dtype` promotes only sub-32-bit ints) and the kernel never cast. `knn.py` and `kernels.md` claimed fp32 (the fp32 in the I/O table was the buffer).

Gate 2, the boundary (624 of 9,859 rows differ in top-100):
| measurement | value |
|---|---|
| swapped pairs | 626 (622 rows one pair, 2 rows two) |
| torch-only item in the true top-100 | 581 / 626 |
| triton-only item in the true top-100 | 48 / 626 |
| kernel's own scores rank its choice >= the dropped one | 626 / 626 (epilogue correct) |
| true gap of a swapped pair | max 0.0049, median 0.00089 |
| kernel abs error on swapped items | max 0.0062, median 0.0010 |
| rank-100 -> 101 gap, all rows | median 0.0068, p10 0.00098, p1 9.3e-5; 63 % under 0.01, 10 % under 0.001 |
| in fp16 ulps of the boundary score (500-row sample) | median 18 ulp, 4.8 % under 1 ulp; candidates within 1 ulp: median 1, max 3 |
One ulp of the boundary is the wrong yardstick for Triton (error set by ~10-30 partial sums, ulp 0.008-0.016); right for torch. torch's residual: a probe-only fp32-accumulating kernel disagrees with torch on 124 rows (jaccard 0.99975), all 124 swapped pairs exact fp16 ties in torch's scores (true gap < 1 ulp: max 0.978, median 0.29).

fp32 variant measured, not shipped (do_bench, 500 reps, 1410 MHz): kernel B=1 P=195,023 0.05235 vs 0.05226 ms; B=16 P=442,864 0.9636 vs 0.9408 ms; `LiNRV2` eager triton / torch 0.471 / 1.515 ms (B=1), 3.335 / 18.30 ms (B=16).

Proposed wording for H WP-4 clause (4): "exact" describes candidate selection, which must be identical (`counts` and ids `torch.equal`), not the arithmetic; top-k may differ by pairs whose true gap is below the accumulation error; a miss is attributed by the probe, not waived.

Gates: ruff clean (76 files); links 0; full suite 645 passed, 0 failed, 0 skipped in 117 s. Changed: docs only (`modules/knn.py` precision contract, `kernels.md` score conventions and an accumulation paragraph). Unverified: only `c0_genre` on goodreads; bloom, arxiv (unit-norm) and V3 stage 2 (same kernel) not measured; L4-b / L4-c left to the harness worker (run at C5).
