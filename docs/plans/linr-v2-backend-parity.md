# LiNR V2 — why do two implementations of an exact algorithm disagree?

> **Status:** planned 2026-09-15 on `development` at `c1ae1b5`; **§2 executed
> 2026-09-15** on `dev/l4-linr-v2-parity` off `ea1243f` — record in §6:
> candidate sets identical, divergence is fp16 accumulation in the Triton
> kernel plus fp16 output rounding on the `torch` path, nothing changed but the
> docs that misstated it. §4's L4-b / L4-c are the harness worker's, not run
> here. Written by the orchestrator from C4's run
> ([evaluation-harness-v2.md](evaluation-harness-v2.md) §12). User decision the
> same day: the divergence is settled as a correctness question in its own
> right, before D1 — V2 is the paper's exact baseline.
>
> Question this plan answers: *`linr_v2` selects the exact top-K of an exactly
> filtered candidate set. Why do the `torch` and `triton` backends return
> different items, and is either of them wrong?*
>
> **Ordering authority:** [00-roadmap.md](00-roadmap.md) §1 Phase L, step
> **L4**.

## 1. The evidence

| where | measurement |
|---|---|
| C4, goodreads-d128 `c0_genre` | `jaccard_vs_first@100` **0.998743**, `score_max_abs_diff` **9.77e-3** |
| the golden, independently, on the *old* harness | `recall@100` torch **0.99969470** vs triton **0.99927376** — **4.2e-4** |
| the control, same cells | `linr_v1` and `linr_v4` torch-vs-triton: **exactly 0.0** |
| SilverTorch, same run | `torch` vs `triton` **exactly 1.0**, `score_max_abs_diff 0.0`, both datasets |

Two harnesses, built years apart in this project's history, reproduce the same
disagreement; each reproduces *its own* number to 1e-11. So it is in the
library, it predates the v2 rewrite, and it is specific to V2.

## 2. The one question that decides everything

**Do the two backends score the same candidate set?**

L3 made `clause_compact` deterministic *and* ascending, `torch.equal` to
`ops.reference.clause_compact` — so after L3 the candidate sets must be
identical, and if they are not, that is a straightforward bug with a
straightforward test. Establish this first, on one query, before any theory:
dump `candidate_ids` and `counts` from both backends and compare.

- **If the sets are identical**, the divergence is in the scoring: `triton`'s
  `fused_masked_knn_topk` against `torch`'s cuBLAS path, over fp16 embeddings.
  Then the question becomes *which accumulation each performs* (fp16 vs fp32,
  and in what order), whether `score_max_abs_diff` 9.77e-3 is consistent with
  that width over `d = 128`, and whether the differing items are **ties or
  near-ties at the rank-100 boundary** — measure the score gap at the boundary
  and the number of items within one fp16 ulp of it. This is precision, it is
  documented, and the clause in H WP-4 that demands `jaccard == 1.0` for "exact
  algos" is the thing that is wrong: *exact* describes the filtering, not the
  arithmetic.
- **If the sets differ**, stop and report. That is a bug in one of the two
  paths and it invalidates a paper baseline.

## 3. What a fix may and may not do

If the answer is precision, the **default is to change nothing in the kernels**
and to write it down instead: both backends are correct implementations of
the same algorithm at fp16, and the paper reports the backend it measured.
Changing the Triton kernel's accumulation width to match cuBLAS would move
every V2 latency and quality number in the campaign, and it must not be done
as a side effect of this investigation — bring it back with the numbers.

If the answer is a genuine disagreement in candidate selection, fix the
defective path, and pin it with a parity test in the shape of L3's
`test_compact_order.py` — the property stated as a test.

## 4. Two items inherited from C4

- **L4-b — the decisive `linr_v4` experiment.** C4 attributed `linr_v4`'s
  7.3e-5 quality miss to query-batch chunk shape (v2 chunks at 16, the golden
  at 64; `PostfilterKNNInt8` pads below `_PAD_M = 17`, and int32 `>>5` → fp16
  makes boundary ties). Three alternatives were killed by measurement; the
  confirmation was not run. Run one `linr_v4` cell at chunk 64: if it
  reproduces the golden exactly, the attribution is proven.
  **Run at C5 (2026-09-15,
  [evaluation-package-layout.md §11.2](evaluation-package-layout.md)): it does not.**
  v2 at chunk 64 lands 2.9e-4 from the golden at `recall@100` — four times further than
  chunk 16's 7.3e-5 — and moves `k = 500 / 1000` by 2.3–2.6e-4 where chunk 16 was within
  1e-5, all downward. Batch shape moves `linr_v4` (the probe was right), but the golden is
  not "v2 at 64"; what the old harness's quality pass did to the batch is the open
  question, and it needs the golden worktree. `QUALITY_CHUNK` stays 16.
- **L4-c — the `unstable` artifacts.** `clocks_drift` compares a
  start-of-process *idle* sample against under-load samples, so it fires on the
  GPU boosting (10 of C4's 18 flags). `clocks_locked` means "within 2 % of
  `expected_sm_mhz`" and read `true` on a box that cannot lock clocks at all —
  the opposite of what H §8.2 F wanted. Both are harness-side and small; they
  belong with this step only because C4 found them.

## 5. Gate

1. The candidate-set question answered with a dumped comparison, not an
   argument.
2. If precision: the boundary-tie measurement (score gap at rank 100, items
   within one fp16 ulp), and the H WP-4 clause (4) wording corrected to
   distinguish exact *filtering* from bit-identical arithmetic.
3. If a bug: the defective path fixed, a parity test pinning it, full library
   suite green (645 at `c1ae1b5`), every existing parity file still bit-exact.
4. L4-b run; L4-c fixed or explicitly deferred with a reason.
5. Record in §6 either way. A "no bug found" result **is** a result and is
   written down with its evidence.

## 6. Validation record

### 6.1 L4 — 2026-09-15, A100-SXM4-80GB, `dev/l4-linr-v2-parity`

**Verdict.** The two backends score the **same candidate set** on every one
of the 9 859 kept rows; the divergence is arithmetic, and it decomposes into
two measured precision effects with nothing left over: the Triton kernel
**accumulates in fp16** (contrary to what `knn.py` and `kernels.md` said), and
the `torch` path rounds its fp32-accumulated score to fp16 on output. No
selection bug; **no kernel changed**; the docs that misstated the precision
are corrected in this commit.

**Environment.** The A100 box (driver 580.159.04, nvcc 12.4), Python 3.11,
torch 2.10.0+cu128, triton 3.6.0, ruff 0.15.6 via `uvx`, `/venvs/l4` from
`uv sync --extra official --all-packages`. Branch point `development` @
`ea1243f`; worktree `/workspace/wt/l4`; private inductor cache
`/tmp/inductor-l4`. Every CUDA job ran under `flock /workspace/gpu.lock`.
SM clock sampled at 1410 MHz before and after every timing (clocks cannot be
locked here, H §7). Artifacts under
[linr-v2-backend-parity-artifacts/](linr-v2-backend-parity-artifacts/):
`probe_v2_parity.py` (+ `.json`, `.log`, the compiled kernel's
`fused_masked_knn_topk_fp16.ptx`), `boundary_and_fp32_variant.py` (+ `.json`,
`.log`), `torch_side_rounding.py` (+ `.json`, `.log`), `library-suite.log`.

**Inputs are the C4 cell's.** goodreads d128, the SASRec cache
`encoded_queries_v2.pt` (797 084 items, first 10 000 users), `item_attrs_narrow`
with the legacy pad row dropped, `c0_genre` (clause 0 live, the other three
`-1`, 141 all-`-1` rows skipped → 9 859 kept), chunks of 16, `k_max = 1000`,
`LiNRV2(ExactAttributeFilter(backend), backend)` built exactly as
`algos.build` does. Proof that they are: the probe reproduces C4's §12.6
numbers to the digit — `jaccard_vs_first@100` **0.9987427** (C4 0.998743),
`score_max_abs_diff` **0.009765625** (C4 9.766e-3) — and both backends are
deterministic on repeat (`torch.equal` on ids and scores, both).

**Gate 1 — the candidate-set question, by dumped comparison.** For every
chunk, `filter.evaluate_indices` on the `torch` module (`ops.reference.clause_compact`)
and on the `triton` module (`ops.triton.clause_compact`), `counts` compared
`torch.equal` and ids compared `torch.equal` after both tails past `counts[b]`
are set to `-1`:

| | rows |
|---|---|
| rows with different `counts` | **0** / 9 859 |
| rows with different candidate ids | **0** / 9 859 |

The sets are identical. So the question is scoring, and §2's first branch
applies.

**What each backend computes (chunk 0, `P = 442 864`, all candidates, against
an fp64 dot of the same fp16 inputs).** Scores here run to `|s| = 31`
(median `|s|` 12) because goodreads queries have norm ≈ 20; the *top* scores
sit near 0 to −1, so the running sums are two orders larger than the answer.

| path | max abs err | mean abs err | note |
|---|---|---|---|
| `torch`: cuBLAS `bmm`, fp16 out | 0.00782 | 0.00203 | ≤ **1.0 ulp** of the fp16 result vs `fp16(fp64)`; = pure output rounding |
| `triton`: shipped kernel | **0.0276** | **0.00336** | every score it writes is an fp16 value (`scores.half() == scores` on all of them) |
| fp16 *sequential* model | 0.155 | 0.0136 | worse than the kernel: its reduction is a tree |
| fp32 sequential model | 1.6e-5 | 1.6e-6 | what "fp32 accumulate" would give |

The compiled PTX (`fused_masked_knn_topk_fp16.ptx`) has **8 × `add.f16`, 0
f32 adds/FMAs/muls, 1 `cvt.f32.f16`** — the kernel multiplies fp16 × fp16 and
reduces in fp16, then widens once to store. Cause: `tl.sum` keeps the input
dtype (Triton 3.6.0 `standard._pick_sum_dtype` promotes only sub-32-bit
ints), and the kernel never casts. `knn.py`'s "accumulate dots in fp32 …
and the fused Triton kernel" and `kernels.md`'s "plain dot product, fp32"
were wrong; the fp32 in `kernels.md`'s I/O table is the *buffer*.

**Gate 2 — the boundary (every row whose top-100 differs, 624 of 9 859;
`boundary_and_fp32_variant.json`).**

| measurement | value |
|---|---|
| swapped pairs | 626 (622 rows swap exactly one pair; 2 rows two) |
| `torch`-only item is in the true top-100 | 581 / 626 |
| `triton`-only item is in the true top-100 | 48 / 626 |
| kernel's own scores rank the item it returned ≥ the one it dropped | **626 / 626** (the epilogue is correct; the swap is entirely score error) |
| true gap between a swapped pair | max **0.0049**, median 0.00089 |
| kernel abs error on the swapped items | max **0.0062**, median 0.0010 |
| rank-100 → 101 gap, all 9 859 rows | median **0.0068**, p10 0.00098, p1 9.3e-5; 63 % of rows under 0.01, 10 % under 0.001 |
| in fp16 ulps *of the boundary score* (500-row sample) | median gap 18 ulp, 4.8 % of rows under 1 ulp; candidates within 1 ulp of rank 100: median 1, max 3 |

Every swap is a pair whose true separation (≤ 0.0049) is inside the kernel's
error on that pair (≤ 0.0062, itself inside the 0.028 whole-candidate
maximum). "Within one fp16 ulp of the boundary" is the wrong yardstick for the
Triton side — the boundary scores are ~0.1–1 where an ulp is 6e-5–1e-3, while
the error is set by the ~10–30 partial sums where an ulp is 0.008–0.016;
that is why 483 of the 1 252 differing items sit "beyond one ulp of the
boundary" yet all are inside the accumulation error. It *is* the right
yardstick for the `torch` side, below.

**The `torch` side's residual, measured (`torch_side_rounding.json`).** A
probe-only copy of the kernel that casts to fp32 before the multiply
(`_kernel_fp32_acc`, artifacts only) disagrees with `torch` on **124** rows
(`jaccard@100` 0.99975). On all 124 swapped pairs the `torch` path's fp16
scores are **exactly equal** (124/124 ties), the pair's true gap is < 1 fp16
ulp of the boundary (max 0.978 ulp, median 0.29), and `torch.topk` broke the
tie. That is the whole residual: fp16 *output* rounding creating ties, on
the 4.8 % of rows whose rank-100 gap is under an ulp.

**The fp32-accumulating variant, measured but not shipped (§3).** `do_bench`
medians, 500 reps, at 1410 MHz, on the cell's real shapes:

| launch | shipped fp16-acc | probe fp32-acc |
|---|---|---|
| kernel `B = 1, P = 195 023` | 0.05235 ms | 0.05226 ms |
| kernel `B = 16, P = 442 864` | 0.9636 ms | 0.9408 ms |
| `LiNRV2` eager forward, `triton` / `torch` | 0.471 / 1.515 ms (B=1), 3.335 / 18.30 ms (B=16) | — |

The kernel is bandwidth-bound on the gather; the cast costs nothing
measurable (the fp32 launch reads as 2 % faster at B = 16, inside noise).
Parity would move from `jaccard@100` 0.998743 / 624 rows to **0.999751 / 124
rows** against `torch`, and the remaining 124 are `torch`'s own ties. Nothing
in this record changes the shipped kernel: per §3 and the orchestrator's brief
that is a decision to take with these numbers, and it is cheaper now than
later — no campaign has run yet (D1 follows L4), so no result would be
invalidated by taking it before D1.

**Proposed wording for H WP-4 clause (4)** (for the orchestrator to apply;
the current text conflates exact *filtering* with bit-identical arithmetic):

> (4) `jaccard_vs_first@100 == 1.0` for `torch` vs `triton` on the exact
> algos — "exact" describes the candidate *selection*, which must be
> identical between backends (`counts` and candidate ids `torch.equal`),
> not the arithmetic: the two backends score fp16 inputs at different
> accumulation widths, so their top-k lists may differ by pairs whose true
> score gap is below that width's error (L4 §6: 0.0049 max, all at the
> rank-100 boundary). A miss on this clause is attributed by the L4 probe
> (candidate sets dumped equal, swaps inside the measured error), not
> waived. On `silvertorch`, `cuda`/`cute` vs `triton` as before.

**Gates — CPU.**

| gate | result |
|---|---|
| `uvx ruff@0.15.6 check retrieve` / `format --check retrieve` | clean (76 files) |
| `python3 scripts/check_doc_links.py` | 0 broken links |

**Gates — GPU (under the lock).**

| gate | result |
|---|---|
| 1. candidate-set question, dumped | identical, 0 / 9 859 rows differ on `counts` or ids (above) |
| 2. boundary-tie measurement + clause wording | above |
| 3. (bug branch) | not applicable — no bug; no kernel or test changed |
| full library suite | **645 passed, 0 failed, 0 skipped** in 117 s (`library-suite.log`); same count as `c1ae1b5`; every parity file and tolerance untouched |

**Changed in this commit.** Documentation only: `modules/knn.py`'s precision
contract (module and `PrefilterKNN` docstrings) and `docs/system/kernels.md`
(score conventions; a new accumulation-width paragraph under
`fused_masked_knn_topk`), both to state what the code does (CLAUDE.md rule 4).
The kernel file is untouched so the diff carries no behaviour change.

**Skipped / unverified.** L4-b and L4-c belong to the harness worker and were
not run here. Only the `c0_genre` sweep on goodreads d128 was probed; other
sweeps, `bloom` cells, arxiv (unit-norm inputs, so a different error
regime) and V3's stage 2 (the same kernel, the same width) were not
measured. The fp32-accumulating kernel exists only in the artifacts. The
roadmap checkbox and the merge are the orchestrator's.


## 7. L5 — the fix, and the audit the fix implies

> **Decided 2026-09-15 by the user, on §6.1's numbers.** Ship fp32
> accumulation, and sweep the other kernels for the same defect. The plan's §3
> default ("change nothing in the kernels") is **overridden**: it assumed a
> real trade-off, and there is none.

### 7.1 Why this is not the trade-off §3 imagined

| | shipped (fp16 acc) | fp32 acc |
|---|---|---|
| kernel, B=1, P=195 023 | 0.05235 ms | **0.05226 ms** |
| kernel, B=16, P=442 864 | 0.9636 ms | **0.9408 ms** (2.4 % *faster*) |
| max abs score error vs fp64 | **0.0276** | ~1e-5 |
| `linr_v2` torch-vs-triton jaccard@100 | 0.998743 (624 rows) | **0.999751** (124 rows) |
| the residual 124 rows | — | `torch`'s *own* fp16 ties, broken by `torch.topk` |

Latency is unchanged or better (the kernel is gather-bound), accuracy improves
by ~3 orders, and **no campaign has run**, so nothing is invalidated. The
documented contract already *said* fp32 (`knn.py`, `kernels.md`): the code
contradicted its own specification, and L4 corrected the docs to match the
code. This step corrects the code instead, and restores the docs.

### 7.2 The audit is the point

`tl.sum` inherits its operand dtype (Triton 3.6.0 promotes only sub-32-bit
ints), so **every fp16 × fp16 reduction in this library has this defect by
default**, and our parity suite is structurally blind to it: it compares
`ops.triton` against `ops.reference` at the *same* input dtype, so both sides
carry the same error and agree. The bug was found only because a *third*
implementation — cuBLAS, in the `torch` backend — accumulates differently.

Hence: sweep every Triton kernel for accumulation width, and add a parity test
against an **fp64 oracle**, not only against `ops.reference`. The instance
matters less than the detection gap.

### 7.3 Work package and gate

**WP-1 — fp32 accumulation and the precision audit (GPU).** `fable`, branch
`dev/l5-fp32-accumulation` off `development`.

1. `fused_masked_knn_topk` accumulates in fp32; scores keep their current
   output dtype and the op schema is unchanged (op names, buffer names and
   `KernelTuneSpec` keys stay valid).
2. **Audit every kernel in `ops/triton/` for reduction width** — read each
   `tl.sum` / `tl.dot` / accumulator and record its dtype in a table in the
   record, including the ones that are correct and why (integer paths:
   `codesigned_probe_score`'s int8 → int32, `oporp_1bit_match_topk`'s
   popcount). Fix what is wrong by the same rule; **if a fix would change
   SilverTorch's numbers, stop and tell the orchestrator before shipping it** —
   B3's head-to-head and the `official` parity gate both rest on that kernel.
3. **A parity file against an fp64 oracle** (`tests/parity/test_accumulation.py`
   or similar): for each scoring kernel, the max abs error against an fp64
   computation of the same operation on the same inputs, asserted under a bound
   that fp32 accumulation meets and fp16 does not. This is the test that would
   have caught the defect; it is the deliverable that outlasts the fix.
4. Restore the precision statements in `modules/knn.py` and
   `docs/system/kernels.md` (L4 corrected them *downward* to match the code;
   they go back to fp32, now truthfully).
5. Re-measure `linr_v2` torch-vs-triton parity and report the new jaccard, and
   re-measure the two kernels' `do_bench` cost.

Gate: full library suite green (**645** at `d9a3200`), every existing parity
file bit-exact with tolerances untouched, the new fp64 parity file green, the
audit table complete, and the cost table showing no regression.

**WP-2 — delete the compatibility shim (CPU; same branch).** `retrieve.layers`
and `retrieve.kernels` were temporary tooling for the old-harness golden
worktree (L D10). The golden stopped being a gate on 2026-09-15, so the shim's
reason to exist is gone: delete both modules and the `KMeansTorch` /
`build_silvertorch` aliases, and drop the paragraph that promised them in
`retrieve/docs/` and `docs/system/architecture.md`. Gate: suite green, `git
grep -l "retrieve.layers\|retrieve.kernels"` clean outside `docs/plans/`
history, links 0.


## 8. Validation record — L5

### 8.1 L5 — 2026-09-15, A100-SXM4-80GB, `dev/l5-fp32-accumulation`

**Verdict.** `fused_masked_knn_topk` now widens both operands to fp32 before
the multiply and reduces in fp32; the two backends of `linr_v2` agree on
every row except the 124 where the `torch` side's fp16 output rounding
creates an exact tie. The audit found **no other floating-point reduction
in `ops/triton/`**: every remaining `tl.sum` / `tl.dot` / `tl.reduce` is an
integer or boolean path, so no SilverTorch number moves and no second fix
was needed. The compatibility shim (`retrieve.layers`, `retrieve.kernels`)
is deleted.

**Environment.** The A100 box (driver 580.159.04, nvcc 12.4), Python 3.11,
torch 2.10.0+cu128, triton 3.6.0, ruff 0.15.6 via `uvx`, `/venvs/l5` from
`uv sync --extra official --all-packages`. Branch point `development` @
`81c55d3`; worktree `/workspace/wt/l5`; private inductor cache
`/tmp/inductor-l5`. Every CUDA job ran under `flock /workspace/gpu.lock`. SM
clock sampled at 1410 MHz before and after every timing (idle sample at
process start 1155 MHz; clocks cannot be locked, H §7). Artifacts under
[linr-v2-backend-parity-artifacts/l5/](linr-v2-backend-parity-artifacts/l5/):
`parity_and_cost.py` (+ `.json`, the shipped kernel's
`fused_masked_knn_topk_fp32.ptx`); `parity_and_cost.log` and
`library-suite.log` sit beside them in the worktree but are gitignored
(`*.log`, as L4's were). Inputs are the L4
cell's, via `../probe_v2_parity.py::load / build` (goodreads d128, SASRec
cache, first 10 000 users, `c0_genre`, 9 859 kept rows, chunks of 16,
`k_max = 1000`).

**WP-1 item 1 — the fix.** Two `.to(tl.float32)` on the loads in
`_fused_masked_knn_topk_kernel`; nothing else in the file changes — op
schema, buffer dtypes, `FusedMaskedKnnTopkConfig` and the `KernelTuneSpec`
entry are untouched. An fp16 × fp16 product is exact in fp32, so the only
error left is the fp32 tree reduction. The compiled PTX (D = 128,
`fused_masked_knn_topk_fp32.ptx`):

| body | `add.f16` | `add.f32` | `fma.rn.f32` | `mul.f32` | `cvt.f32.f16` |
|---|---|---|---|---|---|
| pre-L5 (fp16 acc; the L4 PTX) | 8 | 0 | 0 | 0 | 1 |
| shipped (fp32 acc) | **0** | 8 | 14 | 2 | 24 |

**WP-1 item 2 — the audit.** Every reduction and accumulator in
`retrieve/src/retrieve/ops/triton/`, read from the source:

| file | reduction | operand dtype | accumulates in | verdict |
|---|---|---|---|---|
| `fused_masked_knn_topk.py` | `tl.sum(emb_rows * q, axis=1)` | fp16 or fp32 inputs, **cast to fp32 before the multiply** | fp32 | **fixed here** (was: operand dtype, fp16 in production) |
| `codesigned_probe_score.py` | `tl.dot(q_codes[None, :], codes.T, out_dtype=tl.int32)` | int8 × int8 | int32, exact (`|dot| ≤ 127² · D`, < 2²⁴ for `D ≤ 1024`, so the `.to(tl.float32)` is exact too) | correct |
| `codesigned_probe_score.py` | `tl.sum(dots_2d, axis=0)` (squeeze of the length-1 M axis) | int32 | int32 | correct |
| `codesigned_probe_score.py` | `dots_i32.to(tl.float32) * q_scale * global_scale` | fp32 scalars | two fp32 multiplies, no reduction (≤ 2⁻²³ rel. each) | correct; measured 1.1e-7 rel vs fp64 |
| `codesigned_probe_score_exact.py` | same three lines as above | int8 / int32 / fp32 | int32, then fp32 dequant | correct |
| `oporp_1bit_match_topk.py` | `tl.sum(pop_words, axis=1)` | int32 (from `popcount_int64`) | int32; `(D_TOTAL - 2 * hamming).to(tl.float32)` exact for `D_TOTAL < 2²⁴` | correct; `torch.equal` to the int64 truth |
| `common.py::popcount_int64` | SWAR on int64 lanes | int64 | int64 → int32 | correct, bit-exact twin of `functional.popcount_int64` |
| `common.py::bloom_subset_pass` | `tl.reduce(diff, axis=1, combine_fn=or_combine)` | int64 | int64 OR | boolean; no arithmetic |
| `common.py::clause_pass` | `\|`, `&`, `^` over `int1` | int1 | int1 | boolean; no arithmetic |
| `common.py::compact_store` | `tl.cumsum(pass_int, axis=0)` | int32 | int32 | integer rank; exact |
| `common.py::compact_stash` | `tl.sum(tl.where(pass_mask, 1, 0).to(tl.int32))` | int32 | int32 → stored int64 | integer count; exact |
| `bloom_match.py`, `bloom_compact.py`, `clause_mask.py`, `clause_compact.py` | none of their own — they call the `common.py` helpers above | — | — | no reduction |
| `compact_scatter_kernel`, `_host.py` | no reductions (`torch.cumsum` on the host over int64 tile counts) | int64 | int64 | exact |

The rule that makes the integer rows safe: Triton's `_pick_sum_dtype`
promotes sub-32-bit ints to int32 and leaves int32 / int64 alone, and every
integer path here is either already int32 or bounded well inside it. The
only floating-point reduction in the tree was the one fixed. **No
SilverTorch number moves** (the int8 kernels are untouched), so the
"stop and ask" clause of §7.3 item 2 was not triggered.

**WP-1 item 3 — the fp64 parity file.**
[`tests/parity/test_accumulation.py`](../../retrieve/tests/parity/test_accumulation.py),
8 tests, one per scoring kernel × dimension: asks for every candidate
(`k = P`), maps the returned ids back to an fp64 computation of the same
operation on the same inputs, and asserts a bound *and* that an fp16 model
of the same computation on the same inputs misses it — so the bound's
discriminating power is checked on every run. Goodreads-like inputs
(unnormalised fp16, `|score|` up to ~140):

| kernel | measured error vs fp64 | bound | the fp16 model on the same inputs |
|---|---|---|---|
| `fused_masked_knn_topk`, D ∈ {64, 128, 256} | 7.5e-6 / 9.7e-6 / 1.4e-5 max abs | `1e-4` abs | fp16 tree reduction of the fp16 products: 6.5e-2 / 9.0e-2 / 1.4e-1 |
| `codesigned_probe_score`, D ∈ {64, 128} | 1.1e-7 max rel | `1e-6` rel | fp16 cannot hold the int32 dot (`|dot|` ~1e5 > 65 504 → overflow) |
| `codesigned_probe_score_exact`, D ∈ {64, 128} | 1.0e-7 max rel | `1e-6` rel | same |
| `oporp_1bit_match_topk_indirect`, W = 4 | 0 | `torch.equal` | — (integer) |

Run against the pre-L5 kernel body (the fix stashed, same inputs), this
file fails at every D: 7.2e-2 / 8.9e-2 / 1.4e-1 max abs against the 1e-4
bound — the test that would have caught it.

**WP-1 item 4 — docs.** `modules/knn.py` (module and `PrefilterKNN`
docstrings) and `docs/system/kernels.md` (score conventions; the
accumulation paragraph under `fused_masked_knn_topk`) state fp32 again,
now with the reason it must be explicit; `docs/system/testing.md` lists the
new file.

**WP-1 item 5 — parity and cost, re-measured (`parity_and_cost.json`).**
`linr_v2` `torch` vs `triton`, all 9 859 kept rows:

| | L4 (fp16 acc) | **L5 (fp32 acc)** |
|---|---|---|
| `jaccard@100` | 0.998743 | **0.999751** (predicted 0.999751) |
| `jaccard@500` / `@1000` | — | 0.999500 / 0.999379 |
| rows whose top-100 differ | 624 | **124** |
| … of which the `torch` scores of the swapped pair are exactly equal | — | **124 / 124** |
| `score_max_abs_diff` (`torch` fp16 out vs kernel fp32) | 0.009766 | **0.003904** (= 1 fp16 ulp at `|s|` ∈ [2, 4)) |

Against an fp64 dot of the same fp16 inputs over all `P = 442 864`
candidates of chunk 0 (`|s|` max 31.4), both bodies in the same process:

| body | max abs | mean abs |
|---|---|---|
| pre-L5 fp16 acc | 0.027644 (L4: 0.0276) | 0.003362 |
| **shipped fp32 acc** | **3.2e-6** | 4.0e-7 |

`do_bench` medians (500 reps, warm-up 100), both bodies in the same
session, SM 1410 MHz sampled before and after each:

| launch | pre-L5 fp16 acc | **shipped fp32 acc** | L4's numbers (fp16 / fp32 probe) |
|---|---|---|---|
| kernel `B = 1, P = 195 023` | 0.05315 ms | **0.05302 ms** | 0.05235 / 0.05226 |
| kernel `B = 16, P = 442 864` | 0.9631 ms | **0.9404 ms** (−2.4 %) | 0.9636 / 0.9408 |
| `LiNRV2` eager forward `triton` | — | 0.497 ms (B=1), 3.271 ms (B=16) | 0.471 / 3.335 |
| `LiNRV2` eager forward `torch` | — | 1.518 ms (B=1), 18.28 ms (B=16) | 1.515 / 18.30 |

No regression: the kernel is gather-bound and the cast is free (the
B = 16 launch is 2.4 % faster, as L4's probe measured; B = 1 is inside
noise).

**WP-2 — the shim.** `retrieve/src/retrieve/layers/__init__.py` and
`retrieve/src/retrieve/kernels/__init__.py` deleted (with them the
`KMeansTorch` alias, `build_silvertorch`, and the `retrieve.layers.{filters,
silvertorch,linr,linr.postfilter_knn_int8}` / `retrieve.kernels.silvertorch
.official` module aliases). The paragraphs promising them in
`retrieve/README.md`, `retrieve/docs/getting-started.md` and
`docs/system/architecture.md` (three sites, including the move table's
header) are gone; the move table keeps the 0.1 *file* names as history.
`git grep -l "retrieve\.layers\|retrieve\.kernels" -- ':!docs/plans'` is
empty. Nothing in `evaluation/` or `tests/` imported the shim.

**Gates — CPU.**

| gate | result |
|---|---|
| `uvx ruff@0.15.6 check retrieve` (+ the L5 artifact script) / `format --check retrieve` | clean (75 files) |
| `python3 scripts/check_doc_links.py` | 0 broken links |
| shim grep (above) | clean |

**Gates — GPU (under the lock).**

| gate | result |
|---|---|
| full library suite | **653 passed, 0 failed, 0 skipped** in 123 s (`library-suite.log`) = 645 at `d9a3200` + the 8 new tests; 16 warnings, all pre-existing |
| every existing parity file, tolerances untouched | in the suite; `test_fused_masked_knn_topk.py` 11 passed on its own first — its inputs are fp32, where the cast is a no-op, so it is bit-for-bit the same arithmetic as before |
| `tests/parity/test_accumulation.py` | 8 passed |
| audit table | complete (above) |
| cost table | no regression (above) |

**Changed in this commit.** `ops/triton/fused_masked_knn_topk.py` (the two
casts, the dtype comment); `modules/knn.py` (docstrings);
`tests/parity/test_accumulation.py` (new); `docs/system/{kernels,testing,
architecture}.md`; `retrieve/README.md`, `retrieve/docs/getting-started.md`;
`layers/` and `kernels/` deleted; this record; the `l5/` artifacts.

**Skipped / unverified.** Only the `c0_genre` sweep on goodreads d128 was
re-measured; arxiv (unit-norm inputs) and the `bloom` cells were not, but
the fp64 file bounds the kernel on unnormalised inputs harder than either.
V3's stage 2 goes through the same kernel and inherits the fix unmeasured.
The `torch` backend's fp16 output rounding is left as is — it is cuBLAS's
`bmm` output dtype, and the 124 residual rows are exact ties it creates,
not an error of ours. The roadmap checkbox and the merge are the
orchestrator's; B3 and D1 were not touched.
