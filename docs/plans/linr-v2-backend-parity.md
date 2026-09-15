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
