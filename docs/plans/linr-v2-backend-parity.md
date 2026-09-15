# LiNR V2 — why do two implementations of an exact algorithm disagree?

> **Status:** planned 2026-09-15 on `development` at `c1ae1b5` (nothing
> implemented). Written by the orchestrator from C4's run
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

*(appended when L4 runs.)*
