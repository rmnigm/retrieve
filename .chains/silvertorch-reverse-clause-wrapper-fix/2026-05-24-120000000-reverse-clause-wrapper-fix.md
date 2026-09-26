---
chain: "silvertorch-reverse-clause-wrapper-fix"
branch: "main"
nextStep: "None. Landed in cc85d8f; the harness this fixed was deleted at roadmap C3 and harness v2 passes clause_is_reverse through bench.algos.build. The goodreads reverse-clause reruns are part of the D1 campaign."
created: "2026-05-24T12:00:00Z"
---

# Fix SilvertorchAlgo to propagate clause_is_reverse (archived plan)

Source: `docs/plans/archive/silvertorch-reverse-clause-wrapper-fix.md`, written 2026-05-24, landed in commit `cc85d8f`, archived afterwards.

## 1. Primary request and intent
The old harness wrapper `SilvertorchAlgo` (`evaluation/retrieval/algos/silvertorch.py`) called `SilverTorch.register_index(item_embs, item_clause_attrs=...)` without `clause_is_reverse=`. The buffer defaulted to all-zeros, so the goodreads reverse clause c1 ("language != query language") was evaluated as a positive equality. SilverTorch Recall@100 on `c1_lang_reverse`, `c0c1`, `all4` collapsed to 0.0005 to 0.0425 at d64 / d128 / d256. Three of the six goodreads filter sweeps were unusable for SilverTorch.

## 2. Key technical concepts
- The library already supported reverse clauses: the exact-clause kernel XORs `clause_is_reverse[c]` per clause, and `register_index` accepts the tensor.
- The driver loaded `clause_is_reverse` from disk and passed it to `build_filter` (the LiNR algos were correct) but dropped it before `build_algorithm`.
- Bloom mode is forward-only by design; the library raises on `clause_is_reverse` with bloom, and bloom configs never include reverse sweeps.

## 3. Work completed
- Harness-only wiring change: `SilvertorchAlgo.__init__` gained `clause_is_reverse`; `build_algorithm`, `_build_filter_modules` (fifth tuple element), `run_filter_kind`, `run_one_sweep`, `evaluate_cell`, `_try_build_algo` each forward it.
- Stale docstring removed ("silvertorch on reverse-clause sweeps is skipped by the driver", which was never true).
- New regression test `evaluation/retrieval/tests/test_silvertorch_algo_reverse.py`: one reverse clause, `n_probe = n_lists` so IVF is exact, top-K set-equal to a brute-force filtered scan on `triton` and `torch`.

## 4. Decisions and rationale
- No library, kernel, config, dataset or oracle change: the defect was entirely in the wrapper.
- Out of scope, tracked separately: the stale goodreads d128 / d256 oracle cache (the fix is necessary but not sufficient until the cache is rebuilt), arxiv (no reverse clauses, already correct).

## 5. Verification plan as written
1. library `test_silvertorch.py` green; 2. the new wrapper test green on both backends; 3. `algo.idx.clause_is_reverse[0]` is True; 4. rerun goodreads d128 filter after deleting `gt_topk_*.pt`: `linr_v1_filter_mask` on `c1_lang_reverse` Recall@100 >= 0.99, SilverTorch >= 0.85 and within 5 % of `linr_v1_filter_mask`; same for `c0c1`, `all4`; 5. repeat at d64 / d256; 6. re-upload to the HF results repo.

## 6. Unresolved at archive time
The rerun (steps 4 to 6) and the thesis results-chapter edits were tracked as roadmap "Stage 4b item 7", later folded into D1.
