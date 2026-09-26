---
chain: "q3-library-gates"
branch: "main"
nextStep: "Orchestrator: review commit c5e654e (+ this note's commit) on dev/q3, merge into staging and push, remove Q3 from docs/roadmap.md, and fold the findings below (pre-existing evaluation E501; the thesis-side 1.4 TB/s claim; the boundary-tie limit of assert_ids_equal_up_to_ties) into Q2/Q4/F-phase as fits."
created: "2026-09-26T00:26:59Z"
---

# Q3: library correctness gates hardened — suite green, nothing newly red

## Primary request
Roadmap Q3 worker brief: strengthen the library gates inside existing test files (tests/ + docs only, nothing under retrieve/src/retrieve/), report every red test as a finding, never loosen a tolerance; measure HBM copy bandwidth + launch floor. Worker: no push, no merge, no roadmap edit.

## Result
- `cd retrieve && uv run --no-sync pytest tests/ -x -q` → **708 passed** (baseline before Q3: 653 passed). **No test is red.**
- `ruff check retrieve` / `ruff format --check retrieve` clean; `python3 scripts/check_doc_links.py` 0 broken.
- Import resolution (orchestrator's warning): the venv's editable install points to /workspace/retrieve/retrieve/src; `diff -r` against /scratch/wt/q3/retrieve/src/retrieve = identical, checked before and after the final run. The main checkout has no changes under retrieve/src.
- Commit c5e654e on dev/q3 (local, not pushed).

## What changed (tests)
- `tests/parity/conftest.py`: new `assert_scores_match` (exact non-finite pattern incl. NaN and the sign of each inf; finite values within tol; prints count, first slots, both values), `assert_topk_equal` (bit-exact: scores at zero tol + ids up to ties), `poison_empty` + `POISON`; `assert_topk_matches` takes keyword-only atol/rtol with no default and no longer maps non-finite scores to 0. `tests/conftest.py::assert_topk_id_sets_match`: no default tol.
- Every call site now states its tolerance. Exact (`torch.equal`) paths: OPORP/SimHash parity + cross-backend, both probe scorers (+ bloom, exact), SilverTorch torch vs triton, mask-all-True ≡ unmasked, OneBit all-candidates ≡ full, config-override tests, compile tests. Tolerance kept: fused vs bmm atol=1e-6 (measured drift ≤ 6e-8); 1e-3 where one side is fp16 (PrefilterKNN torch backend: measured 1.2e-4) or official fp16.
- Compile: SilverTorch reduce-overhead (3 modes), LiNR V1-V3 × clause/bloom + B=1 bloom V2, OneBit/SimHash reduce-overhead: warm-up, then 2 new queries, each bit-exact to eager, 0 cudagraph_skips.
- Poisoned output + allocation hit count: fused (existing test fixed: its "counts < k" comment was false, k now 48), codesigned (± bloom), codesigned_exact, OPORP full + indirect.
- Cutoffs from constants (regime asserted in-test): `_P_BUCKETS[0]`±1 (fused, both `_impl` and the op, which agree bit for bit), `_N_BUCKETS[0]`±1 (OPORP indirect), P % block ∈ {0,1} for fused/OPORP/both probe scorers, N % block_n ∈ {0,1} for clause_compact.
- Degenerate rows count 0/1 → exact (-1,-inf) tails: OPORP indirect, codesigned, codesigned_exact.
- Identities: OPORP indirect(arange N) ≡ full; bloom op with an all-zero qb ≡ no-bloom op; clause_compact ≡ compact_mask(clause_mask) incl. the -1 tail; row alone ≡ row in batch (fused, both scorers, OPORP); item permutation (fused, codesigned, OPORP full) at k = P.
- `tests/compile/test_export_kernel_ref.py`: op-registry gates — the 10 `retrieve::` schemas (≥10 asserted) each have a same-name reference twin with matching arg names/kinds; no mutable args; inputs `torch.equal` after the op and after the twin; `torch.library.opcheck` on clause_compact / bloom_compact.
- `test_official.py`: `PINNED_SCHEMAS` for the 6 REQUIRED_OPS at 21aa35e (gated by require_official).
- `test_boundary.py`: TorchDispatchMode footprint. Feasible at module level only (custom ops are opaque: the recorder sees the `retrieve::` call and its outputs). Triton forwards of SilverTorch + LiNR V1-V4 have no ≥3-d tensor > B·N; the torch backend must violate it (control: it materializes [B,·,C,A_max]).

## Test-side defects found and fixed (not library findings)
- `test_oporp_1bit_match_topk.py::_ref_indices` returned real ids at -inf slots; the op contract says -1. The old set-based helper filtered non-finite slots and hid it. Fixed in the oracle.
- The fused poison test never exercised counts < k (see above).

## Findings for the orchestrator
1. Nothing in retrieve/src turned red.
2. `docs/validation.md` claimed "Triton vs reference, every parity file: bit-exact". That is false for `fused_masked_knn_topk` (tl.sum vs bmm, ≤ 6e-8). Row corrected.
3. `ruff check retrieve evaluation` fails on a pre-existing E501 at `evaluation/bench/records.py:46` (not touched by Q3; belongs to Q2).
4. `assert_ids_equal_up_to_ties` cannot judge a tie run cut by the K boundary. Tests that reorder items run at k = P to avoid it; the helper itself is unchanged (not loosened).
5. The "~1.4 TB/s ≈ 90% of HBM" claim appears nowhere in this repository (probably in the thesis sources). It stays not yet validated until restated against the measured 1754 GB/s.

## Roofline (docs/artifacts/q3/roofline.py → roofline.json)
A100-SXM4-80GB, torch 2.10.0+cu128, triton 3.6.0. 4 GiB D2D `copy_`: **1754 GB/s** read+write (median of 20 windows × 10 copies; 1752-1766) = 86% of the ~2039 spec. Empty Triton kernel back-to-back: **11.7 µs**/launch median (10.3-14.1). sm_mhz 1275-1410 during the copy, 1410 during launches, mem 1593 → `unstable: true`. Recorded in validation.md's library-gates table.

## Docs
docs/system/testing.md (helpers, tolerance convention, compile, registry, footprint, parity-file gates, official schemas), docs/system/kernels.md (the stale OPORP/fused tolerance paragraph), docs/validation.md (library gates section only). Roadmap untouched.
