---
chain: "refactor-validation-handoff"
branch: "main"
parent: "2026-09-06-120000000-steps-2-3-passed-a1-ran-1-4-7-harness-half-archived.md"
nextStep: "None. Closed; its role is taken by the library parity suite and the harness v2 gates."
created: "2026-09-15T12:00:00Z"
---

# Steps 5 and 6 moot; runbook closed

User steer 2026-09-15: "We don't care about reproducing old results now, we're improving all code and rewriting, then testing and profiling, then running the full evals step by step." The A1 golden stops being a gate and becomes information.

Step 6 diffs this track against `main` and step 5 gates per-kernel tuning on both sides; both exist to show the refactor changed no number. The library has since been re-laid-out (L1, L2), a kernel deliberately changed for determinism (L3), and `main` is ~150 commits behind. Neither step should be run as written. Replaced by the library's own parity suite (bit-exact, 645 tests at that date) and the harness v2 records (evaluation-harness-v2 §11-§12).

Known-stale items carried at close:
- state-dict load post-hook: later delivered (review #4, L2 builders);
- k-means++ init: later delivered (L2, opt-in);
- fused in-kernel top-k: future-work menu;
- `test_quantize.py` uses the SWAR constant `0x5555555555555555` as test data: the one benign extra hit of the "SWAR constants appear only twice" grep.
