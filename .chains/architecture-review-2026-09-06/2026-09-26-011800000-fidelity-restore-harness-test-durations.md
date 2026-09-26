---
chain: "architecture-review-2026-09-06"
branch: "main"
parent: "2026-09-06-160000000-evaluation-harness-review-and-applied-items.md"
nextStep: "None from this note."
created: "2026-09-26T01:18:00Z"
---

# Fidelity restore: the measured harness test durations (review §4.3, §4.6)

Written 2026-09-26 during the docs reorganisation. The review note kept the outcome (82 passed in 206 s, then 94 passed in 99.8 s) but dropped the review's own measurement and where the time went. Restored from `docs/plans/architecture-review-2026-09-06-evaluation.md` §4.3 and the §4.6 appendix.

On the review's box (`CUDA_VISIBLE_DEVICES=""`, torch on CPU): **82 passed in 319.05 s**. Three sinks made up about 290 s:
- `test_cli_run_campaign_and_report`, 109.4 s: two real child processes, each importing torch and `retrieve` from cold.
- Seven separate SilverTorch builds on the torch path, about 130 s in total: `test_wrapper_k_setter_slices[silvertorch-*]` 23.2 / 22.8 / 22.2 s, `test_index_bytes_includes_filter_submodule` 20.3 s, `test_layer_silvertorch_k_not_baked` 17.4 / 10.2 / 9.6 s, `test_silvertorch_query_params_revalidate` 5.1 s. The build was the cost, not the assertions; the LiNR wrappers took 0.7-3 s.
- `test_run.py`, about 43 s, of which `test_failed_cell_is_recorded_and_the_loop_continues` was 29.3 s (two full runs over four cells with V4's `_int_mm` on CPU).

The fix: one session-scoped SilverTorch build per mode, a campaign test with one child, and the failing-cell test narrowed to one sweep. After it, the campaign test took 21 s, `test_failed_cell...` 15 s, and the three session builds 5.8 s in total. The per-layer `k`-slice tests were kept at under 1.5 s each.
