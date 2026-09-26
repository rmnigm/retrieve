---
chain: "reproducibility-paper"
branch: "main"
parent: "2026-09-05-120000000-plan-p-venue-gap-audit-and-paper-outline.md"
nextStep: "F2 (needs B3), then F4 (needs D1 and the user's accounts) and F5."
created: "2026-09-15T16:00:00Z"
---

# Part D record: G1 and G9 (roadmap F1 and F3), 2026-09-15

Branch `dev/f1-f3-paper` off `development` @ `81c55d3`, worktree `/workspace/wt/f1f3`; writing only (no venv, GPU or measurement). Merged `5dad4df`.

Steer consequence recorded in the plan (2026-09-15): no claim of equivalence with the pre-v2 harness from the C4 numbers; G9 must say what was and was not compared (the golden re-derived once, v2 matched 9 of 11 cells at <= 7.5e-9, two residuals unresolved: `linr_v4` 7.3e-5, arxiv `silvertorch` 2.0e-6). Three 2026-09-15 findings are paper material once gated: deterministic k-means makes SilverTorch's torch and triton backends agree exactly (was 1.7e-4); the Triton stream compaction was order-nondeterministic (LiNR V2 / V3 irreproducible at 2.0e-6 / 6.8e-5 until L3); `linr_v2`'s backends disagreed by 4.2e-4 recall (L4, then fixed by L5).

## What was written
1. The thesis reframed (its sources are private and not in this repository): all 33 occurrences of QuantizedIVF gone; the algorithm is SilverTorch (Algorithm 1) in prose, headings, captions, labels, tables; the four places claiming it as new work now describe an independent reimplementation from the paper's text. The thesis had not cited the SilverTorch paper at all: an entry for arXiv 2511.14881 (full 32-author list) added and cited at each place plus the BitFunnel and SONG lineage sentences. "Proposed methods" became "implemented" (only LiNR V4 is ours).
2. `docs/paper/reproduction-deviations.md` (G1): naming map; ST-1..ST-12 SilverTorch rows; LN-1..LN-7 LiNR rows; OF-1..OF-8 deviations of Meta's released code from Meta's paper; HB-1..HB-5 harness / protocol rows; D-1..D-7 defects in our own implementation found by reproducing; R-1..R-3 unexplained residuals; a section on what the document does not contain. Every row cites where it was measured; ungated rows say "not yet validated".
3. `docs/paper/provenance-and-disclosure.md` (G9): hardware / software with Meta's pin; which recorded provenance fields may be cited; §4 clock estimators; §5 what was and was not compared; §6 the not-yet-validated list with the roadmap step per line; §7 data provenance incl. the mirror's 1-indexed layout; §8 commands to reproduce the environment.

## Deliberately not claimed
No equivalence between the v2 and pre-v2 harnesses (what C4 closed on is reported as the result; the thesis's tables are declared superseded); no kernel speed claim (A3's counts are host-side counts; B3 / F2 own speed); no post-fix `linr_v2` numbers (D-1 / LN-4 state the fp16-accumulation finding, fix pending L5); no deleted-backend number presented as current (ST-9 quotes the transposed-index record and says the backend is gone); no venue or reviewer reasoning.

## Contradictions found between the records and the thesis
1. The thesis said step 1 clustering uses k-means++; the code default is random init (L D9: every recorded number was taken on it). Thesis corrected; whether init changes quality is unmeasured (D1).
2. The thesis did not cite SilverTorch: fixed.
3. The thesis presented pre-v2-harness tables as results: HB-5 and provenance §5 say so; the paper takes its tables from D1 / D4.

Gates: links 0; nothing outside `docs/` changed. Skipped: F2, F4, F5. The delivered defense talk's sources still spell QuantizedIVF in 22 places and were left alone on purpose (a delivered talk with a committed PDF beside it); CLAUDE.md records that decision.
