---
chain: "evaluation-harness-v2"
branch: "main"
parent: "2026-09-06-120000000-c1-c3-authored-and-c4-library-prerequisites.md"
nextStep: "Re-derive the golden against the deterministic k-means library (roadmap eval-queue item 1), then run C4."
created: "2026-09-06T18:00:00Z"
---

# §10 record: WP-0 / roadmap A1, the golden baseline on the old harness, 2026-09-06

A100-SXM4-80GB, driver 570.195.03, CUDA 12.8, torch 2.10.0+cu128, triton 3.6.0, Python 3.11. Branch `dev/a1-golden`; rows carry `extra.commit = 70bafc4`. Runbook `docs/artifacts/evaluation-harness-v2/a1_golden_run.sh`, stages `golden step4 step7`. Golden JSONs and logs: `evaluation/golden/`.

Clocks not locked (this container cannot: `nvidia-smi -lgc 1410` denied, no sudo): `clocks.sm` sampled every 30 s into `evaluation/golden/_logs/clocks.csv`: 1140 MHz under load (the application-clock default), 210 MHz idle, 28-31 °C. Quality unaffected; latency not clock-controlled.

## Golden cells, 11/11 green
99 rows (ks 100 / 500 / 1000 x bs 1 / 8 / 16). 35.9 min total, 2.5-4.6 min per cell (fresh process each; first pays the oracle build, 62 s for goodreads `c0_genre`). `users_limit 10000`; goodreads keeps 9,859 of 10,000 on `c0_genre`, arxiv 10,000.

| cell | k | recall@k | ndcg@k | median_ms bs=1 | bs=8 | bs=16 |
|---|---|---|---|---|---|---|
| arxiv c0_maincat silvertorch-triton | 100 | 0.884007 | 0.913704 | 0.1868 | 0.3097 | 0.4701 |
| same | 500 | 0.847281 | 0.875401 | 0.2012 | 0.3197 | 0.4790 |
| same | 1000 | 0.819665 | 0.848330 | 0.2020 | 0.3198 | 0.4801 |
| goodreads c0_genre linr_v1_filter_mask-torch | 100 | 0.999695 | 0.999781 | 0.4340 | 0.9449 | 1.6256 |
| same | 500 | 0.999657 | 0.999729 | 0.4506 | 0.9574 | 1.6389 |
| same | 1000 | 0.999627 | 0.999696 | 0.4500 | 0.9576 | 1.6382 |
| linr_v1_filter_mask-triton | 100 | 0.999695 | 0.999781 | 0.4600 | 1.0907 | 1.9004 |
| same | 500 | 0.999657 | 0.999729 | 0.4756 | 1.1021 | 1.9098 |
| same | 1000 | 0.999627 | 0.999696 | 0.5475 | 1.1028 | 1.9136 |
| linr_v2-torch | 100 | 0.999695 | 0.999781 | 0.9320 | 4.6478 | 9.1234 |
| same | 500 | 0.999657 | 0.999729 | 0.9088 | 4.6556 | 9.1410 |
| same | 1000 | 0.999627 | 0.999696 | 0.8409 | 4.6580 | 9.1373 |
| linr_v2-triton | 100 | 0.999279 | 0.999483 | 0.3679 | 1.6625 | 3.2321 |
| same | 500 | 0.999372 | 0.999503 | 0.4577 | 1.6842 | 3.2532 |
| same | 1000 | 0.999388 | 0.999501 | 0.4602 | 1.6953 | 3.2561 |
| linr_v3-torch | 100 | 0.876957 | 0.909106 | 0.4742 | 1.8755 | 3.4024 |
| same | 500 | 0.724322 | 0.774934 | 0.5834 | 1.8952 | 3.4184 |
| same | 1000 | 0.617541 | 0.676164 | 0.5791 | 1.8919 | 3.4193 |
| linr_v3-triton | 100 | 0.876978 | 0.909125 | 0.4347 | 1.3807 | 2.4714 |
| same | 500 | 0.724367 | 0.774973 | 0.5496 | 1.4048 | 2.4867 |
| same | 1000 | 0.617544 | 0.676167 | 0.4720 | 1.4036 | 2.4864 |
| linr_v4-torch | 100 | 0.982054 | 0.987055 | 0.5993 | 1.1424 | 1.8444 |
| same | 500 | 0.986094 | 0.988973 | 0.6142 | 1.1539 | 1.8551 |
| same | 1000 | 0.987316 | 0.989631 | 0.7399 | 1.1551 | 1.8556 |
| linr_v4-triton | 100 | 0.982054 | 0.987055 | 0.6273 | 1.3012 | 2.5555 |
| same | 500 | 0.986094 | 0.988973 | 0.6470 | 1.3123 | 2.1000 |
| same | 1000 | 0.987316 | 0.989631 | 0.6462 | 1.3135 | 2.1002 |
| silvertorch-torch | 100 | 0.912643 | 0.935841 | 0.6128 | 3.4215 | 6.8018 |
| same | 500 | 0.847044 | 0.876175 | 0.6074 | 3.4368 | 6.8182 |
| same | 1000 | 0.788911 | 0.823259 | 0.6025 | 3.4635 | 6.8771 |
| silvertorch-triton | 100 | 0.912759 | 0.935927 | 0.2107 | 0.4509 | 0.8570 |
| same | 500 | 0.847199 | 0.876303 | 0.2721 | 0.4686 | 0.8727 |
| same | 1000 | 0.789083 | 0.823403 | 0.2777 | 0.4767 | 0.8859 |

Reading at the time: `linr_v2` ~0.9997 is the tie-order ceiling of the exact filtered top-K; torch and triton agree to ~1e-4 (later shown to be the non-deterministic k-means for SilverTorch, and a real fp16 accumulation defect for `linr_v2`); SilverTorch triton 3.2-7.9x faster than torch at bs=16, recall within 1.2e-4.

## Handoff steps
Step 1 eval CPU tests: pass, 33 passed. Step 4 compile gate: pass, zero graph breaks (the "no new breaks vs main" comparison vacuous; main leg not run). Step 7 orchestrator smoke: pass (killed after the first algo at 250 s, resumed, exit 0, 5/5 JSONs). Step 5: deferred (scripted: `a1_step5_compare.py`; 332 sweep points per side, ~44 min). Step 6: deferred (scripted: `a1_step6_diff.py`).

## Three bugs, none in the plan
1. `users_limit` row count (`0129e25`).
2. Hub datasets are the pre-`3b1b5b3` 1-indexed `[N+1, ...]` artifacts (`df6db40`): goodreads crashed in the oracle (797,085 vs 797,084); arxiv silently wrong (`cos(query, text_emb[target_id])` 0.9891 vs `cos(query, text_emb[target_id - 1])` 0.6248). Diagnostic `a1_check_item_alignment.py`.
3. K3 helpers under inductor (`70bafc4`): `common.clause_pass(...)` dies with `NameError('common is not defined')` when compiled; fixed by importing helpers by name; compiled mask `torch.equal` to eager; `pytest -k "clause or bloom or compact"` 165 passed / 33 skipped.

Operational: killing a cell mid-`torch.save` left a truncated oracle blob and the next run failed with `PytorchStreamReader failed reading zip archive`; an unreadable cache should rebuild, not raise (folded into WP-2; now `atomic_write` + rebuild with a warning).
